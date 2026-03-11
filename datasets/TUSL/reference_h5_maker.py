"""
TUSL dataset → HDF5 segment files.

Similar to TUEV/reference_h5_maker_per_channel.py: finds EDFs under data_folder,
pairs each with its .tse_agg label file (same stem, same directory), and writes
one H5 per 5-second segment. recording/ contains: data (n_t, n_ch) i.e. (sequence, channels), ch_names,
bad_channels, index. fs=250 Hz.

.tse_agg format (space-separated):
  version = tse_v1.0.0
  <blank>
  start_time end_time label_string confidence
  e.g. 98.0000 108.0000 seiz 1.0000

Usage:
  python reference_h5_maker.py --data_folder /path/to/TUSL/data --save_folder /path/to/TUSL/h5
  python reference_h5_maker.py --debug --verify
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import h5py
import pandas as pd
import mne
import numpy as np
from tqdm import tqdm

# Label string in .tse_agg -> integer (for filename and H5 attrs)
TUSL_LABEL_MAP = {
    "bckg": 0,
    "seiz": 1,
    "slow": 2,
}

# Same channel set as TUEV (TUH standard)
TUSL_CHANNEL_ORDER = [
    "fp1", "fp2", "f3", "f4", "c3", "c4", "p3", "p4", "o1", "o2",
    "f7", "f8", "t3", "t4", "t5", "t6", "a1", "a2", "fz", "cz", "pz",
]

TUSL_CHANNEL_MAPPING = {
    "EEG FP1-REF": "fp1",
    "EEG FP2-REF": "fp2",
    "EEG F3-REF": "f3",
    "EEG F4-REF": "f4",
    "EEG C3-REF": "c3",
    "EEG C4-REF": "c4",
    "EEG P3-REF": "p3",
    "EEG P4-REF": "p4",
    "EEG O1-REF": "o1",
    "EEG O2-REF": "o2",
    "EEG F7-REF": "f7",
    "EEG F8-REF": "f8",
    "EEG T3-REF": "t3",
    "EEG T4-REF": "t4",
    "EEG T5-REF": "t5",
    "EEG T6-REF": "t6",
    "EEG A1-REF": "a1",
    "EEG A2-REF": "a2",
    "EEG FZ-REF": "fz",
    "EEG CZ-REF": "cz",
    "EEG PZ-REF": "pz",
}

TUSL_FS = 250.0
SEGMENT_SEC = 5.0
EXPECTED_N_SAMPLES = int(TUSL_FS * SEGMENT_SEC)  # 1250


def _label_str_to_int(s: str) -> int:
    """Map .tse_agg label string to int. Unknown -> 0."""
    s = (s or "").strip().lower()
    return TUSL_LABEL_MAP.get(s, 0)


def read_tse_agg(tse_path: str) -> list[dict]:
    """
    Parse .tse_agg file. Returns list of {"start_sec", "end_sec", "label"}.
    Format: first line version, blank, then "start end label_str confidence".
    """
    if not os.path.isfile(tse_path):
        return []
    events = []
    with open(tse_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                start_sec = float(parts[0])
                end_sec = float(parts[1])
                label_str = parts[2]
                if start_sec >= end_sec:
                    continue
                label = _label_str_to_int(label_str)
                events.append({"start_sec": start_sec, "end_sec": end_sec, "label": label})
            except (ValueError, IndexError):
                continue
    return events


def read_events_for_edf(edf_path: str) -> list[dict]:
    """
    Find {stem}.tse_agg next to the EDF and return segments.
    Returns list of {"start_sec", "end_sec", "label"} (no padding).
    """
    dirpath = os.path.dirname(edf_path)
    stem = os.path.basename(edf_path).replace(".edf", "").replace(".EDF", "")
    tse_path = os.path.join(dirpath, stem + ".tse_agg")
    return read_tse_agg(tse_path)


def find_edf_files(root_folder: str) -> list[str]:
    """Recursively find all .edf files under root_folder."""
    edf_files = []
    for dirpath, _dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".edf"):
                edf_files.append(os.path.join(dirpath, filename))
    return edf_files


def preprocess_data(raw: mne.io.Raw) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Drop channels not in TUSL_CHANNEL_ORDER, filter 0.1–75 Hz, notch 60 Hz, resample to TUSL_FS.
    Returns (signal (n_ch, n_t), sfreq) or (None, None).
    """
    fs = float(raw.info["sfreq"])
    target_fs = TUSL_FS
    if len(raw.info.get("bads", [])) > 0:
        return None, None
    channel_names = raw.ch_names
    if len(channel_names) == 0:
        return None, None
    unnecessary = [ch for ch in channel_names if ch not in TUSL_CHANNEL_ORDER]
    if unnecessary:
        raw.drop_channels(unnecessary)
    raw.load_data()
    raw.filter(l_freq=0.1, h_freq=75, method="iir")
    raw.notch_filter(np.arange(60, fs / 2, 60))
    if fs != target_fs:
        raw.resample(target_fs)
    signal = raw.get_data()
    return signal, target_fs


def build_index(h5_path: str, group_name: str = "recording", n_samples: Optional[int] = None) -> None:
    """Infer n_samples from data shape (n_t, n_ch) -> shape[0]."""
    with h5py.File(h5_path, "a") as f:
        if group_name not in f:
            return
        g = f[group_name]
        if n_samples is None and "data" in g:
            n_samples = g["data"].shape[0]
        if n_samples is None:
            return
        if "index" in g:
            del g["index"]
        g.create_dataset("index", data=np.array([[0, n_samples]], dtype=np.int64))


def write_one_segment_h5(
    signal_slice: np.ndarray,
    h5_path: str,
    ch_names: list[str],
    raw: mne.io.Raw,
    edf_path: str,
    label: int,
    target_units: str = "uV",
    compression: str = "lzf",
    sfreq: float = TUSL_FS,
) -> None:
    """Write one 5-second segment to H5: recording/data (n_t, n_ch) i.e. (sequence, channels), ch_names, bad_channels, index."""
    n_ch, n_t = signal_slice.shape
    signal_slice = signal_slice.astype(np.float32)
    if target_units == "uV":
        signal_slice *= 1e6
    data_stored = np.ascontiguousarray(signal_slice.T)  # (n_t, n_ch)
    str_dt = h5py.string_dtype("utf-8")
    chunk_t = min(n_t, max(1, int(sfreq * 5.0)))
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("recording")
        g.create_dataset("data", data=data_stored, chunks=(chunk_t, n_ch), compression=compression)
        g.create_dataset("ch_names", data=np.array(ch_names, dtype=str_dt))
        g.create_dataset("bad_channels", data=np.array(raw.info.get("bads", []), dtype=str_dt))
        g.attrs["fs"] = sfreq
        g.attrs["units"] = target_units
        g.attrs["label"] = label
        meas_date = raw.info.get("meas_date")
        g.attrs["start_time_iso"] = (
            meas_date.isoformat() if hasattr(meas_date, "isoformat") else ""
        )
        g.attrs["source_path"] = os.path.abspath(edf_path)
    build_index(h5_path, "recording", n_samples=n_t)


def edf_to_segment_h5s(
    edf_path: str,
    save_folder: str,
    compression: str = "lzf",
    segment_sec: float = SEGMENT_SEC,
) -> int:
    """
    Load EDF, preprocess, read .tse_agg segments; write one H5 per 5-second chunk per segment.
    Returns number of H5 files written.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    existing = set(raw.ch_names)
    safe_mapping = {k: v for k, v in TUSL_CHANNEL_MAPPING.items() if k in existing}
    if safe_mapping:
        raw.rename_channels(safe_mapping)
    try:
        signal, sfreq = preprocess_data(raw)
    except Exception as e:
        print(f"Preprocess failed for {edf_path}: {e}")
        return 0
    if signal is None or sfreq is None:
        return 0

    n_ch, n_t = signal.shape
    ch_names_after = list(raw.ch_names)
    if set(ch_names_after) != set(TUSL_CHANNEL_ORDER):
        return 0
    order_idx = [ch_names_after.index(ch) for ch in TUSL_CHANNEL_ORDER]
    signal = signal[order_idx]
    ch_names_after = list(TUSL_CHANNEL_ORDER)

    n_samples_per_segment = int(sfreq * segment_sec)
    events = read_events_for_edf(edf_path)
    if not events:
        return 0

    edf_stem = os.path.basename(edf_path).replace(".edf", "").replace(".EDF", "")
    written = 0
    for evt_idx, evt in enumerate(events):
        start_idx = int(evt["start_sec"] * sfreq)
        end_idx = int(evt["end_sec"] * sfreq)
        start_idx = max(0, start_idx)
        end_idx = min(n_t, end_idx)
        if end_idx - start_idx < n_samples_per_segment:
            continue
        label = evt["label"]
        chunk_idx = 0
        for seg_start in range(start_idx, end_idx - n_samples_per_segment + 1, n_samples_per_segment):
            seg_end = seg_start + n_samples_per_segment
            slice_signal = signal[:, seg_start:seg_end]
            h5_name = f"{edf_stem}_evt{evt_idx}_chunk{chunk_idx}_label{label}.h5"
            h5_path = os.path.join(save_folder, h5_name)
            write_one_segment_h5(
                slice_signal, h5_path, ch_names_after, raw, edf_path, label,
                compression=compression, sfreq=sfreq,
            )
            written += 1
            chunk_idx += 1
    return written


def load_tusl_labels(data_folder: Optional[str] = None) -> pd.DataFrame:
    """
    Discover all .tse_agg under data_folder and return DataFrame with
    filename (stem), start_time, end_time, label (int).
    """
    if data_folder is None:
        data_folder = "/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUSL/data/v2.0.1/edf"
    rows = []
    for dirpath, _dirnames, filenames in os.walk(data_folder):
        for f in filenames:
            if not f.endswith(".tse_agg"):
                continue
            stem = f[:- len(".tse_agg")]
            path = os.path.join(dirpath, f)
            for evt in read_tse_agg(path):
                rows.append({
                    "filename": stem,
                    "start_time": evt["start_sec"],
                    "end_time": evt["end_sec"],
                    "label": evt["label"],
                })
    if not rows:
        return pd.DataFrame(columns=["filename", "start_time", "end_time", "label"])
    return pd.DataFrame(rows)


def _parse_segment_h5_stem(stem: str) -> tuple[str, int]:
    """Parse stem like 'name_evt0_chunk0_label1' -> (base_stem, label)."""
    if "_label" not in stem:
        return stem, -1
    pre, label_str = stem.rsplit("_label", 1)
    try:
        label = int(label_str)
        base_stem = pre.rsplit("_evt", 1)[0] if "_evt" in pre else pre
        return base_stem, label
    except ValueError:
        return stem, -1


def verify_h5_files(save_folder: str, data_folder: Optional[str] = None) -> bool:
    """Verify each H5: 5s @ 250 Hz, recording/data, ch_names set, label in filename vs load_tusl_labels."""
    labels_df = load_tusl_labels(data_folder=data_folder)
    h5_files = []
    for dirpath, _dirnames, filenames in os.walk(save_folder):
        for f in filenames:
            if f.lower().endswith(".h5"):
                h5_files.append(os.path.join(dirpath, f))
    if not h5_files:
        print(f"No .h5 files found in {save_folder}")
        return False

    all_ok = True
    for h5_path in tqdm(h5_files, desc="Verify H5"):
        basename = os.path.basename(h5_path)
        stem = basename.replace(".h5", "")
        try:
            with h5py.File(h5_path, "r") as f:
                if "recording" not in f:
                    print(f"  FAIL {basename}: missing 'recording' group")
                    all_ok = False
                    continue
                g = f["recording"]
                if g.attrs.get("fs") != TUSL_FS:
                    print(f"  FAIL {basename}: fs={g.attrs.get('fs')}, expected {TUSL_FS}")
                    all_ok = False
                    continue
                if "data" not in g:
                    print(f"  FAIL {basename}: missing 'recording/data'")
                    all_ok = False
                    continue
                data_dset = g["data"]
                if data_dset.ndim != 2:
                    print(f"  FAIL {basename}: data ndim={data_dset.ndim}")
                    all_ok = False
                    continue
                n_samples = data_dset.shape[0]
                if n_samples != EXPECTED_N_SAMPLES:
                    print(f"  FAIL {basename}: n_samples={n_samples}, expected {EXPECTED_N_SAMPLES}")
                    all_ok = False
                    continue
                ch_names = list(g["ch_names"].asstr()[:]) if "ch_names" in g else []
                if set(ch_names) != set(TUSL_CHANNEL_ORDER) or len(ch_names) != len(TUSL_CHANNEL_ORDER):
                    print(f"  FAIL {basename}: ch_names mismatch")
                    all_ok = False
                    continue
        except Exception as e:
            print(f"  FAIL {basename}: {e}")
            all_ok = False
            continue

        base_stem, file_label = _parse_segment_h5_stem(stem)
        file_rows = labels_df[labels_df["filename"] == base_stem]
        if file_rows.empty:
            print(f"  FAIL {basename}: base '{base_stem}' not in load_tusl_labels()")
            all_ok = False
            continue
        expected_labels = file_rows["label"].unique().tolist()
        if file_label < 0 or file_label not in expected_labels:
            if file_label >= 0:
                print(f"  FAIL {basename}: label {file_label} not in {expected_labels} for '{base_stem}'")
            all_ok = False

    if all_ok:
        print(f"Verify OK: all {len(h5_files)} files passed.")
    return all_ok


def prepare_tusl(
    data_folder: str,
    save_folder: str,
    compression: str = "lzf",
    debug: bool = False,
) -> None:
    """Discover EDFs under data_folder, write segment H5s to save_folder."""
    edf_files = find_edf_files(data_folder)
    if debug:
        edf_files = edf_files[:10]
        print(f"Debug mode: processing only {len(edf_files)} files.")
    os.makedirs(save_folder, exist_ok=True)
    total_h5 = 0
    no_events = 0
    for edf_file in tqdm(edf_files, desc="TUSL EDF → H5"):
        n = edf_to_segment_h5s(edf_file, save_folder, compression=compression)
        total_h5 += n
        if n == 0:
            no_events += 1
    if total_h5:
        print(f"Wrote {total_h5} segment H5 files to {save_folder}")
    elif no_events:
        print(
            f"No H5 files written: none of the {len(edf_files)} EDF(s) had a .tse_agg in the same directory."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="TUSL EDF → HDF5 segments (recording/data, 250 Hz).")
    parser.add_argument("--data_folder", type=str, default="/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUSL/data/")
    parser.add_argument("--save_folder", type=str, default="/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUSL/h5/")
    parser.add_argument("--compression", type=str, default="lzf", choices=("lzf", "gzip", "none"))
    parser.add_argument("--debug", action="store_true", help="Process only first 10 EDFs")
    parser.add_argument("--verify", action="store_true", help="Verify H5s (5s, channels, labels)")
    args = parser.parse_args()

    data_folder = args.data_folder
    save_folder = args.save_folder
    if not args.verify and (not data_folder or not save_folder):
        parser.error("Provide --data_folder and --save_folder (or run with --verify only)")

    comp = None if args.compression == "none" else args.compression

    if not args.verify:
        prepare_tusl(data_folder=data_folder, save_folder=save_folder, compression=comp, debug=args.debug)
        print(f"Done. H5 files in {save_folder}")

    if args.verify:
        ok = verify_h5_files(save_folder, data_folder=data_folder)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
