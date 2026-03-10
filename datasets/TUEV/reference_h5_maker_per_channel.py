"""
TUEV dataset → HDF5 with one dataset per channel.

Processes EDF files the same way as BIOT/datasets/TUAB/reference_h5_maker.py
(channel mapping, filtering, resampling) but writes each channel as a separate
HDF5 dataset under group "recording", e.g. recording/fp1, recording/fp2, ...
Each channel dataset has shape (n_samples,) and dtype float32.

Usage:
  python reference_h5_maker_per_channel.py --data_folder /path/to/TUEV/data --save_folder /path/to/TUEV/h5
  python reference_h5_maker_per_channel.py --prepare_tuev  # uses default paths
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

def merge_adjacent_segments(df, fs=200):
    """
    Collapse segments that have the same label and adjacent/overlapping time ranges.
    For each (filename, label), sort by start_time and merge intervals where
    next.start_time <= current.end_time. Updates start_time, end_time (and start_idx, end_idx if present).
    """
    if df.empty:
        return df
    rows = []
    for (filename, label), grp in df.groupby(["filename", "label"]):
        grp = grp.sort_values("start_time").reset_index(drop=True)
        start_time = grp.iloc[0]["start_time"]
        end_time = grp.iloc[0]["end_time"]
        extra_cols = {c: grp.iloc[0][c] for c in grp.columns if c not in ("start_time", "end_time", "start_idx", "end_idx", "filename", "label")}
        for _, row in grp.iloc[1:].iterrows():
            if row["start_time"] <= end_time:
                # adjacent or overlapping: extend current segment
                end_time = max(end_time, row["end_time"])
            else:
                # gap: save current and start new
                rec = {"start_time": start_time, "end_time": end_time, "start_idx": int(start_time * fs), "end_idx": int(end_time * fs), "label": label, "filename": filename, **extra_cols}
                rows.append(rec)
                start_time = row["start_time"]
                end_time = row["end_time"]
        rec = {"start_time": start_time, "end_time": end_time, "start_idx": int(start_time * fs), "end_idx": int(end_time * fs), "label": label, "filename": filename, **extra_cols}
        rows.append(rec)
    out = pd.DataFrame(rows)
    # ensure column order matches original where possible
    return out[[c for c in df.columns if c in out.columns]]


def _annotation_files_under(root: str) -> list[tuple[str, str]]:
    """(path, stem) for each annotation file. stem is the EDF-matching name (e.g. 'x' for x.rec or x_rec.edf)."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            base, ext = os.path.splitext(f)
            path = os.path.join(dirpath, f)
            if ext.lower() == ".rec":
                out.append((path, base))
            elif base.endswith("_rec") and ext.lower() == ".edf":
                out.append((path, base[: -len("_rec")]))  # x_rec.edf -> stem x
            elif base.endswith("_events") and ext.lower() in (".edf", ".csv"):
                out.append((path, base[: -len("_events")]))  # x_events.edf -> stem x
            else:
                continue
    return out


def load_tuev_labels(fs=250, data_folder: Optional[str] = None):
    """Load TUEV annotations. Returns DataFrame with filename, start_time, end_time, label (int). Adjacent same-label segments are merged.
    Discovers .rec, *_rec.edf, *_events.edf, *_events.csv under data_folder (or default path)."""
    if data_folder is not None:
        txt_dir = data_folder
    else:
        txt_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEV/data/v2.0.1/edf/'
    annotation_list = _annotation_files_under(txt_dir)
    df_list = []
    for rec_file, stem in annotation_list:
        try:
            event_data = np.genfromtxt(rec_file, delimiter=",")
        except Exception:
            continue
        if event_data.size == 0:
            continue
        if event_data.ndim == 1:
            event_data = event_data.reshape(1, -1)
        records = []
        for row in event_data:
            start_time = float(row[1])
            end_time = float(row[2])
            label = int(row[3])
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            records.append({
                "start_time": start_time,
                "end_time": end_time,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "label": label
            })
        df = pd.DataFrame(records, columns=["start_time", "end_time", "start_idx", "end_idx", "label"])
        df["filename"] = stem
        df_list.append(df)
    if not df_list:
        return pd.DataFrame(columns=["start_time", "end_time", "start_idx", "end_idx", "label", "filename"])
    df = pd.concat(df_list).reset_index(drop=True)
    return merge_adjacent_segments(df, fs=fs)


# Segment logic from process.py BuildEvents: each event window is (start_time - PAD_SEC, end_time + PAD_SEC).
PAD_SEC = 2.0


def _read_event_file(candidate_path: str) -> list[dict]:
    """Read one annotation file (CSV: chan, start_time, end_time, label). Returns events with process.py padding."""
    if not os.path.isfile(candidate_path):
        return []
    try:
        event_data = np.genfromtxt(candidate_path, delimiter=",")
    except Exception:
        return []
    if event_data.size == 0:
        return []
    if event_data.ndim == 1:
        event_data = event_data.reshape(1, -1)
    events = []
    for row in event_data:
        start_time = float(row[1])
        end_time = float(row[2])
        label = int(row[3])
        start_sec = start_time - PAD_SEC
        end_sec = end_time + PAD_SEC
        if start_sec < end_sec:
            events.append({"start_sec": start_sec, "end_sec": end_sec, "label": label})
    return events


def read_events_for_edf(edf_path: str) -> list[dict]:
    """
    Read annotation file next to the EDF (same CSV format as .rec): chan, start_time, end_time, label.
    Tries, in order: {stem}.rec, {stem}_rec.edf, {stem}_events.edf, {stem}_events.csv.
    Uses process.py logic: segment = (start_time - PAD_SEC, end_time + PAD_SEC).
    Returns list of {"start_sec", "end_sec", "label"}.
    """
    dirpath = os.path.dirname(edf_path)
    stem = os.path.basename(edf_path).replace(".edf", "").replace(".EDF", "")
    candidates = [
        os.path.join(dirpath, stem + ".rec"),
        os.path.join(dirpath, stem + "_rec.edf"),
        os.path.join(dirpath, stem + "_events.edf"),
        os.path.join(dirpath, stem + "_events.csv"),
    ]
    for rec_path in candidates:
        events = _read_event_file(rec_path)
        if events:
            return events
    return []


# Channel names we keep after mapping (same logic as reference CHANNEL_ORDER for TUEV).
# Must match the values in TUEV_CHANNEL_MAPPING below.
TUEV_CHANNEL_ORDER = [
    "fp1", "fp2", "f3", "f4", "c3", "c4", "p3", "p4", "o1", "o2",
    "f7", "f8", "t3", "t4", "t5", "t6", "a1", "a2", "fz", "cz", "pz",
]

# TUEV EDF channel names → canonical lowercase names (from reference_h5_maker.py).
TUEV_CHANNEL_MAPPING = {
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


def find_edf_files(root_folder: str) -> list[str]:
    """Recursively find all .edf files under root_folder."""
    edf_files = []
    for dirpath, _dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".edf"):
                edf_files.append(os.path.join(dirpath, filename))
    return edf_files


def find_rec_files(root_folder: str) -> list[str]:
    """Find .rec files and optionally rename to .edf; return paths to .edf."""
    edf_files = []
    for dirpath, _dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".rec"):
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, filename.replace(".rec", ".edf"))
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                edf_files.append(new_path)
            elif filename.lower().endswith(".edf"):
                edf_files.append(os.path.join(dirpath, filename))
    return edf_files


def preprocess_data(raw: mne.io.Raw) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Same pipeline as reference_h5_maker.preprocess_data:
    - Drop channels not in TUEV_CHANNEL_ORDER
    - Filter 0.1–75 Hz, notch at 60 Hz, resample to 200 Hz.
    Returns (signal shape (n_ch, n_t), sfreq) or (None, None) on failure.
    """
    fs = float(raw.info["sfreq"])
    target_fs = 200.0

    if len(raw.info.get("bads", [])) > 0:
        print(f"Bads: {raw.info['bads']}")
        return None, None

    channel_names = raw.ch_names
    if len(channel_names) == 0:
        print(f"No channels in {raw.filenames[0]}")
        return None, None

    unnecessary = [ch for ch in channel_names if ch not in TUEV_CHANNEL_ORDER]
    if unnecessary:
        raw.drop_channels(unnecessary)

    raw.load_data()
    raw.filter(l_freq=0.1, h_freq=75, method="iir")
    raw.notch_filter(np.arange(60, fs / 2, 60))
    if fs != target_fs:
        raw.resample(target_fs)

    signal = raw.get_data()  # (n_ch, n_t)
    return signal, target_fs


def build_index(h5_path: str, group_name: str = "recording", n_samples: Optional[int] = None) -> None:
    """
    Create an index dataset under group: one segment [0, n_samples].
    If n_samples is None, infer from first channel dataset.
    """
    with h5py.File(h5_path, "a") as f:
        if group_name not in f:
            return
        g = f[group_name]
        ch_names = list(g.keys())
        data_like = [k for k in ch_names if k not in ("ch_names", "bad_channels", "index")]
        if not data_like and n_samples is None:
            return
        if n_samples is None:
            n_samples = g[data_like[0]].shape[0]
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
    sfreq: float = 200.0,
) -> None:
    """
    Write a single 5-second segment (signal_slice shape (n_ch, 1000)) to one HDF5 file.
    """
    n_ch, n_t = signal_slice.shape
    signal_slice = signal_slice.astype(np.float32)
    if target_units == "uV":
        signal_slice *= 1e6
    str_dt = h5py.string_dtype("utf-8")
    chunk_t = min(n_t, max(1, int(sfreq * 5.0)))
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        gname = "recording"
        g = f.create_group(gname)
        for i, ch in enumerate(ch_names):
            dset = g.create_dataset(
                ch,
                data=signal_slice[i],
                chunks=(chunk_t,),
                compression=compression,
            )
            dset.attrs["channel"] = ch
        g.create_dataset("ch_names", data=np.array(ch_names, dtype=str_dt))
        g.create_dataset(
            "bad_channels",
            data=np.array(raw.info.get("bads", []), dtype=str_dt),
        )
        g.attrs["fs"] = sfreq
        g.attrs["units"] = target_units
        g.attrs["label"] = label
        meas_date = raw.info.get("meas_date")
        g.attrs["start_time_iso"] = (
            meas_date.isoformat() if hasattr(meas_date, "isoformat") else ""
        )
        g.attrs["source_path"] = os.path.abspath(edf_path)
    build_index(h5_path, gname, n_samples=n_t)


def edf_to_segment_h5s(
    edf_path: str,
    save_folder: str,
    target_units: str = "uV",
    compression: str = "lzf",
    chunk_sec: float = 5.0,
    segment_sec: float = 5.0,
) -> int:
    """
    Convert one TUEV EDF to many HDF5 files, one per 5-second segment.
    Segments are defined by process.py logic: read .rec next to EDF; for each event
    use (start_time - PAD_SEC, end_time + PAD_SEC); slice into non-overlapping
    5-second windows and write one H5 per window.
    Returns the number of H5 files written.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    existing = set(raw.ch_names)
    safe_mapping = {k: v for k, v in TUEV_CHANNEL_MAPPING.items() if k in existing}
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
    # Reorder to TUEV_CHANNEL_ORDER so verify and downstream code see consistent order
    if set(ch_names_after) != set(TUEV_CHANNEL_ORDER):
        return 0
    order_idx = [ch_names_after.index(ch) for ch in TUEV_CHANNEL_ORDER]
    signal = signal[order_idx]
    ch_names_after = list(TUEV_CHANNEL_ORDER)

    n_samples_per_segment = int(sfreq * segment_sec)  # 1000 at 200 Hz
    events = read_events_for_edf(edf_path)
    if not events:
        return 0

    edf_stem = os.path.basename(edf_path).replace(".edf", "").replace(".EDF", "").replace(".rec", "")
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
                slice_signal,
                h5_path,
                ch_names_after,
                raw,
                edf_path,
                label,
                target_units=target_units,
                compression=compression,
                sfreq=sfreq,
            )
            written += 1
            chunk_idx += 1
    return written


EXPECTED_FS = 200
EXPECTED_DURATION_SEC = 5.0
EXPECTED_N_SAMPLES = int(EXPECTED_FS * EXPECTED_DURATION_SEC)  # 1000


def _parse_segment_h5_stem(stem: str) -> tuple[str, int]:
    """Parse stem like 'edfname_evt0_chunk0_label1' -> (edfname, 1)."""
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
    """
    Verify each .h5 in save_folder:
    - Length: 5 seconds at 200 Hz (1000 samples per channel).
    - Channels: exactly TUEV_CHANNEL_ORDER, named correctly.
    - Label: filename stem parses to (base, label); base exists in load_tuev_labels()
      and label is one of the labels for that file.
    If data_folder is given, labels are loaded from .rec files under it (same as creation).
    Returns True if all pass, False otherwise.
    """
    labels_df = load_tuev_labels(data_folder=data_folder)
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

                # fs
                fs = g.attrs.get("fs", None)
                if fs != EXPECTED_FS:
                    print(f"  FAIL {basename}: fs={fs}, expected {EXPECTED_FS}")
                    all_ok = False
                    continue

                # ch_names
                if "ch_names" in g:
                    ch_names = list(g["ch_names"].asstr()[:])
                else:
                    data_like = [k for k in g.keys() if k in TUEV_CHANNEL_ORDER]
                    ch_names = sorted(data_like) if data_like else []

                if set(ch_names) != set(TUEV_CHANNEL_ORDER) or len(ch_names) != len(TUEV_CHANNEL_ORDER):
                    print(f"  FAIL {basename}: ch_names mismatch. Got {len(ch_names)} channels, expected {len(TUEV_CHANNEL_ORDER)}. Names: {ch_names}")
                    all_ok = False
                    continue

                # length: 5 seconds
                first_ch = TUEV_CHANNEL_ORDER[0]
                if first_ch not in g:
                    print(f"  FAIL {basename}: missing channel dataset {first_ch}")
                    all_ok = False
                    continue
                n_samples = g[first_ch].shape[0]
                if n_samples != EXPECTED_N_SAMPLES:
                    print(f"  FAIL {basename}: n_samples={n_samples}, expected {EXPECTED_N_SAMPLES} (5 sec at {EXPECTED_FS} Hz)")
                    all_ok = False
                    continue

        except Exception as e:
            print(f"  FAIL {basename}: {e}")
            all_ok = False
            continue

        # Label vs dataframe: parse segment stem to get base filename and label
        base_stem, file_label = _parse_segment_h5_stem(stem)
        file_rows = labels_df[labels_df["filename"] == base_stem]
        if file_rows.empty:
            print(f"  FAIL {basename}: base stem '{base_stem}' not found in load_tuev_labels() dataframe")
            all_ok = False
            continue
        expected_labels = file_rows["label"].unique().tolist()
        if file_label < 0:
            print(f"  FAIL {basename}: could not parse label from filename '{stem}'")
            all_ok = False
        elif file_label not in expected_labels:
            print(f"  FAIL {basename}: label {file_label} not in expected labels {expected_labels} for '{base_stem}'")
            all_ok = False

    if all_ok:
        print(f"Verify OK: all {len(h5_files)} files passed.")
    return all_ok


def prepare_tuev(
    data_folder: str,
    save_folder: str,
    use_rec: bool = False,
    compression: str = "lzf",
    chunk_sec: float = 5.0,
    debug: bool = False,
) -> None:
    """Discover EDF (or .rec) files under data_folder and write per-channel H5 to save_folder."""
    if use_rec:
        edf_files = find_rec_files(data_folder)
    else:
        edf_files = find_edf_files(data_folder)

    if debug:
        edf_files = edf_files[:10]
        print(f"Debug mode: processing only {len(edf_files)} files.")

    os.makedirs(save_folder, exist_ok=True)
    total_h5 = 0
    no_events_count = 0
    for edf_file in tqdm(edf_files, desc="TUEV EDF → H5 (per channel)"):
        n = edf_to_segment_h5s(
            edf_file,
            save_folder,
            compression=compression,
            chunk_sec=chunk_sec,
            segment_sec=5.0,
        )
        total_h5 += n
        if n == 0:
            no_events_count += 1
    if total_h5:
        print(f"Wrote {total_h5} segment H5 files to {save_folder}")
    elif no_events_count:
        print(
            f"No H5 files written: none of the {len(edf_files)} EDF(s) had annotation data. "
            "Per EDF we look for (in same directory): {stem}.rec, {stem}_rec.edf, {stem}_events.edf, {stem}_events.csv "
            "(CSV format: chan, start_time, end_time, label). If you renamed .rec → .edf, use _rec.edf or _events.csv for annotations."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TUEV EDF → HDF5 with one dataset per channel (reference processing)."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default='/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEV/data/',
        help="Root folder containing TUEV EDF (or .rec) files",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default='/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEV/h5/',
        help="Folder where to write .h5 files",
    )
    parser.add_argument(
        "--prepare_tuev",
        action="store_true",
        help="Use default TUEV paths: data_folder and save_folder under EXTERNAL_DATASETS/TUEV",
    )
    parser.add_argument(
        "--use_rec",
        action="store_true",
        help="Also search for .rec files and treat as EDF",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="lzf",
        choices=("lzf", "gzip", "none"),
        help="HDF5 compression for channel datasets",
    )
    parser.add_argument(
        "--chunk_sec",
        type=float,
        default=5.0,
        help="Chunk size in seconds for each channel dataset",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the first 10 files (for testing)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After creating (or alone) verify each H5: 5s length, channels, and label vs load_tuev_labels",
    )
    args = parser.parse_args()

    if args.prepare_tuev:
        # Default paths similar to reference_h5_maker.py
        base = os.environ.get(
            "EXTERNAL_DATASETS",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "EXTERNAL_DATASETS"),
        )
        data_folder = os.path.join(base, "TUEV", "data")
        save_folder = os.path.join(base, "TUEV", "h5")
    else:
        data_folder = args.data_folder
        save_folder = args.save_folder

    if not args.verify and (not data_folder or not save_folder):
        parser.error("Provide --data_folder and --save_folder, or use --prepare_tuev (or run with --verify only)")

    comp = None if args.compression == "none" else args.compression

    if not args.verify:
        prepare_tuev(
            data_folder=data_folder,
            save_folder=save_folder,
            use_rec=args.use_rec,
            compression=comp,
            chunk_sec=args.chunk_sec,
            debug=args.debug,
        )
        print(f"Done. H5 files written to {save_folder}")

    if args.verify:
        ok = verify_h5_files(save_folder, data_folder=data_folder)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
