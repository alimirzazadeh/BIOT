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
import mne
import numpy as np
from tqdm import tqdm


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


def build_index(h5_path: str, group_name: str = "recording") -> None:
    """
    Create an index dataset under group: one segment [start_sample, end_sample]
    for the full recording, so downstream code that expects 'index' can use it.
    """
    with h5py.File(h5_path, "a") as f:
        if group_name not in f:
            return
        g = f[group_name]
        # Infer length from first channel dataset (e.g. fp1)
        ch_names = list(g.keys())
        data_like = [k for k in ch_names if k not in ("ch_names", "bad_channels", "index")]
        if not data_like:
            return
        first_ch = data_like[0]
        n_samples = g[first_ch].shape[0]
        if "index" in g:
            del g["index"]
        g.create_dataset("index", data=np.array([[0, n_samples]], dtype=np.int64))


def edf_to_h5_per_channel(
    edf_path: str,
    h5_path: str,
    target_units: str = "uV",
    compression: str = "lzf",
    chunk_sec: float = 5.0,
) -> None:
    """
    Convert one TUEV EDF to one HDF5 file with each channel as a separate dataset.

    - Applies TUEV channel mapping and preprocessing (same as reference).
    - Writes group "recording" with:
      - One dataset per channel: recording/<ch_name> shape (n_t,) float32
      - ch_names: array of channel names (order preserved)
      - bad_channels: array (possibly empty)
      - attrs: fs, units, start_time_iso, source_path
    - Optionally builds index (one segment for full recording).
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
        return
    if signal is None or sfreq is None:
        return

    # signal: (n_ch, n_t)
    n_ch, n_t = signal.shape
    signal = signal.astype(np.float32)
    if target_units == "uV":
        signal *= 1e6

    ch_names_after = raw.ch_names  # order after rename and drop
    str_dt = h5py.string_dtype("utf-8")
    chunk_t = max(1, int(sfreq * chunk_sec))

    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)

    with h5py.File(h5_path, "a") as f:
        gname = "recording"
        g = f.require_group(gname)

        # Remove existing per-channel datasets and metadata so re-run is clean
        for key in list(g.keys()):
            if key in ("ch_names", "bad_channels", "index") or key in TUEV_CHANNEL_ORDER:
                del g[key]

        for i, ch in enumerate(ch_names_after):
            # HDF5 dataset names must be valid; channel names are already safe (e.g. fp1, fz)
            dset = g.create_dataset(
                ch,
                data=signal[i],
                chunks=(min(chunk_t, n_t),),
                compression=compression,
            )
            dset.attrs["channel"] = ch

        g.create_dataset("ch_names", data=np.array(ch_names_after, dtype=str_dt))
        g.create_dataset(
            "bad_channels",
            data=np.array(raw.info.get("bads", []), dtype=str_dt),
        )
        g.attrs["fs"] = sfreq
        g.attrs["units"] = target_units
        meas_date = raw.info.get("meas_date")
        g.attrs["start_time_iso"] = (
            meas_date.isoformat() if hasattr(meas_date, "isoformat") else ""
        )
        g.attrs["source_path"] = os.path.abspath(edf_path)

    build_index(h5_path, gname)


def prepare_tuev(
    data_folder: str,
    save_folder: str,
    use_rec: bool = False,
    compression: str = "lzf",
    chunk_sec: float = 5.0,
) -> None:
    """Discover EDF (or .rec) files under data_folder and write per-channel H5 to save_folder."""
    if use_rec:
        edf_files = find_rec_files(data_folder)
    else:
        edf_files = find_edf_files(data_folder)

    os.makedirs(save_folder, exist_ok=True)
    for edf_file in tqdm(edf_files, desc="TUEV EDF → H5 (per channel)"):
        basename = os.path.basename(edf_file)
        name = basename.replace(".edf", "").replace(".rec", "")
        h5_path = os.path.join(save_folder, name + ".h5")
        edf_to_h5_per_channel(
            edf_file,
            h5_path,
            compression=compression,
            chunk_sec=chunk_sec,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TUEV EDF → HDF5 with one dataset per channel (reference processing)."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=None,
        help="Root folder containing TUEV EDF (or .rec) files",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
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

    if not data_folder or not save_folder:
        parser.error("Provide --data_folder and --save_folder, or use --prepare_tuev")

    comp = None if args.compression == "none" else args.compression
    prepare_tuev(
        data_folder=data_folder,
        save_folder=save_folder,
        use_rec=args.use_rec,
        compression=comp,
        chunk_sec=args.chunk_sec,
    )
    print(f"Done. H5 files written to {save_folder}")


if __name__ == "__main__":
    main()
