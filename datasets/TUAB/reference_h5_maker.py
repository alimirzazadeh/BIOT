"""
This script will run a battery of external datasets with linear probing. Starts with REVE-B model. 
Trying on the following datasets: Mumtaz, TUAB, TUEV, ISRUC, HMC, 

Waiting on credentials: TUEP, TUSL
Site is down: ISRUC 
To-do: Mumtaz, TUAB, TUEV

1. Get the training and test splits for each dataset. 
2. Probe REVE and validate AUC 
3. Check performance at patient level as well as different segmentation levels 

assume they are all in h5 format. 

"""

import h5py 
import numpy as np
import pandas as pd
import os 
import argparse 
import torch 
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import sys 
from torch.nn import CrossEntropyLoss
sys.path.append('../rep-learning')
from data.eegtext import EEGTextDataset
from baselines_downstream_internal import load_model, train_model, SafeDataset, DownstreamProbeModel, safe_collate_fn, worker_init_fn
from utils_gpu_tricks import main_parallel
from utils_downstream import parallel_calculate_final_results
from summarize_exp import summarize_exp_parallel
import json 
import time 
import random 
import torch.optim as optim
from ipdb import set_trace as bp
from tqdm import tqdm
from preprocess_eeg_v3 import build_index
from sklearn.model_selection import train_test_split
import mne
from taming.data.spec_utils import CHANNEL_ORDER

REPO_DIR = "../taming-transformers"

## preprocess for Mumtaz
## need a label hunter and a dataset for session based dataset (TUEV)
## switch to long setup for metrics and val dataset 
## need to add implementation for other datasets 
class Object:
    pass

class SegmentDataset(EEGTextDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset_name = dataset_name
        assert dataset_name in ['tuev', 'tusl']
        super().__init__(*args, psg_mode=True, **kwargs)
        if self.dataset_name == 'tuev':
            self.df = self.load_tuev_labels()
        elif self.dataset_name == 'tusl':
            self.df = self.load_tusl_labels()

    def load_tuev_labels(self, ) -> pd.DataFrame:
        txt_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEV/data/v2.0.1/edf/'
        txt_files = [os.path.join(item[0], ii) for item in os.walk(txt_dir) for ii in item[2] if ii.endswith('.rec')]
        df_list = [] 
        for rec_file in txt_files:
            event_data = np.genfromtxt(rec_file, delimiter=",")
            
            if event_data.ndim == 1:
                event_data = event_data.reshape(1, -1)
            
            records = []
            for row in event_data:
                start_time = row[1]
                end_time = row[2]
                label = int(row[3])
                
                start_idx = int(start_time * self.fs)
                end_idx = int(end_time * self.fs)
                
                records.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "label": label
                })
    
            df = pd.DataFrame(records, columns=["start_idx", "end_idx", "label"])
            df['filename'] = rec_file.split('/')[-1].split('.')[0]
            df_list.append(df)
        df = pd.concat(df_list).reset_index(drop=True)
        return df


    def load_tusl_labels(self):
        txt_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUSL/data/v2.0.1/edf/'
        txt_files = [os.path.join(item[0], ii) for item in os.walk(txt_dir) for ii in item[2] if ii.endswith('.tse_agg')]
        df_list = [] 
        for tse_filename in txt_files:
            records = [] 
            with open(tse_filename, "r") as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith("version") or line == "":
                        continue
                    
                    parts = line.split()
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    label = parts[2]
                    confidence = float(parts[3])
                    
                    records.append({
                        "start_idx": int(start_time * self.fs),
                        "end_idx": int(end_time * self.fs),
                        "label": label,
                        "confidence": confidence
                    })
            df = pd.DataFrame(records, columns=["start_idx", "end_idx", "label", "confidence"])
            df['filename'] = tse_filename.split('/')[-1].split('.')[0]
            df_list.append(df)
        df = pd.concat(df_list).reset_index(drop=True)
        return df

    # def __getitem__(self, idx):
        

class PSGDataset(EEGTextDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset_name = dataset_name
        assert dataset_name in ['isruc', 'hmc']
        super().__init__(*args, psg_mode=True, **kwargs)
        if self.dataset_name == 'hmc':
            self.df = self.load_hmc_labels()
        elif self.dataset_name == 'isruc':
            self.df = self.load_isruc_labels()
        else:
            raise NotImplementedError(f'Dataset {self.dataset_name} not implemented')
    
    def load_hmc_labels(self):
        txt_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/HMC/data/physionet.org/files/hmc-sleep-staging/1.1/recordings'
        txt_files = [item for item in os.listdir(txt_dir) if item.endswith('.txt')]
        df_list = [] 
        STAGE_DICT = {' Sleep stage W': 0, ' Sleep stage N1': 1, ' Sleep stage N2': 2, ' Sleep stage N3': 3, ' Sleep stage R': 4}
        for item in txt_files:
            df = pd.read_csv(os.path.join(txt_dir, item))
            df.rename(columns={' Recording onset': 'recording_onset'}, inplace=True)

            df['filename'] = item.split('_')[0]
            df['stage'] = df[' Annotation'].apply(lambda x: STAGE_DICT[x] if x in STAGE_DICT else -1)

            df = df[df['stage'] != -1]
            df = df[['filename', 'recording_onset', 'stage']].copy() 
            df_list.append(df)
        df = pd.concat(df_list).reset_index(drop=True)
        return df
    
    def load_isruc_labels(self):
        txt_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/ISRUC/data/subgroup1/'
        
        txt_files = [os.path.join(item[0], ii) for item in os.walk(txt_dir) for ii in item[2] if ii.endswith('_1.txt')]
        df_list = [] 
        STAGE_DICT = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
        for item in txt_files:
            # txt_file = open(os.path.join(txt_dir, item), 'r')
            ## read all lines as list 
            with open(os.path.join(txt_dir, item), 'r') as f:
                lines = f.readlines()
            lines = [STAGE_DICT[int(line.strip())] for line in lines if line.strip() != '']
            times = [30.0 * i for i in range(len(lines))]
            # stages = [int(line.split(' ')[1]) for line in lines]
            stages = lines
            df = pd.DataFrame({'filename': item.split('/')[-1], 'recording_onset': times, 'stage': stages})
            df_list.append(df)
        df = pd.concat(df_list).reset_index(drop=True)
        return df
        
        
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        filename = item['eeg_file'].split('/')[-1].split('.')[0] + '_1.txt'
        start_idx = item['start_idx']
        if len(start_idx) > 1:
            raise NotImplementedError('Multiple start indices not implemented')
        start_idx = start_idx[0]
        if self.raw_eeg:
            # Raw EEG: parent returns (C, T_samples); time is dimension 1
            num_samples = item['eeg'].shape[1]
            end_idx = start_idx + num_samples
        else:
            # Spectrogram: parent returns (C, F, T); time is dimension -1, use frame indices
            num_frames = item['eeg'].shape[-1]
            end_idx = start_idx + num_frames * self.hop_length_samples

        nearest_30_second_start = start_idx // self.fs
        nearest_30_second_end = end_idx // self.fs

        # ## find the nearest 30 second start / end so the label matches
        # nearest_30_second_start = np.round(start_idx / self.fs / 30) * 30
        # if nearest_30_second_start < start_idx / self.fs:
        #     nearest_30_second_start += 30
        # end_sec = end_idx / self.fs
        # nearest_30_second_end = np.round(end_sec / 30) * 30
        # if nearest_30_second_end > end_sec:
        #     nearest_30_second_end -= 30
        # # Last 30s epoch must end before end_idx (its end = nearest_30_second_end + 30)
        # if nearest_30_second_end + 30 > end_sec:
        #     nearest_30_second_end = int((end_sec - 30) // 30) * 30

        label = self.df[self.df['filename'] == filename]
        label = label[label['recording_onset'] >= nearest_30_second_start]
        label = label[label['recording_onset'] < nearest_30_second_end]
        label_dict = dict(zip(label['recording_onset'], label['stage']))
        assert nearest_30_second_start in label_dict, f'{nearest_30_second_start} not in {label_dict}'
        time_template = np.arange(nearest_30_second_start, nearest_30_second_end, 30)
        label_template = torch.tensor(np.array([label_dict.get(time, -1) for time in time_template]))
        item['stage'] = label_template
        
        return item 

        # if self.raw_eeg:
        #     # Slice in sample space: (C, T_samples), eeg_mask (T,), channel_mask (C, T)
        #     data_start_idx = int(nearest_30_second_start * self.fs - start_idx)
        #     data_end_idx = int(end_idx - nearest_30_second_end * self.fs)
        #     data_start_idx = max(0, min(data_start_idx, num_samples))
        #     data_end_idx = max(0, min(data_end_idx, num_samples))
        #     end_slice = num_samples - data_end_idx
        #     if data_start_idx >= end_slice:
        #         data_start_idx, end_slice = 0, num_samples
        #     item['eeg'] = item['eeg'][:, data_start_idx:end_slice]
        #     item['eeg_mask'] = item['eeg_mask'][data_start_idx:end_slice]
        #     item['channel_mask'] = item['channel_mask'][:, data_start_idx:end_slice]
        #     # Fixed target length: (n_30s_blocks - 2) * 30s, so data and stage always match
        #     len_30s = 30 * self.fs
        #     n_30s = num_samples // len_30s
        #     target_len = max(len_30s, (n_30s - 2) * len_30s)
        #     expected_stage_len = max(1, n_30s - 2)
        # else:
        #     # Slice in frame space: (C, F, T), eeg_mask (T,), channel_mask (C, F, T)

        #     data_start_frame = (int(nearest_30_second_start * self.fs) - start_idx) // self.hop_length_samples
        #     data_end_frame = (int(nearest_30_second_end * self.fs) - start_idx) // self.hop_length_samples
        #     data_start_frame = max(0, min(data_start_frame, num_frames))
        #     data_end_frame = max(0, min(data_end_frame, num_frames))
        #     if data_start_frame >= data_end_frame:
        #         data_start_frame, data_end_frame = 0, num_frames
        #     item['eeg'] = item['eeg'][:, :, data_start_frame:data_end_frame]
        #     item['eeg_mask'] = item['eeg_mask'][data_start_frame:data_end_frame]
        #     item['channel_mask'] = item['channel_mask'][:, :, data_start_frame:data_end_frame]
        #     # Same formula in frames: (n_30s_blocks - 2) * 30s worth of frames
        #     len_30s_frames = int(30 * self.fs / self.hop_length_samples)
        #     n_30s = num_frames // len_30s_frames
        #     target_len = max(len_30s_frames, (n_30s - 2) * len_30s_frames)
        #     expected_stage_len = max(1, n_30s - 2)

        # # Pad or truncate to fixed length so batches stack
        # if self.raw_eeg:
        #     T = item['eeg'].shape[1]
        #     if T >= target_len:
        #         item['eeg'] = item['eeg'][:, :target_len]
        #         item['eeg_mask'] = item['eeg_mask'][:target_len]
        #         item['channel_mask'] = item['channel_mask'][:, :target_len]
        #     else:
        #         pad = target_len - T
        #         item['eeg'] = torch.nn.functional.pad(item['eeg'], (0, pad), value=0)
        #         item['eeg_mask'] = torch.nn.functional.pad(item['eeg_mask'], (0, pad), value=0)
        #         item['channel_mask'] = torch.nn.functional.pad(item['channel_mask'], (0, pad), value=0)
        # else:
        #     T = item['eeg'].shape[-1]
        #     if T >= target_len:
        #         item['eeg'] = item['eeg'][:, :, :target_len]
        #         item['eeg_mask'] = item['eeg_mask'][:target_len]
        #         item['channel_mask'] = item['channel_mask'][:, :, :target_len]
        #     else:
        #         pad = target_len - T
        #         item['eeg'] = torch.nn.functional.pad(item['eeg'], (0, pad), value=0)
        #         item['eeg_mask'] = torch.nn.functional.pad(item['eeg_mask'], (0, pad), value=0)
        #         item['channel_mask'] = torch.nn.functional.pad(item['channel_mask'], (0, pad), value=0)

        # # Stage length must match target_len (expected_stage_len = max(1, n_30s - 2) set in branch above)
        # stage = item['stage']
        # if stage.numel() >= expected_stage_len:
        #     item['stage'] = stage[:expected_stage_len]
        # else:
        #     raise ValueError(
        #         f"Stage length {stage.numel()} < expected {expected_stage_len} (target_len={target_len}); "
        #         "stage should never need padding—check 30s alignment."
        #     )

        # return item 
    

def find_edf_files(root_folder):
    edf_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.edf'):
                edf_files.append(os.path.join(dirpath, filename))
    return edf_files

def find_rec_files(root_folder):
    edf_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.rec'):
                ## rename it to end with .edf 
                os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, filename.replace('.rec', '.edf')))
                edf_files.append(os.path.join(dirpath, filename.replace('.rec', '.edf')))
            elif filename.lower().endswith('.edf'):
                edf_files.append(os.path.join(dirpath, filename))
    return edf_files

def verify_channels(datasets):
    
    for dataset in datasets:
        unique_channel_sets = set()
        folder_path = f'/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/{dataset.upper()}/h5/'
        
        for file in os.listdir(folder_path):
            if file.endswith('.h5'):
                with h5py.File(os.path.join(folder_path, file), 'r') as f:
                    try:
                        channels = [ch.decode('utf-8') for ch in f['recording']['ch_names'][:]]
                        channels = sorted(channels)
                        unique_channel_sets.add(tuple(channels))
                    except:
                        print(f'Error reading {file}')
        print(f'{dataset} has {len(unique_channel_sets)} unique channel sets')
        print(unique_channel_sets)
        print('-'*100)

def prepare_tuep():
    data_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEP/data/'
    edf_files = find_edf_files(data_folder)
    save_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEP/h5/'
    os.makedirs(save_folder, exist_ok=True)
    for edf_file in tqdm(edf_files):
        edf_to_h5(os.path.join(data_folder, edf_file), h5_path = save_folder + edf_file.split('/')[-1].replace('.edf', '.h5'), compression="lzf", chunk_sec=5, dataset='tuep')

def prepare_mumtaz():
    data_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/MUMTAZ/data/'
    edf_files = os.listdir(data_folder)
    save_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/MUMTAZ/h5/'
    for edf_file in tqdm(edf_files):
        if not edf_file.endswith('.edf'):
            continue
        edf_to_h5(os.path.join(data_folder, edf_file), h5_path = save_folder + edf_file.replace('.edf', '.h5'), compression="lzf", chunk_sec=5, dataset='mumtaz')

def prepare_tuev():
    data_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEV/data/'
    edf_files = find_edf_files(data_folder)
    save_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEV/h5/'
    os.makedirs(save_folder, exist_ok=True)
    for edf_file in tqdm(edf_files):
        edf_to_h5(os.path.join(data_folder, edf_file), h5_path = save_folder + edf_file.split('/')[-1].replace('.edf', '.h5'), compression="lzf", chunk_sec=5, dataset='tuev')

def prepare_tusl():
    data_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUSL/data/'
    edf_files = find_edf_files(data_folder)
    save_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUSL/h5/'
    os.makedirs(save_folder, exist_ok=True)
    for edf_file in tqdm(edf_files):
        edf_to_h5(os.path.join(data_folder, edf_file), h5_path = save_folder + edf_file.split('/')[-1].replace('.edf', '.h5'), compression="lzf", chunk_sec=5, dataset='tusl')

def prepare_hmc():
    data_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/HMC/data/'
    edf_files = find_edf_files(data_folder)
    save_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/HMC/h5/'
    os.makedirs(save_folder, exist_ok=True)
    for edf_file in tqdm(edf_files):
        if os.path.exists(save_folder + edf_file.split('/')[-1].replace('.edf', '.h5')):
            continue
        edf_to_h5(os.path.join(data_folder, edf_file), h5_path = save_folder + edf_file.split('/')[-1].replace('.edf', '.h5'), compression="lzf", chunk_sec=5, dataset='hmc')

def prepare_isruc():
    data_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/ISRUC/data/subgroup1/'
    edf_files = find_rec_files(data_folder)
    save_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/ISRUC/h5/'
    os.makedirs(save_folder, exist_ok=True)
    for edf_file in tqdm(edf_files):
        edf_to_h5(os.path.join(data_folder, edf_file), h5_path = save_folder + edf_file.split('/')[-1].replace('.edf', '.h5'), compression="lzf", chunk_sec=5, dataset='isruc')

def prepare_tuab():
    data_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUAB/data/'
    edf_files = find_edf_files(data_folder)
    save_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUAB/h5/'
    os.makedirs(save_folder, exist_ok=True)
    for edf_file in tqdm(edf_files):
        edf_to_h5(os.path.join(data_folder, edf_file), h5_path = save_folder + edf_file.split('/')[-1].replace('.edf', '.h5'), compression="lzf", chunk_sec=5, dataset='tuab')

def prepare_cognition():
    data_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/COGNITION/data/'
    edf_files = find_edf_files(data_folder)
    save_folder = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/COGNITION/h5/'
    os.makedirs(save_folder, exist_ok=True)
    for edf_file in tqdm(edf_files):
        edf_to_h5(os.path.join(data_folder, edf_file), h5_path = save_folder + edf_file.split('/')[-1].replace('.edf', '.h5'), compression="lzf", chunk_sec=5, dataset='cognition')

def preprocess_data(raw):
    fs = float(raw.info["sfreq"])
    target_fs = 200    
    
    if len(raw.info['bads']) > 0:
        print(f'Bads: {raw.info["bads"]}')
        raise ValueError(f'Bads: {raw.info["bads"]}')
    
    ## drop channels not in CHANNEL_ORDER
    channel_names = raw.ch_names
    if len(channel_names) == 0:
        print(f'No channels in {raw.filenames[0]}')
        return None, None
    unnecessary_channels = [channel for channel in channel_names if channel not in CHANNEL_ORDER]
    try:
        raw.drop_channels(unnecessary_channels)
    except:
        bp() 
    
    raw.load_data() 
    
    raw.filter(l_freq=0.1, h_freq=75, method='iir')
    raw.notch_filter(np.arange(60, fs/2, 60))
    if fs != target_fs:
        raw.resample(target_fs)
    
    signal = raw.get_data()     

    return signal, target_fs

def edf_to_h5(edf_path, h5_path, dataset='mumtaz', group=None, target_units="uV",
              compression="lzf", compression_opts=None, shuffle=None, chunk_sec=5):
   
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

    ## ['EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1', 'EEG 23A-23R', 'EEG 24A-24R']
    if dataset == 'mumtaz':
        mapping = {'EEG Fp1-LE': 'fp1', 'EEG F3-LE': 'f3', 'EEG C3-LE': 'c3', 'EEG P3-LE': 'p3', 'EEG O1-LE': 'o1', 'EEG F7-LE': 'f7', 'EEG T3-LE': 't3', 'EEG T5-LE': 't5', 'EEG Fz-LE': 'fz', 'EEG Fp2-LE': 'fp2', 'EEG F4-LE': 'f4', 'EEG C4-LE': 'c4', 'EEG P4-LE': 'p4', 'EEG O2-LE': 'o2', 'EEG F8-LE': 'f8', 'EEG T4-LE': 't4', 'EEG T6-LE': 't6', 'EEG Cz-LE': 'cz', 'EEG Pz-LE': 'pz'}
        existing = set(raw.ch_names)
        safe_mapping = {k: v for k, v in mapping.items() if k in existing}
        if safe_mapping:
            raw.rename_channels(safe_mapping)
    elif dataset == 'tuep':
        # ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG A1-LE', 'EEG A2-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE', 'EEG OZ-LE', 'EEG PG1-LE', 'EEG PG2-LE', 'EEG EKG-LE', 'EEG SP2-LE', 'EEG SP1-LE', 'EEG 28-LE', 'EEG 29-LE', 'EEG 30-LE', 'EEG T1-LE', 'EEG T2-LE', 'PHOTIC PH']
        # ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG 26-REF', 'EEG 27-REF', 'EEG 28-REF', 'EEG 29-REF', 'EEG 30-REF', 'EEG T1-REF', 'EEG T2-REF', 'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR']
        mapping = {'EEG Fp1-LE': 'fp1', 'EEG F3-LE': 'f3', 'EEG C3-LE': 'c3', 'EEG P3-LE': 'p3', 'EEG O1-LE': 'o1', 'EEG F7-LE': 'f7', 'EEG T3-LE': 't3', 'EEG T5-LE': 't5', 'EEG Fz-LE': 'fz', 'EEG Fp2-LE': 'fp2', 'EEG F4-LE': 'f4', 'EEG C4-LE': 'c4', 'EEG P4-LE': 'p4', 'EEG O2-LE': 'o2', 'EEG F8-LE': 'f8', 'EEG T4-LE': 't4', 'EEG T6-LE': 't6', 'EEG Cz-LE': 'cz', 'EEG Pz-LE': 'pz',
                   'EEG FP1-REF': 'fp1', 'EEG FP2-REF': 'fp2', 'EEG F3-REF': 'f3', 'EEG F4-REF': 'f4', 'EEG C3-REF': 'c3', 'EEG C4-REF': 'c4', 'EEG P3-REF': 'p3', 'EEG P4-REF': 'p4', 'EEG O1-REF': 'o1', 'EEG O2-REF': 'o2', 'EEG F7-REF': 'f7', 'EEG F8-REF': 'f8', 'EEG T3-REF': 't3', 'EEG T4-REF': 't4', 'EEG T5-REF': 't5', 'EEG T6-REF': 't6', 'EEG FZ-REF': 'fz', 'EEG CZ-REF': 'cz', 'EEG PZ-REF': 'pz', } # 'EEG A1-REF': 'a1', 'EEG A2-REF': 'a2', 'EEG T1-REF': 't1', 'EEG T2-REF': 't2',
        existing = set(raw.ch_names)
        safe_mapping = {k: v for k, v in mapping.items() if k in existing}
        if safe_mapping:
            raw.rename_channels(safe_mapping)
    elif dataset == 'tuev':
        #  ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EEG T1-REF', 'EEG T2-REF', 'PHOTIC-REF']
        mapping = {'EEG FP1-REF': 'fp1', 'EEG FP2-REF': 'fp2', 'EEG F3-REF': 'f3', 'EEG F4-REF': 'f4', 'EEG C3-REF': 'c3', 'EEG C4-REF': 'c4', 'EEG P3-REF': 'p3', 'EEG P4-REF': 'p4', 'EEG O1-REF': 'o1', 'EEG O2-REF': 'o2', 'EEG F7-REF': 'f7', 'EEG F8-REF': 'f8', 'EEG T3-REF': 't3', 'EEG T4-REF': 't4', 'EEG T5-REF': 't5', 'EEG T6-REF': 't6', 'EEG A1-REF': 'a1', 'EEG A2-REF': 'a2', 'EEG FZ-REF': 'fz', 'EEG CZ-REF': 'cz', 'EEG PZ-REF': 'pz'}
        existing = set(raw.ch_names)
        safe_mapping = {k: v for k, v in mapping.items() if k in existing}
        if safe_mapping:
            raw.rename_channels(safe_mapping)
    elif dataset == 'tusl':
        
        #  ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG 26-REF', 'EEG 27-REF', 'EEG 28-REF', 'EEG 29-REF', 'EEG 30-REF', 'EEG T1-REF', 'EEG T2-REF', 'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR']
        mapping = {'EEG FP1-REF': 'fp1', 'EEG FP2-REF': 'fp2', 'EEG F3-REF': 'f3', 'EEG F4-REF': 'f4', 'EEG C3-REF': 'c3', 'EEG C4-REF': 'c4', 'EEG P3-REF': 'p3', 'EEG P4-REF': 'p4', 'EEG O1-REF': 'o1', 'EEG O2-REF': 'o2', 'EEG F7-REF': 'f7', 'EEG F8-REF': 'f8', 'EEG T3-REF': 't3', 'EEG T4-REF': 't4', 'EEG T5-REF': 't5', 'EEG T6-REF': 't6', 'EEG A1-REF': 'a1', 'EEG A2-REF': 'a2', 'EEG FZ-REF': 'fz', 'EEG CZ-REF': 'cz', 'EEG PZ-REF': 'pz',
                   'EEG FP1-LE': 'fp1', 'EEG FP2-LE': 'fp2', 'EEG F3-LE': 'f3', 'EEG F4-LE': 'f4', 'EEG C3-LE': 'c3', 'EEG C4-LE': 'c4', 'EEG A1-LE': 'a1', 'EEG A2-LE': 'a2', 'EEG P3-LE': 'p3', 'EEG P4-LE': 'p4', 'EEG O1-LE': 'o1', 'EEG O2-LE': 'o2', 'EEG F7-LE': 'f7', 'EEG F8-LE': 'f8', 'EEG T3-LE': 't3', 'EEG T4-LE': 't4', 'EEG T5-LE': 't5', 'EEG T6-LE': 't6', 'EEG FZ-LE': 'fz', 'EEG CZ-LE': 'cz', 'EEG PZ-LE': 'pz'}
        # ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG A1-LE', 'EEG A2-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE', 'EEG OZ-LE', 'EEG PG1-LE', 'EEG PG2-LE', 'EEG EKG-LE', 'EEG SP2-LE', 'EEG SP1-LE', 'EEG 28-LE', 'EEG 29-LE', 'EEG 30-LE', 'EEG T1-LE', 'EEG T2-LE', 'PHOTIC PH']
        existing = set(raw.ch_names)
        safe_mapping = {k: v for k, v in mapping.items() if k in existing}
        if safe_mapping:
            raw.rename_channels(safe_mapping)
    elif dataset == 'hmc':
        # ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2', 'EMG chin', 'EOG E1-M2', 'EOG E2-M2', 'ECG']
        mapping = {'EEG F4-M1': 'f4', 'EEG C4-M1': 'c4', 'EEG O2-M1': 'o2', 'EEG C3-M2': 'c3'} ## note: not including EOG 
        existing = set(raw.ch_names)
        safe_mapping = {k: v for k, v in mapping.items() if k in existing}
        if safe_mapping:
            raw.rename_channels(safe_mapping)
            
    elif dataset == 'isruc':
        # ['LOC-A2', 'ROC-A1', 'F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'DC3', 'X7', 'X8', 'SaO2', 'DC8']
        # ['ROC', 'A1', 'LOC', 'A2', 'C4', 'O2', 'C3', 'O1', 'F4', 'F3', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'DC8', 'DC3', 'SaO2']
        # ['E1-M2', 'E2-M1', 'F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'DC4', 'X7', 'X8', 'SpO2', 'DC8', 'DC7']
        mapping = {'F3-A2': 'f3', 'C3-A2': 'c3', 'O1-A2': 'o1', 'F4-A1': 'f4', 'C4-A1': 'c4', 'O2-A1': 'o2', 'F3-M2': 'f3', 'C3-M2': 'c3', 'O1-M2': 'o1', 'F4-M1': 'f4', 'C4-M1': 'c4', 'O2-M1': 'o2', 'A1': 'a1', 'A2': 'a2', 'C3': 'c3', 'O1': 'o1', 'F4': 'f4', 'C4': 'c4', 'O2': 'o2', 'F3': 'f3'}
        existing = set(raw.ch_names)
        safe_mapping = {k: v for k, v in mapping.items() if k in existing}
        if safe_mapping:
            raw.rename_channels(safe_mapping)
        # if len(safe_mapping) == 4:
        #     bp() 
    elif dataset == 'tuab':
        mapping = {'EEG FP1-REF': 'fp1', 'EEG FP2-REF': 'fp2', 'EEG F3-REF': 'f3', 'EEG F4-REF': 'f4', 'EEG C3-REF': 'c3', 'EEG C4-REF': 'c4', 'EEG P3-REF': 'p3', 'EEG P4-REF': 'p4', 'EEG O1-REF': 'o1', 'EEG O2-REF': 'o2', 'EEG F7-REF': 'f7', 'EEG F8-REF': 'f8', 'EEG T3-REF': 't3', 'EEG T4-REF': 't4', 'EEG T5-REF': 't5', 'EEG T6-REF': 't6', 'EEG A1-REF': 'a1', 'EEG A2-REF': 'a2', 'EEG FZ-REF': 'fz', 'EEG CZ-REF': 'cz', 'EEG PZ-REF': 'pz',
                   'EEG FP1-LE': 'fp1', 'EEG FP2-LE': 'fp2', 'EEG F3-LE': 'f3', 'EEG F4-LE': 'f4', 'EEG C3-LE': 'c3', 'EEG C4-LE': 'c4', 'EEG A1-LE': 'a1', 'EEG A2-LE': 'a2', 'EEG P3-LE': 'p3', 'EEG P4-LE': 'p4', 'EEG O1-LE': 'o1', 'EEG O2-LE': 'o2', 'EEG F7-LE': 'f7', 'EEG F8-LE': 'f8', 'EEG T3-LE': 't3', 'EEG T4-LE': 't4', 'EEG T5-LE': 't5', 'EEG T6-LE': 't6', 'EEG FZ-LE': 'fz', 'EEG CZ-LE': 'cz', 'EEG PZ-LE': 'pz'}
        # ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG 26-REF', 'EEG 27-REF', 'EEG 28-REF', 'EEG 29-REF', 'EEG 30-REF', 'EEG T1-REF', 'EEG T2-REF', 'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR']
        existing = set(raw.ch_names)
        safe_mapping = {k: v for k, v in mapping.items() if k in existing}
        if safe_mapping:
            raw.rename_channels(safe_mapping)
    elif dataset == 'cognition':
        # ['NasalOral', 'Chest', 'Abdomen', 'M1', 'M2', 'F3', 'F4', 'C3', 'C4', 'LAT-U', 'LAT-L', 'RAT-U', 'RAT-L', 'E1', 'O1', 'E2', 'O2', 'SaO2', 'Nasal']
        mapping = {'F3-M2': 'f3', 'F4-M1': 'f4', 'C3-M2': 'c3', 'C4-M1': 'c4', 'O1-M2': 'o1', 'O2-M1': 'o2', 'F3': 'f3', 'F4': 'f4', 'C3': 'c3', 'C4': 'c4', 'O1': 'o1', 'O2': 'o2', 'C3-A2': 'c3', 'C4-A1': 'c4', 'O1-A2': 'o1','O2-A1': 'o2',
                   'F3-AVG': 'f3', 'F4-AVG': 'f4', 'C3-AVG': 'c3', 'C4-AVG': 'c4', 'O1-AVG': 'o1', 'O2-AVG': 'o2'}
        # ['C3-A2', 'C4-A1', 'O1-A2', 'O2-A1', 'Chin EMG', 'ROC-A1', 'LOC-A2', 'EKG', 'L/RAT', 'Airflow', 'Chest', 'Abdomen', 'Cannula', 'C-PRES', 'SpO2']
        #  ['arousal_caisr', 'caisr_prob_no-ar', 'caisr_prob_arous', 'limb_caisr', 'resp_caisr', 'stage_caisr', 'caisr_prob_n3', 'caisr_prob_n2', 'caisr_prob_n1', 'caisr_prob_r', 'caisr_prob_w']
        #  ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1', 'E2', 'CHIN', 'EKG', 'LAT', 'RAT', 'NPT', 'THERM', 'CHEST', 'ABDOMINAL', 'SaO2', 'C-FLOW', 'C PRESS']
        existing = set(raw.ch_names)
        safe_mapping = {k: v for k, v in mapping.items() if k in existing}
        if safe_mapping:
            raw.rename_channels(safe_mapping)
        if len(safe_mapping) == 0 and any([item.startswith('caisr') or item.endswith('expert') for item in raw.ch_names]):
            print('skiiping file', edf_path,'with channels: ', raw.ch_names)
            return 
        if len(safe_mapping) < 4:
            bp() 
            
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    try:
        data, sfreq = preprocess_data(raw)
    except:
        print('failed at ', edf_path)
        return 
    if data is None:
        return
    data = data.T.astype(np.float32)                       # → (n_t, n_ch)
    if target_units == "uV":
        data *= 1e6

    
    str_dt = h5py.string_dtype('utf-8')
    chunk_t = max(1, int(sfreq * chunk_sec))                       # ~5s stripes, >=1

    with h5py.File(h5_path, "a") as f:
        gname = str(group) if group is not None else "recording"
        g = f.require_group(gname)

        # If re-running, replace datasets cleanly
        for key in ("data", "ch_names", "bad_channels", "index"):
            if key in g: del g[key]

        g.create_dataset(
            "data", data=data,
            chunks=(min(chunk_t, data.shape[0]), data.shape[1]),
            compression=compression, compression_opts=compression_opts, shuffle=shuffle
        )
        g.create_dataset("ch_names", data=np.array(raw.ch_names, dtype=str_dt))
        g.create_dataset("bad_channels", data=np.array(raw.info.get("bads", []), dtype=str_dt))
        g.attrs["fs"] = sfreq
        g.attrs["units"] = target_units
        g.attrs["start_time_iso"] = (raw.info.get("meas_date") or "").isoformat() \
                                    if hasattr(raw.info.get("meas_date"), "isoformat") else ""
        g.attrs["source_path"] = os.path.abspath(edf_path)
    build_index(h5_path)

class ExternalLabelHunter:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == 'tuab':
            self.eeg_data_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUAB/h5/TUAB'
            ## open the files and get the filenames
            with open(os.path.join(REPO_DIR, "data/tuab_train.txt"), 'r') as f:
                self.train_filenames = f.read().splitlines()
            with open(os.path.join(REPO_DIR, "data/tuab_test.txt"), 'r') as f:
                self.val_filenames = f.read().splitlines()
            NORMAL_FILE_LIST = os.path.join(REPO_DIR, "data/tuab_normal.txt")
            ABNORMAL_FILE_LIST = os.path.join(REPO_DIR, "data/tuab_abnormal.txt")
            normal_files = open(NORMAL_FILE_LIST, 'r').read().splitlines()
            abnormal_files = open(ABNORMAL_FILE_LIST, 'r').read().splitlines()
            normal_abnormal_dict = {file: 0 for file in normal_files}
            normal_abnormal_dict.update({file: 1 for file in abnormal_files})
            self.label_dict = normal_abnormal_dict
            self.downstream_tasks = ['tuab']
        if dataset == 'tuep':
            self.eeg_data_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEP/h5/'
        
            EPILEPSY_FILE_LIST = os.listdir('/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEP/data/v3.0.0/00_epilepsy/')
            NON_EPILEPSY_FILE_LIST = os.listdir('/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEP/data/v3.0.0/01_no_epilepsy/')
            
            all_filenames = [self.eeg_data_dir + file for file in os.listdir(self.eeg_data_dir) if file.endswith('.h5')]
            if os.path.exists(os.path.join(REPO_DIR, "data/tuep_train.txt")):
                with open(os.path.join(REPO_DIR, "data/tuep_train.txt"), 'r') as f:
                    self.train_filenames = f.read().splitlines()
                with open(os.path.join(REPO_DIR, "data/tuep_test.txt"), 'r') as f:
                    self.val_filenames = f.read().splitlines()
            else:
                train_val_split = train_test_split(all_filenames, test_size=0.25, random_state=42)
                self.train_filenames = train_val_split[0]
                self.val_filenames = train_val_split[1]
                with open(os.path.join(REPO_DIR, "data/tuep_train.txt"), 'w') as f:
                    for file in self.train_filenames:
                        f.write(file + '\n')
                with open(os.path.join(REPO_DIR, "data/tuep_test.txt"), 'w') as f:
                    for file in self.val_filenames:
                        f.write(file + '\n')
            self.label_dict = {file: (0 if file.split('/')[-1].split('_')[0] in NON_EPILEPSY_FILE_LIST else 1 if file.split('/')[-1].split('_')[0] in EPILEPSY_FILE_LIST else None) for file in all_filenames}
            self.downstream_tasks = ['tuep']
        if dataset == 'mumtaz':
            self.eeg_data_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/MUMTAZ/h5/'
            all_filenames = [self.eeg_data_dir + file for file in os.listdir(self.eeg_data_dir) if file.endswith('.h5')]
            if os.path.exists(os.path.join(REPO_DIR, "data/mumtaz_train.txt")):
                with open(os.path.join(REPO_DIR, "data/mumtaz_train.txt"), 'r') as f:
                    self.train_filenames = f.read().splitlines()
                with open(os.path.join(REPO_DIR, "data/mumtaz_test.txt"), 'r') as f:
                    self.val_filenames = f.read().splitlines()
            else:
                train_val_split = train_test_split(all_filenames, test_size=0.25, random_state=42)
                self.train_filenames = train_val_split[0]
                self.val_filenames = train_val_split[1]
                with open(os.path.join(REPO_DIR, "data/mumtaz_train.txt"), 'w') as f:
                    for file in self.train_filenames:
                        f.write(file + '\n')
                with open(os.path.join(REPO_DIR, "data/mumtaz_test.txt"), 'w') as f:
                    for file in self.val_filenames:
                        f.write(file + '\n')
            self.label_dict = {file: (1 if file.split('/')[-1].split(' ')[0] == 'MDD' else 0) for file in all_filenames}
            self.downstream_tasks = ['mumtaz']
        if dataset == 'hmc':
            self.downstream_tasks = ['stage_hmc']
            self.eeg_data_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/HMC/h5/'
            all_filenames = [self.eeg_data_dir + file for file in os.listdir(self.eeg_data_dir) if file.endswith('.h5')]
            if os.path.exists(os.path.join(REPO_DIR, "data/hmc_train.txt")):
                with open(os.path.join(REPO_DIR, "data/hmc_train.txt"), 'r') as f:
                    self.train_filenames = f.read().splitlines()
                with open(os.path.join(REPO_DIR, "data/hmc_test.txt"), 'r') as f:
                    self.val_filenames = f.read().splitlines()
            else:
                train_val_split = train_test_split(all_filenames, test_size=0.25, random_state=42)
                self.train_filenames = train_val_split[0]
                self.val_filenames = train_val_split[1]
                with open(os.path.join(REPO_DIR, "data/hmc_train.txt"), 'w') as f:
                    for file in self.train_filenames:
                        f.write(file + '\n')
                with open(os.path.join(REPO_DIR, "data/hmc_test.txt"), 'w') as f:
                    for file in self.val_filenames:
                        f.write(file + '\n')
            self.label_dict = None 
        if dataset == 'isruc': 
            self.downstream_tasks = ['stage_isruc']
            self.eeg_data_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/ISRUC/h5/'
            all_filenames = [self.eeg_data_dir + file for file in os.listdir(self.eeg_data_dir) if file.endswith('.h5')]
            if os.path.exists(os.path.join(REPO_DIR, "data/isruc_train.txt")):
                with open(os.path.join(REPO_DIR, "data/isruc_train.txt"), 'r') as f:
                    self.train_filenames = f.read().splitlines()
                with open(os.path.join(REPO_DIR, "data/isruc_test.txt"), 'r') as f:
                    self.val_filenames = f.read().splitlines()
            else:
                train_val_split = train_test_split(all_filenames, test_size=0.25, random_state=42)
                self.train_filenames = train_val_split[0]
                self.val_filenames = train_val_split[1]
                with open(os.path.join(REPO_DIR, "data/isruc_train.txt"), 'w') as f:
                    for file in self.train_filenames:
                        f.write(file + '\n')
                with open(os.path.join(REPO_DIR, "data/isruc_test.txt"), 'w') as f:
                    for file in self.val_filenames:
                        f.write(file + '\n')
            self.label_dict = None
        if dataset == 'tuev':
            self.downstream_tasks = ['tuev']
            self.data_dir = '/orcd/compute/dinaktbi/001/2026/EEG_FM/EXTERNAL_DATASETS/TUEV/data/v2.0.1/edf/'
            
            
        ## tuev, tusl, isruc, hmc, are all session based labels 
                        
    def get_label(self, filename):
        return self.label_dict[filename]

def run_downstream_external(args, labels, EEG_DATA_DIR):
    """ This function runs the downstream tuab training for the given seed and gpu id.
    Args:
        args: argparse.Namespace object containing the arguments
    """

    args.downstream_tasks = labels.downstream_tasks
    args.num_downstream_tasks = len(args.downstream_tasks)
    print('Downstream tasks: ', args.downstream_tasks)
    print('Number of downstream tasks: ', args.num_downstream_tasks)
    lr_dict = {'per_embedding_mlp': 1e-4, 'mlp_then_pooling': 1e-4, 'psg': 1e-4}
    batch_size_dict = {'labram': 64, 'reve': 96, 'cbramod': 96, 'ours': 128, 'ours_multitaper': 128, 'multitaper': 128}
    if args.rep_learning_model == 'mlp_then_pooling':
        batch_size_dict[args.baseline_model] = batch_size_dict[args.baseline_model] // 8
    run_args = {
        'batch_size': batch_size_dict[args.baseline_model] if not args.debug else 4, ## 24 if L40S, 64 if H200 (~2hrs for 64), note: needs ~64cpus
        'num_workers': 0 if args.debug else 12,  # 0 in debug so __getitem__ runs in main process and breakpoints work 
        'lr': lr_dict[args.rep_learning_model], 
        'num_classes': 1,
        'epochs': 5,
        'lr_warmup_prop': 0.1,
        'data_train_path': os.path.join(REPO_DIR, "data/tuab_train.txt"),
        'data_val_path': os.path.join(REPO_DIR, "data/tuab_test.txt"),
        'eeg_data_dir': EEG_DATA_DIR,
        'report_dir': '',
        'exp_path': args.exp_path,
        'custom_name': args.custom_name,
        'seed': args.seed,
        'rep_learning_model': args.rep_learning_model,
        'save_path': args.save_path,
        'checkpoint_name': args.checkpoint_name,
        'load_ema': args.load_ema,
        'downstream_tasks': args.downstream_tasks,
        'debug': args.debug,
        'start_from_tokens': args.start_from_tokens,
    }
    if args.start_from_tokens:
        raise NotImplementedError('Start from tokens not implemented for external datasets')
        # train_loader, test_loader = load_data_tokenized(args, run_args, labels)
    else:
        train_loader, test_loader = load_data(args, run_args, labels)
    model = load_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    downstream_model = DownstreamProbeModel(encoder=model, baseline_model=args.baseline_model, model_type=args.rep_learning_model, downstream_tasks=args.downstream_tasks, num_channels=19, data_length=8)
    downstream_model = downstream_model.to(device)
    train_loader.dataset[0]
    test1 = next(iter(train_loader))
    output1 = downstream_model(test1['eeg'].cuda(), test1['eeg_mask'].cuda())

    #optimizer = optim.Adam(downstream_model.downstream_model.parameters(), lr=run_args['lr'])
    optimizers = {}
    # for task in args.downstream_tasks:
    #     optimizers[task] = optim.Adam(downstream_model.downstream_models[task].parameters(), lr=run_args['lr'])
    for i, task in enumerate(args.downstream_tasks):
        task_params = [
            downstream_model.weight1_list[i],
            downstream_model.bias1_list[i],
            downstream_model.weight2_list[i],
            downstream_model.bias2_list[i]
        ]
        optimizers[task] = optim.Adam(task_params, lr=run_args['lr'])
    
    for param in downstream_model.encoder.parameters():
        param.requires_grad = False
    return train_model(downstream_model, train_loader, test_loader, optimizers, run_args, labels)

def load_data(args, run_args, labels):
    if args.dataset in ['hmc', 'isruc']: 
        train_dataset = PSGDataset(args.dataset, 
            file_list_path=labels.train_filenames,
            eeg_data_dir=run_args['eeg_data_dir'],
            report_dir=run_args['report_dir'],
            max_length_seconds=args.max_length_seconds,
            fs=args.fs,
            window_length=args.window_length,
            stride_length=args.stride_length,
            max_text_length=args.max_text_length,
            disable_clip=True, # args.disable_clip,
            mode='train' if not args.debug else 'val', 
            raw_eeg=False if args.baseline_model == 'ours' or args.baseline_model == 'ours_multitaper' or args.baseline_model == 'multitaper' else True, 
            debug=args.debug
        )
        train_dataset[0]
        val_dataset = PSGDataset(args.dataset, 
            file_list_path=labels.val_filenames,
            eeg_data_dir=run_args['eeg_data_dir'],
            report_dir=run_args['report_dir'],
            max_length_seconds=args.max_length_seconds,
            fs=args.fs,
            window_length=args.window_length,
            stride_length=args.stride_length,
            max_text_length=args.max_text_length,
            disable_clip=True, # args.disable_clip,
            mode='val', 
            raw_eeg=False if args.baseline_model == 'ours' or args.baseline_model == 'ours_multitaper' or args.baseline_model == 'multitaper' else True, 
            debug=args.debug
        )

    else:
        train_dataset = EEGTextDataset(
            file_list_path=labels.train_filenames,
            eeg_data_dir=run_args['eeg_data_dir'],
            report_dir=run_args['report_dir'],
            max_length_seconds=args.max_length_seconds,
            fs=args.fs,
            window_length=args.window_length,
            stride_length=args.stride_length,
            max_text_length=args.max_text_length,
            disable_clip=True, # args.disable_clip,
            mode='train' if not args.debug else 'val', 
            raw_eeg=False if args.baseline_model == 'ours' or args.baseline_model == 'ours_multitaper' or args.baseline_model == 'multitaper' else True, 
            debug=args.debug
        )
            # Validation dataset and loader
        val_dataset = EEGTextDataset(
            file_list_path=labels.val_filenames,
            eeg_data_dir=run_args['eeg_data_dir'],
            report_dir=run_args['report_dir'],
            max_length_seconds=args.max_length_seconds,
            fs=args.fs,
            window_length=args.window_length,
            stride_length=args.stride_length,
            max_text_length=args.max_text_length,
            disable_clip=True, #args.disable_clip,
            mode='val',
            raw_eeg=False if args.baseline_model == 'ours' or args.baseline_model == 'ours_multitaper' or args.baseline_model == 'multitaper' else True
        )
    if args.debug:
        train_dataset = Subset(train_dataset, range(min(1000, len(train_dataset))))
    else:
        train_dataset = SafeDataset(train_dataset)
    nw = run_args['num_workers']
    train_loader_kw = dict(batch_size=run_args['batch_size'], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, collate_fn=safe_collate_fn if not args.debug else None, worker_init_fn=worker_init_fn)
    if nw > 0:
        train_loader_kw.update(prefetch_factor=2)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_kw)

    if args.debug:
        val_dataset = Subset(val_dataset, range(min(1000, len(val_dataset))))
    else:
        val_dataset = SafeDataset(val_dataset)
    val_batch_size = 2 * run_args['batch_size'] if args.baseline_model != 'reve' else run_args['batch_size']
    val_loader_kw = dict(batch_size=val_batch_size, shuffle=False, num_workers=nw, pin_memory=True, drop_last=False, collate_fn=safe_collate_fn if not args.debug else None, worker_init_fn=worker_init_fn)
    if nw > 0:
        val_loader_kw.update(prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_kw)
    return train_loader, val_loader


def setup_downstream_external(gpu_id, seed, args, return_dict, dataset):
    """ This function runs the downstream tuab training for the given seed and gpu id.
    Args:
        gpu_id: gpu id to use
        seed: seed to use
        args: argparse.Namespace object containing the arguments
        return_dict: dictionary to store the results
    """
    

    exp_path = args.exp_path 
    custom_name = args.custom_name
    is_debug = args.debug
    rep_learning_model = args.rep_learning_model
    save_path = args.save_path
    checkpoint_name = args.checkpoint_name
    load_ema = args.load_ema
    start_from_tokens = args.start_from_tokens
    baseline_model = args.baseline_model
    vqgan_ckpt_folder = args.vqgan_ckpt_folder
    ## load the args 
    if args.baseline_model == 'ours':
        with open(os.path.join(args.exp_path, 'args.json'), 'r') as f:
            base_args = json.load(f)
        # Create a complete args namespace with all fields
        args = argparse.Namespace(**base_args)
    else:
        args = argparse.Namespace()
        args.max_length_seconds = 1280
        args.fs = 200
        args.window_length = 4
        args.stride_length = 1
        args.max_text_length = 256
        args.disable_clip = True
        args.vqgan_ckpt_folder = vqgan_ckpt_folder
        
    args.baseline_model = baseline_model
    args.seed = seed
    args.exp_path = exp_path
    args.custom_name = custom_name
    args.debug = is_debug
    args.rep_learning_model = rep_learning_model
    args.save_path = save_path
    args.checkpoint_name = checkpoint_name
    args.load_ema = load_ema
    args.start_from_tokens = start_from_tokens
    
    args.dataset = dataset
    args.eeg_token_dir = '/home/alimirz/orcd/scratch/eeg_tokenization/eeg_tokens_v3/train/'
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    

    labels = ExternalLabelHunter(dataset)
    EEG_DATA_DIR = labels.eeg_data_dir


    
    result = run_downstream_external(args, labels, EEG_DATA_DIR)
    return_dict[seed] = {'seed': seed, 'gpu_id': gpu_id, 'success': True, 'result': result}
    return
    # # Set the GPU for this process
    # torch.cuda.set_device(gpu_id)
    # ## commenting out because only 1 gpu is used 
    # # cpu_affinity = pin_process_to_gpu(gpu_id)
    # # print(f"GPU {gpu_id} pinned to CPUs {cpu_affinity}")
    # print(f"Starting seed {seed} on GPU {gpu_id}")
    # try:
    #     # Run training
    #     result = run_downstream_tuab(args)
    #     print(f"Completed seed {seed} on GPU {gpu_id}")
    #     return_dict[seed] = {'seed': seed, 'gpu_id': gpu_id, 'success': True, 'result': result}
    # except Exception as e:
    #     print(f"Failed seed {seed} on GPU {gpu_id}: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     return_dict[seed] = {'seed': seed, 'gpu_id': gpu_id, 'success': False, 'error': str(e), 'traceback': traceback.format_exc()}



if __name__ == "__main__":
    # create_data_text_file()
    # create_normal_abnormal_text_file() 
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_model', type=str, choices=['labram', 'reve', 'cbramod', 'ours', 'ours_multitaper', 'multitaper'], default='ours')
    parser.add_argument('--rep_learning_model', type=str, choices=['per_embedding_mlp', 'mlp_then_pooling', 'psg'], default='psg')
    parser.add_argument('--dataset', type=str, nargs='+', choices=['mumtaz', 'tuab', 'tuev', 'isruc', 'hmc', 'tuep', 'tusl', 'cognition'], default=['isruc'])
    parser.add_argument('--exp_path', type=str, default='/orcd/compute/dinaktbi/001/2026/EEG_FM/exp_runs/rep-learning/recon_profile_report_large')
    parser.add_argument('--vqgan_ckpt_folder', type=str, default='/orcd/data/dinaktbi/001/2026/EEG_FM/exp_runs/vqgan/logs/harvard_vqgan_engaging_128x640_-40_40_BW_1/checkpoints/epoch=000005-step=000200000.ckpt')
    parser.add_argument('--custom_name', type=str, default='')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,], #2, 3, 4
                       help='List of seeds to run')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Debug mode')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint-10.pth',
                       help='Checkpoint name')
    parser.add_argument('--load_ema', action='store_true', default=False,
                       help='Load EMA model')
    parser.add_argument('--start_from_tokens', action='store_true', default=False,
                       help='Start from tokens')
    parser.add_argument('--prepare_mumtaz', action='store_true', default=False,
                       help='Prepare Mumtaz dataset')
    parser.add_argument('--prepare_tuep', action='store_true', default=False,
                       help='Prepare TUEP dataset')
    parser.add_argument('--prepare_tuev', action='store_true', default=False,
                       help='Prepare TUEV dataset')
    parser.add_argument('--prepare_tusl', action='store_true', default=False,
                       help='Prepare TUSL dataset')
    parser.add_argument('--prepare_hmc', action='store_true', default=False,
                       help='Prepare HMC dataset')
    parser.add_argument('--prepare_isruc', action='store_true', default=False,
                       help='Prepare ISRUC dataset')
    parser.add_argument('--prepare_tuab', action='store_true', default=False,
                       help='Prepare TUAB dataset')
    parser.add_argument('--prepare_cognition', action='store_true', default=False,
                       help='Prepare Cognition dataset')
    parser.add_argument('--verify_channels', action='store_true', default=False, help='Verify channels')
    
    args = parser.parse_args()

    if args.prepare_mumtaz:
        prepare_mumtaz()
        exit()
    
    if args.prepare_tuep:
        prepare_tuep()
        exit()
    if args.prepare_tuev:
        prepare_tuev()
        exit()
    if args.prepare_tusl:
        prepare_tusl()
        exit()
    if args.prepare_hmc:
        prepare_hmc()
        exit()
    if args.prepare_isruc:
        prepare_isruc()
        exit()
    if args.prepare_tuab:
        prepare_tuab()
        exit()
    if args.prepare_cognition:
        prepare_cognition()
        exit()
    if args.verify_channels:
        verify_channels(args.dataset)
        exit()
    if args.exp_path.endswith('/'):
        args.exp_path = args.exp_path[:-1]
    last_dir = args.exp_path.split('/')[-1]
    save_path = args.exp_path.replace('rep-learning', f'baselines_downstream_external/{args.baseline_model}')
    if not args.baseline_model.startswith('ours'):
        save_path = save_path.replace(last_dir, args.baseline_model)
    os.makedirs(save_path, exist_ok=True)
    args.save_path = save_path

    start_time = time.time()
    # all_results = main_parallel(setup_downstream_tuab, args, seeds=args.seeds, num_runs=args.num_runs, debug=args.debug)
    return_dict = {}
    seed = args.seeds[0]
    all_results = {}
    for dataset in args.dataset:
        setup_downstream_external(gpu_id=0, seed=seed, args=args, return_dict=return_dict, dataset=dataset)
        all_results[dataset] = [return_dict[seed]]
    
    bp() 
    total_time = time.time() - start_time
    print(f'Total time: {total_time / 3600:.2f} hours')


    
    final_results, exp_name = parallel_calculate_final_results(all_results)
    print(final_results)
    print(exp_name)
    ## create a dataframe from the final results, every row is a 
    
    tasks = set()
    for key in final_results.keys():
        parts = key.split('/')
        if len(parts) == 4:
            tasks.add(parts[2])

    # Load task split counts if available (train/test N positive and N negative per task)
    task_split_counts = {}
    counts_path = os.path.join(args.save_path, 'task_split_counts.json')
    if os.path.isfile(counts_path):
        with open(counts_path) as f:
            task_split_counts = json.load(f)

    # Create a dictionary to build the DataFrame
    # Create a dictionary to build the DataFrame
    data = {}
    for task in tasks:
        row = {
            'best/test': final_results.get(f"best/test/{task}/auroc"),
            'best/train': final_results.get(f"best/train/{task}/auroc"),
            'last/test': final_results.get(f"last/test/{task}/auroc"),
            'last/train': final_results.get(f"last/train/{task}/auroc")
        }
        if task in task_split_counts:
            row['train_N_positive'] = task_split_counts[task].get('train_N_positive')
            row['train_N_negative'] = task_split_counts[task].get('train_N_negative')
            row['test_N_positive'] = task_split_counts[task].get('test_N_positive')
            row['test_N_negative'] = task_split_counts[task].get('test_N_negative')
        else:
            row['train_N_positive'] = row['train_N_negative'] = row['test_N_positive'] = row['test_N_negative'] = None
        data[task] = row
    data['overall_downstream'] = {
        'best/test': np.nanmean([final_results.get(f"best/test/{task}/auroc") for task in tasks]),
        'best/train': np.nanmean([final_results.get(f"best/train/{task}/auroc") for task in tasks]),
        'last/test': np.nanmean([final_results.get(f"last/test/{task}/auroc") for task in tasks]),
        'last/train': np.nanmean([final_results.get(f"last/train/{task}/auroc") for task in tasks]),
        'train_N_positive': None, 'train_N_negative': None, 'test_N_positive': None, 'test_N_negative': None
    }
    for category in ['med_', 'dis_', 'smed_', 'cond_', 'diag_', 'feat_']:
        data['overall_downstream_'+category] = {
            'best/test': np.nanmean([final_results.get(f"best/test/{task}/auroc") for task in tasks if task.startswith(category)]),
            'best/train': np.nanmean([final_results.get(f"best/train/{task}/auroc") for task in tasks if task.startswith(category)]),
            'last/test': np.nanmean([final_results.get(f"last/test/{task}/auroc") for task in tasks if task.startswith(category)]),
            'last/train': np.nanmean([final_results.get(f"last/train/{task}/auroc") for task in tasks if task.startswith(category)]),
            'train_N_positive': None, 'train_N_negative': None, 'test_N_positive': None, 'test_N_negative': None
        }
    

    # Create DataFrame with tasks as rows
    final_results_df = pd.DataFrame.from_dict(data, orient='index')
    final_results_df.index.name = 'task'

    # Sort by last/test from highest to lowest
    final_results_df = final_results_df.sort_values('last/test', ascending=False)
    
    ## put the rows with overall_downstream at the top
    overall_downstream_rows = [idx for idx in final_results_df.index if idx.startswith('overall_downstream')]
    overall_downstream = final_results_df.loc[overall_downstream_rows]
    ## sort them alphabetically
    overall_downstream = overall_downstream.sort_index()
    final_results_df = final_results_df.drop(index=overall_downstream_rows)
    final_results_df = pd.concat([overall_downstream, final_results_df])

    summarize_exp_parallel(final_results, os.path.join(args.save_path, exp_name), args)
    final_results_df.to_csv(os.path.join(args.save_path, exp_name, 'final_results.csv'))
    print('saved final results to csv: ')
    print(os.path.join(args.save_path, exp_name, 'final_results.csv'))
