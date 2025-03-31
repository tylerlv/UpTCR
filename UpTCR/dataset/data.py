import os
import pickle
import json

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import random
from sklearn.utils import shuffle

class TCRABpMHCDataset_combine(Dataset):

    def __init__(self, data_path, TCRA_emb_path, TCRB_emb_path, pep_emb_path):
        super().__init__()
        self.data_path = data_path
        self.TCRA_emb_path = TCRA_emb_path
        self.TCRB_emb_path = TCRB_emb_path
        self.pep_emb_path = pep_emb_path
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        epitope_seq = row['Epitope']
        pep_emb_file = os.path.join(self.pep_emb_path, f"{epitope_seq}.pt")
        pep_emb = torch.load(pep_emb_file)

        tcrb_cdr3_seq = row['TRB.CDR3']
        tcrb_fv_seq = row['FV_beta']
        tcrb_emb_file = os.path.join(self.TCRB_emb_path, f"{tcrb_fv_seq}.pt")

        try:
            tcrb_emb = torch.load(tcrb_emb_file).squeeze(dim=0)
        except EOFError:
            print(f"Error loading file: {tcrb_emb_file}. The file may be corrupted or incomplete.")
        tcra_cdr3_seq = row['TRA.CDR3']
        tcra_fv_seq = row['FV_alpha']
        tcra_emb_file = os.path.join(self.TCRA_emb_path, f"{tcra_fv_seq}.pt")

        if not os.path.exists(tcra_emb_file):
            raise FileNotFoundError(f"File not found: {tcra_emb_file}")

        try:
            tcra_emb = torch.load(tcra_emb_file)
        except EOFError:
            print(f"Error loading file: {tcra_emb_file}. The file may be corrupted or incomplete.")

        if tcra_emb is None:
            raise ValueError(f"Failed to load file: {tcra_emb_file}")
        
        tcra_emb = tcra_emb.squeeze(dim=0)

        tcra_cdr3_start_index = tcra_fv_seq.find(tcra_cdr3_seq)
        if tcra_cdr3_start_index == -1:
            raise ValueError(f"CDR3 sequence '{tcra_cdr3_seq}' not found in FV sequence '{tcra_fv_seq}'")
        
        tcrb_cdr3_start_index = tcrb_fv_seq.find(tcrb_cdr3_seq)
        if tcrb_cdr3_start_index == -1:
            raise ValueError(f"CDR3 sequence '{tcrb_cdr3_seq}' not found in FV sequence '{tcrb_fv_seq}'")

        mhcseq = row['MHCseq']
        try:
            label = row['Label']
        except KeyError:
            label = row['label']

        if 'TCRname' in row.index:
            tcrname = row['TCRname']
            tcrmut = row['TCR_mut']
            return {
                'epitope_seq': epitope_seq,
                'pep_emb': pep_emb,
                'tcra_cdr3_seq': tcra_cdr3_seq,
                'tcra_fv_seq': tcra_fv_seq,
                'tcra_emb': tcra_emb,
                'tcra_cdr3_start_index': tcra_cdr3_start_index,
                'tcrb_cdr3_seq': tcrb_cdr3_seq,
                'tcrb_fv_seq': tcrb_fv_seq,
                'tcrb_emb': tcrb_emb,
                'tcrb_cdr3_start_index': tcrb_cdr3_start_index,
                'mhc': mhcseq,
                'label': label,
                'pdb_code': '1',
                'tcrname': tcrname,
                'tcrmut': tcrmut
            }
        else:
            if 'PDB Code' not in row.index:
                return {
                    'epitope_seq': epitope_seq,
                    'pep_emb': pep_emb,
                    'tcra_cdr3_seq': tcra_cdr3_seq,
                    'tcra_fv_seq': tcra_fv_seq,
                    'tcra_emb': tcra_emb,
                    'tcra_cdr3_start_index': tcra_cdr3_start_index,
                    'tcrb_cdr3_seq': tcrb_cdr3_seq,
                    'tcrb_fv_seq': tcrb_fv_seq,
                    'tcrb_emb': tcrb_emb,
                    'tcrb_cdr3_start_index': tcrb_cdr3_start_index,
                    'mhc': mhcseq,
                    'label': label,
                    'pdb_code': '1',
                    'tcrname': '1',
                    'tcrmut': '1'
                }
            else:
                return {
                    'epitope_seq': epitope_seq,
                    'pep_emb': pep_emb,
                    'tcra_cdr3_seq': tcra_cdr3_seq,
                    'tcra_fv_seq': tcra_fv_seq,
                    'tcra_emb': tcra_emb,
                    'tcra_cdr3_start_index': tcra_cdr3_start_index,
                    'tcrb_cdr3_seq': tcrb_cdr3_seq,
                    'tcrb_fv_seq': tcrb_fv_seq,
                    'tcrb_emb': tcrb_emb,
                    'tcrb_cdr3_start_index': tcrb_cdr3_start_index,
                    'mhc': mhcseq,
                    'label': label,
                    'pdb_code': row['PDB Code'],
                    'tcrname': '1',
                    'tcrmut': '1'
                }
    
    def get_raw_data(self):
        return self.data
    
    def __len__(self):
        return len(self.data)

class Structure_Dataset(Dataset):
    def __init__(self, data_path, TCRA_emb_path, TCRB_emb_path, pep_emb_path, structure_path):
        super().__init__()

        self.data_path = data_path
        self.TCRA_emb_path = TCRA_emb_path
        self.TCRB_emb_path = TCRB_emb_path
        self.pep_emb_path = pep_emb_path
        self.structure_path = structure_path
        self.data = self.load_data()
    
    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        epitope_seq = row['Epitope']
        pep_emb_file = os.path.join(self.pep_emb_path, f"{epitope_seq}.pt")
        pep_emb = torch.load(pep_emb_file)

        tcrb_cdr3_seq = row['TRB.CDR3']
        tcrb_fv_seq = row['FV_beta']
        tcrb_emb_file = os.path.join(self.TCRB_emb_path, f"{tcrb_fv_seq}.pt")

        try:
            tcrb_emb = torch.load(tcrb_emb_file).squeeze(dim=0)
        except EOFError:
            print(f"Error loading file: {tcrb_emb_file}. The file may be corrupted or incomplete.")
        tcra_cdr3_seq = row['TRA.CDR3']
        tcra_fv_seq = row['FV_alpha']
        tcra_emb_file = os.path.join(self.TCRA_emb_path, f"{tcra_fv_seq}.pt")

        if not os.path.exists(tcra_emb_file):
            raise FileNotFoundError(f"File not found: {tcra_emb_file}")

        try:
            tcra_emb = torch.load(tcra_emb_file)
        except EOFError:
            print(f"Error loading file: {tcra_emb_file}. The file may be corrupted or incomplete.")

        if tcra_emb is None:
            raise ValueError(f"Failed to load file: {tcra_emb_file}")
        
        tcra_emb = tcra_emb.squeeze(dim=0)

        tcra_cdr3_start_index = tcra_fv_seq.find(tcra_cdr3_seq)
        if tcra_cdr3_start_index == -1:
            raise ValueError(f"CDR3 sequence '{tcra_cdr3_seq}' not found in FV sequence '{tcra_fv_seq}'")
        
        tcrb_cdr3_start_index = tcrb_fv_seq.find(tcrb_cdr3_seq)
        if tcrb_cdr3_start_index == -1:
            raise ValueError(f"CDR3 sequence '{tcrb_cdr3_seq}' not found in FV sequence '{tcrb_fv_seq}'")

        mhcseq = row['MHCseq']

        pdb_code = row['PDB Code']
        pdb_path = os.path.join(self.structure_path, f'{pdb_code}.npy')
        npy = np.load(pdb_path, allow_pickle=True)
        labels = npy.item()[pdb_code]

        return {
                'epitope_seq': epitope_seq,
                'pep_emb': pep_emb,
                'tcra_cdr3_seq': tcra_cdr3_seq,
                'tcra_fv_seq': tcra_fv_seq,
                'tcra_emb': tcra_emb,
                'tcra_cdr3_start_index': tcra_cdr3_start_index,
                'tcrb_cdr3_seq': tcrb_cdr3_seq,
                'tcrb_fv_seq': tcrb_fv_seq,
                'tcrb_emb': tcrb_emb,
                'tcrb_cdr3_start_index': tcrb_cdr3_start_index,
                'mhc': mhcseq,
                'label': labels,
                'pdb_code': pdb_code
        }
    def __len__(self):
        return len(self.data)

class pMHC_finetune_Dataset(Dataset):
    def __init__(self, data_path, pep_emb_path):
        super().__init__()
        self.data_path = data_path
        self.pep_emb_path = pep_emb_path
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        epitope = row['Epitope']
        mhcseq = row['MHCseq']
        affinity = row['Score']

        emb_path = os.path.join(self.pep_emb_path, f"{epitope}.pt")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embedding file {emb_path} not found for epitope {epitope}")
        embedding = torch.load(emb_path)

        return {
            'mhc': mhcseq,
            'epitope': epitope,
            'pep_emb': embedding,
            'Score': affinity,
        }

    def __len__(self):
        return len(self.data)