import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from collections import defaultdict

class TCRApSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        # Group indices by unique TRA.CDR3
        cdr3_to_indices = defaultdict(list)
        for idx in range(len(self.data_source)):
            original_idx = self.data_source.indices[idx]
            row = self.data_source.dataset.data.iloc[original_idx]
            cdr3_to_indices[row['TRA.CDR3']].append(idx)

        # Shuffle the groups
        unique_cdr3s = list(cdr3_to_indices.keys())
        random.shuffle(unique_cdr3s)

        # Create batches
        batches = []
        current_batch = []
        used_indices = set()
        for cdr3 in unique_cdr3s:
            indices = cdr3_to_indices[cdr3]
            random.shuffle(indices)
            for idx in indices:
                if idx not in used_indices:
                    current_batch.append(idx)
                    used_indices.add(idx)
                    if len(current_batch) == self.batch_size:
                        batches.append(current_batch)
                        current_batch = []
                        break  # Move to the next cdr3 group

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # Shuffle the batches
        random.shuffle(batches)

        # Flatten the list of batches
        flat_batches = [idx for batch in batches for idx in batch]
        return iter(flat_batches)

    def __len__(self):
        return len(self.data_source)


class TCRBpSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        # Group indices by unique TRA.CDR3
        cdr3_to_indices = defaultdict(list)
        for idx in range(len(self.data_source)):
            original_idx = self.data_source.indices[idx]
            row = self.data_source.dataset.data.iloc[original_idx]
            cdr3_to_indices[row['TRB.CDR3']].append(idx)

        # Shuffle the groups
        unique_cdr3s = list(cdr3_to_indices.keys())
        random.shuffle(unique_cdr3s)

        # Create batches
        batches = []
        current_batch = []
        used_indices = set()
        for cdr3 in unique_cdr3s:
            indices = cdr3_to_indices[cdr3]
            random.shuffle(indices)
            for idx in indices:
                if idx not in used_indices:
                    current_batch.append(idx)
                    used_indices.add(idx)
                    if len(current_batch) == self.batch_size:
                        batches.append(current_batch)
                        current_batch = []
                        break  # Move to the next cdr3 group

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # Shuffle the batches
        random.shuffle(batches)

        # Flatten the list of batches
        flat_batches = [idx for batch in batches for idx in batch]
        return iter(flat_batches)

    def __len__(self):
        return len(self.data_source)

class TCRABpSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        # Group indices by unique TRA.CDR3
        cdr3_to_indices = defaultdict(list)
        for idx in range(len(self.data_source)):
            original_idx = self.data_source.indices[idx]
            row = self.data_source.dataset.data.iloc[original_idx]
            cdr3_to_indices[row['TRB.CDR3']].append(idx)

            

        # Shuffle the groups
        unique_cdr3s = list(cdr3_to_indices.keys())
        random.shuffle(unique_cdr3s)

        # Create batches
        batches = []
        current_batch = []
        used_indices = set()
        for cdr3 in unique_cdr3s:
            indices = cdr3_to_indices[cdr3]
            random.shuffle(indices)
            for idx in indices:
                if idx not in used_indices:
                    current_batch.append(idx)
                    used_indices.add(idx)
                    if len(current_batch) == self.batch_size:
                        batches.append(current_batch)
                        current_batch = []
                        break  # Move to the next cdr3 group

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # Shuffle the batches
        random.shuffle(batches)

        # Flatten the list of batches
        flat_batches = [idx for batch in batches for idx in batch]
        return iter(flat_batches)

    def __len__(self):
        return len(self.data_source)

class TCRApMHCSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        # Group indices by unique TRA.CDR3
        cdr3_to_indices = defaultdict(list)
        for idx in range(len(self.data_source)):
            original_idx = self.data_source.indices[idx]
            row = self.data_source.dataset.data.iloc[original_idx]
            cdr3_to_indices[row['TRA.CDR3']].append(idx)

        # Shuffle the groups
        unique_cdr3s = list(cdr3_to_indices.keys())
        random.shuffle(unique_cdr3s)

        # Create batches
        batches = []
        current_batch = []
        used_indices = set()
        for cdr3 in unique_cdr3s:
            indices = cdr3_to_indices[cdr3]
            random.shuffle(indices)
            for idx in indices:
                if idx not in used_indices:
                    current_batch.append(idx)
                    used_indices.add(idx)
                    if len(current_batch) == self.batch_size:
                        batches.append(current_batch)
                        current_batch = []
                        break  # Move to the next cdr3 group

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # Shuffle the batches
        random.shuffle(batches)

        # Flatten the list of batches
        flat_batches = [idx for batch in batches for idx in batch]
        return iter(flat_batches)

    def __len__(self):
        return len(self.data_source)

class TCRBpMHCSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        # Group indices by unique TRA.CDR3
        cdr3_to_indices = defaultdict(list)
        for idx in range(len(self.data_source)):
            original_idx = self.data_source.indices[idx]
            row = self.data_source.dataset.data.iloc[original_idx]
            cdr3_to_indices[row['TRB.CDR3']].append(idx)

        # Shuffle the groups
        unique_cdr3s = list(cdr3_to_indices.keys())
        random.shuffle(unique_cdr3s)

        # Create batches
        batches = []
        current_batch = []
        used_indices = set()
        for cdr3 in unique_cdr3s:
            indices = cdr3_to_indices[cdr3]
            random.shuffle(indices)
            for idx in indices:
                if idx not in used_indices:
                    current_batch.append(idx)
                    used_indices.add(idx)
                    if len(current_batch) == self.batch_size:
                        batches.append(current_batch)
                        current_batch = []
                        break  # Move to the next cdr3 group

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        # Shuffle the batches
        random.shuffle(batches)

        # Flatten the list of batches
        flat_batches = [idx for batch in batches for idx in batch]
        return iter(flat_batches)

    def __len__(self):
        return len(self.data_source)