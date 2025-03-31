from argparse import ArgumentParser
import os
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from UpTCR.trainer.pmhc_trainer import pMHCTrainer
from UpTCR.dataset.data import pMHC_finetune_Dataset
from UpTCR.dataset.batch_converter import pMHC_finetune_BatchConverter
from UpTCR.utils import set_seed, load_config

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)
    
    if not os.path.exists(config.training.log_dir):
        os.makedirs(config.training.log_dir)
    os.system(f'cp {args.config} {config.training.log_dir}/config.yml')
    
    set_seed(config.training.seed)

    selected_fold = 0

    if config.task == 'pMHC':
        dataset = pMHC_finetune_Dataset(config.data.data_path, config.data.pep_emb_path)

        kf = KFold(n_splits=5, shuffle=True, random_state=3407)

        for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            if fold == selected_fold:
                print(f'Using Fold {fold + 1} for prediction')
                train_data = Subset(dataset, train_indices)
                val_data = Subset(dataset, val_indices)

                print(f'Train data size: {len(train_data)}')
                print(f'Val data size: {len(val_data)}')

                batch_converter = pMHC_finetune_BatchConverter(max_epitope_len=config.data.max_epitope_len, max_mhc_len=config.data.max_mhc_len)

                # Create DataLoaders
                train_loader = DataLoader(train_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                        num_workers=config.training.num_workers, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                        num_workers=config.training.num_workers, shuffle=False)

                Trainer = pMHCTrainer(config)

                Trainer.fit(train_loader, val_loader)
                Trainer.test(val_loader, 'best_val', True, f'Fold{fold+1}')
                break

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-train-pMHC.yml')
    args = parser.parse_args()
    
    main(args)
