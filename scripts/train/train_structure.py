from argparse import ArgumentParser
import os
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from UpTCR.trainer.structure_trainer import StructureTrainer
from UpTCR.dataset.data import Structure_Dataset
from UpTCR.dataset.batch_converter import StructureBatchConverter
from UpTCR.utils import set_seed, load_config


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)
    
    if not os.path.exists(config.training.log_dir):
        os.makedirs(config.training.log_dir)
    os.system(f'cp {args.config} {config.training.log_dir}/config.yml')
    
    set_seed(config.training.seed)

    if config.task == 'Structure_kfold':
        dataset = Structure_Dataset(config.data.data_path, config.data.TCRA_emb_path, config.data.TCRB_emb_path, config.data.pep_emb_path, config.data.structure_path)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=config.training.seed)

        val_loaders = []
        model_types = []
        
        for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            print(f'Fold {fold + 1}')
            
            train_data = [dataset[i] for i in train_indices]
            val_data = [dataset[i] for i in val_indices]
            
            batch_converter = StructureBatchConverter(max_TCRA_fv=config.data.max_TCRA_fv, 
                                                      max_TCRA_cdr=config.data.max_TCRA_cdr, 
                                                      max_TCRB_fv=config.data.max_TCRB_fv, 
                                                      max_TCRB_cdr=config.data.max_TCRB_cdr, 
                                                      max_epitope_len=config.data.max_epitope_len, 
                                                      max_mhc_len=config.data.max_mhc_len)
            Trainer = StructureTrainer(config)

            print(f'Train data size: {len(train_data)}')
            print(f'Val data size: {len(val_data)}')
            
            train_loader = DataLoader(train_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                      num_workers=config.training.num_workers, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                    num_workers=config.training.num_workers, shuffle=False)

            val_loaders.append(val_loader)
            model_types.append(f'fold{fold+1}_best_val')

            Trainer.fit(train_loader, val_loader, fold+1)
            
        Trainer.test_kfold(val_loaders[0], model_types[0],
                            val_loaders[1], model_types[1],
                            val_loaders[2], model_types[2],
                            val_loaders[3], model_types[3],
                            val_loaders[4], model_types[4])

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-train-Structure.yml')
    args = parser.parse_args()
    
    main(args)
