from argparse import ArgumentParser
import os
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from UpTCR.trainer.tcrabpm_trainer import TCRABpMHCTrainer
from UpTCR.dataset.data import TCRABpMHCDataset_combine
from UpTCR.dataset.batch_converter import TCRABpMHCBatchConverter
from UpTCR.utils import set_seed, load_config

def main(args):
    config_path = args.config
    config = load_config(config_path)
    
    if not os.path.exists(config.training.log_dir):
        os.makedirs(config.training.log_dir)
    os.system(f'cp {args.config} {config.training.log_dir}/config.yml')
    
    set_seed(config.training.seed)

    if config.task == 'TCRABpMHC_fewshot':
        test_data = TCRABpMHCDataset_combine(config.data.test_data_path, config.data.TCRA_emb_path, config.data.TCRB_emb_path, config.data.pep_emb_path)

        batch_converter = TCRABpMHCBatchConverter(max_TCRA_fv=config.data.max_TCRA_fv, max_TCRA_cdr=config.data.max_TCRA_cdr, 
                                                max_TCRB_fv=config.data.max_TCRB_fv, max_TCRB_cdr=config.data.max_TCRB_cdr, 
                                                max_epitope_len=config.data.max_epitope_len, max_mhc_len=config.data.max_mhc_len)
        Trainer = TCRABpMHCTrainer(config)
    
    print(f'Test data size: {len(test_data)}')
    
    test_loader = DataLoader(test_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                             num_workers=config.training.num_workers, shuffle=False)
    
    Trainer.test(test_loader, 'best_val', True, 'fewshot')

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-train-TCRABpMHC_fewshot.yml')
    args = parser.parse_args()
    
    main(args)
