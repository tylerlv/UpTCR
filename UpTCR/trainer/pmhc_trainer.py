import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import manifold
import pandas as pd
import os

from .base import BaseTrainer
from ..model.encoder import MHCEncoder, EpitopeEncoder, TCRAEncoder, TCRBEncoder
from ..model.fusion import pMHCFusion, TCRABFusion
from ..model.finetuner import Finetuner_TCRABpMHC, Finetuner_pMHC
from ..model.finetuner import Finetuner_TCRBpMHC, Finetuner_TCRBp, Finetuner_TCRABp
from ..model.finetuner import Finetuner_Structure
from ..model.utils import EarlyStopping
from ..utils.misc import get_scores_dist

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from sklearn.metrics.pairwise import cosine_similarity

class pMHCTrainer(BaseTrainer):
    def __init__(self, config):
        super(pMHCTrainer, self).__init__(config)

        self.MHCEncoder = MHCEncoder(**config.MHCEncoder)
        self.EpitopeEncoder = EpitopeEncoder(**config.EpitopeEncoder)
        self.pMHCFusion = pMHCFusion(**config.pMHCFusion)
        self.Finetuner = Finetuner_pMHC(**config.Finetuner)

        self.MHCEncoder.to(self.device)
        self.EpitopeEncoder.to(self.device)
        self.pMHCFusion.to(self.device)
        self.Finetuner.to(self.device)


        self.optimizer, self.scheduler = self.configure_optimizer(
            list(self.MHCEncoder.parameters()) + list(self.EpitopeEncoder.parameters()) + 
            list(self.pMHCFusion.parameters()) + list(self.Finetuner.parameters()), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)

        self.loss_fn = nn.MSELoss()

        self.best_train_pearson = 0
        self.best_val_pearson = 0

        self.epitope_model_path = config.pretrained_model.epitope_model_path
        self.mhc_model_path = config.pretrained_model.mhc_model_path
        self.pmhc_model_path = config.pretrained_model.pmhc_model_path

        self.initialize_model()
    
    def initialize_model(self):
        """Initialize model weights from pretrained models."""
        self.MHCEncoder.load_state_dict(torch.load(self.mhc_model_path))
        self.EpitopeEncoder.load_state_dict(torch.load(self.epitope_model_path))
        self.pMHCFusion.load_state_dict(torch.load(self.pmhc_model_path))
        print("Pretrained models loaded.")
    
    def proceed_one_epoch(self, iteration, training=True, is_save=False):
        if training:
            self.MHCEncoder.train()
            self.EpitopeEncoder.train()
            self.pMHCFusion.train()
            self.Finetuner.train()
        else:
            self.MHCEncoder.eval()
            self.EpitopeEncoder.eval()
            self.pMHCFusion.eval()
            self.Finetuner.eval()
        
        avg_loss = 0
        data_size = 0
        all_preds = []
        all_labels = []

        all_results = []

        for batch in tqdm(iteration):
            loss, pred, label = self.compute_loss(batch, training=training)
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss += loss
            data_size += 1

            # Convert tensors to numpy arrays and append to list
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

            if is_save:
                epitope_name = batch['epitope_name']
                mhc_name = batch['mhc_name']

                for i in range(len(epitope_name)):
                    all_results.append({
                        'MHCseq': mhc_name[i],
                        'peptide': epitope_name[i],
                        'binder': label[i].item(),
                        'prediction': pred[i].item()
                    })

        # Concatenate all the numpy arrays
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        mse, pearson_corr, spearman_corr, concordance_idx, auc, aupr = self.cal_metric(all_preds, all_labels)

        if is_save:
            return avg_loss / data_size, mse, pearson_corr, spearman_corr, concordance_idx, auc, aupr, all_results
        else:
            return avg_loss / data_size, mse, pearson_corr, spearman_corr, concordance_idx, auc, aupr
    
    def train_one_epoch(self, iteration):
        return self.proceed_one_epoch(iteration, training=True)
    
    def eval_one_epoch(self, iteration, is_save=False):
        return self.proceed_one_epoch(iteration, training=False, is_save=is_save)
    
    def fit(self, train_loader, val_loader, test_loader=None):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_loss, train_mse, train_pearson_corr, train_spearman_corr, train_concordance_idx, train_auc, train_aupr = self.train_one_epoch(train_loader)
            val_loss, val_mse, val_pearson_corr, val_spearman_corr, val_concordance_idx, val_auc, val_aupr = self.eval_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train Loss: {train_loss:.3f}, Train mse: {train_mse:.3f}, Train Pearson: {train_pearson_corr:.3f}, Train Spearman: {train_spearman_corr}, Train AUC: {train_auc:.3f}, Train AUPR: {train_aupr:.3f}')
            self.logger.info(f'Val Loss: {val_loss:.3f}, Val mse: {val_mse:.3f}, Val Pearson: {val_pearson_corr:.3f}, Val Spearman: {val_spearman_corr}, Val AUC: {val_auc:.3f}, Val AUPR: {val_aupr:.3f}')

            self.writer.add_scalar('Train/Loss', train_loss, epoch+1)
            self.writer.add_scalar('Valid/Loss', val_loss, epoch+1)

            if train_pearson_corr > self.best_train_pearson:
                self.best_train_pearson = train_pearson_corr
                self.save_model(model_type='best_train')

            if val_pearson_corr > self.best_val_pearson:
                self.best_val_pearson = val_pearson_corr
                self.save_model(model_type='best_val')
            
            if test_loader is not None:
                self.test(test_loader, 'best_val')

        self.writer.close()
    
    def save_model(self, model_type):
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type}.pt"
        pmhc_model_path = f"{self.save_dir}/pMHCFusion_{model_type}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type}.pt"
        
        torch.save(self.MHCEncoder.state_dict(), mhc_model_path)
        torch.save(self.EpitopeEncoder.state_dict(), epitope_model_path)
        torch.save(self.pMHCFusion.state_dict(), pmhc_model_path)
        torch.save(self.Finetuner.state_dict(), finetuner_model_path)
        
        self.logger.info(f"Saved model to {model_type}")
    
    def load_model(self, epitope_model_path=None, 
                    mhc_model_path=None, 
                    pmhc_model_path=None, 
                    finetuner_model_path=None):
        self.EpitopeEncoder.load_state_dict(torch.load(epitope_model_path))
        self.MHCEncoder.load_state_dict(torch.load(mhc_model_path))
        self.pMHCFusion.load_state_dict(torch.load(pmhc_model_path))
        self.Finetuner.load_state_dict(torch.load(finetuner_model_path))

    def test(self, test_loader, model_type, is_save=False, task='None'):
        self.configure_logger(self.log_dir, 'testing')
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type}.pt"
        pmhc_model_path = f"{self.save_dir}/pMHCFusion_{model_type}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type}.pt"
        self.load_model(epitope_model_path, mhc_model_path, pmhc_model_path, finetuner_model_path)

        if is_save:
            test_loss, test_mse, test_pearson_corr, test_spearman_corr, test_concordance_idx, test_auc, test_aupr, test_results = self.eval_one_epoch(test_loader, is_save)
            self.logger.info(f'Model ({model_type}) Test Loss: {test_loss:.3f}')
            self.logger.info(f'Model ({model_type}) Test mse: {test_mse:.3f}, Test pearson: {test_pearson_corr:.3f}, Test spearman: {test_spearman_corr}, Test AUC: {test_auc:.3f}, Test AUPR: {test_aupr:.3f}')
            results_df = pd.DataFrame(test_results)
            results_df.to_csv(f'{task}_{model_type}.csv', mode='a', header=not os.path.exists('inference_results.csv'), index=False)
        else:
            test_loss, test_mse, test_pearson_corr, test_spearman_corr, test_concordance_idx, test_auc, test_aupr = self.eval_one_epoch(test_loader, is_save)
            self.logger.info(f'Model ({model_type}) Test Loss: {test_loss:.3f}')
            self.logger.info(f'Model ({model_type}) Test mse: {test_mse:.3f}, Test pearson: {test_pearson_corr:.3f}, Test spearman: {test_spearman_corr}, Test AUC: {test_auc:.3f}, Test AUPR: {test_aupr:.3f}')

    def compute_loss(self, batch, training):
        epitope_seq = batch['epitope_seq'].to(self.device)
        epitope_emb = batch['epitope_emb'].to(self.device)
        epitope_mask = batch['epitope_mask'].to(self.device)
        mhc = batch['mhc'].to(self.device)
        mhc_mask = batch['mhc_mask'].to(self.device)
        label = batch['label'].to(self.device)

        epitope_name = batch['epitope_name']
        mhc_name = batch['mhc_name']

        if training:
            mhc_emb, mhc_pooling, mhc_mask = self.MHCEncoder(mhc, mhc_mask)
            epi_emb, epitope_pooling, epitope_mask = self.EpitopeEncoder(epitope_seq, epitope_emb, epitope_mask)
            pmhc, _, pmhc_mask, logit_scale = self.pMHCFusion(mhc_emb, epi_emb, mhc_mask, epitope_mask)

            pred = self.Finetuner(pmhc, pmhc_mask)
        
        else:
            with torch.no_grad():
                mhc_emb, mhc_pooling, mhc_mask = self.MHCEncoder(mhc, mhc_mask)
                epi_emb, epitope_pooling, epitope_mask = self.EpitopeEncoder(epitope_seq, epitope_emb, epitope_mask)
                pmhc, _, pmhc_mask, logit_scale = self.pMHCFusion(mhc_emb, epi_emb, mhc_mask, epitope_mask)

                pred = self.Finetuner(pmhc, pmhc_mask)
        
        loss = self.loss_fn(pred, label)
        return loss, pred, label
    
    def cal_metric(self, all_preds, all_labels):
        # Calculate regression metrics
        mse = mean_squared_error(all_labels, all_preds)
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        spearman_corr, _ = spearmanr(all_labels, all_preds)
        concordance_idx = concordance_index(all_labels, all_preds)

        # Define CUTOFF
        CUTOFF = 1.0 - math.log(500) / math.log(50000)

        # Calculate AUC and AUPR using the CUTOFF
        auc = roc_auc_score(np.array(all_labels) >= CUTOFF, all_preds)
        aupr = average_precision_score(np.array(all_labels) >= CUTOFF, all_preds)

        return mse, pearson_corr, spearman_corr, concordance_idx, auc, aupr