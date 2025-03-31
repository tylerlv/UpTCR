
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

class TCRABpMHCTrainer(BaseTrainer):
    def __init__(self, config):
        super(TCRABpMHCTrainer, self).__init__(config)

        self.MHCEncoder = MHCEncoder(**config.MHCEncoder)
        self.EpitopeEncoder = EpitopeEncoder(**config.EpitopeEncoder)
        self.TCRAEncoder = TCRAEncoder(**config.TCRAEncoder)
        self.TCRBEncoder = TCRBEncoder(**config.TCRBEncoder)
        self.pMHCFusion = pMHCFusion(**config.pMHCFusion)
        self.TCRABFusion = TCRABFusion(**config.TCRABFusion)
        self.Finetuner = Finetuner_TCRABpMHC(**config.Finetuner)

        self.MHCEncoder.to(self.device)
        self.EpitopeEncoder.to(self.device)
        self.TCRAEncoder.to(self.device)
        self.TCRBEncoder.to(self.device)
        self.pMHCFusion.to(self.device)
        self.TCRABFusion.to(self.device)
        self.Finetuner.to(self.device)


        self.optimizer, self.scheduler = self.configure_optimizer(
            list(self.MHCEncoder.parameters()) + list(self.EpitopeEncoder.parameters()) + 
            list(self.TCRAEncoder.parameters()) + list(self.TCRBEncoder.parameters()) +
            list(self.pMHCFusion.parameters()) + list(self.TCRABFusion.parameters())
            +list(self.Finetuner.parameters()), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)

        self.loss_fn = nn.BCELoss()

        self.best_train_aupr = 0
        self.best_val_aupr = 0

        self.epitope_model_path = config.pretrained_model.epitope_model_path
        self.tcra_model_path = config.pretrained_model.tcra_model_path
        self.tcrb_model_path = config.pretrained_model.tcrb_model_path
        self.mhc_model_path = config.pretrained_model.mhc_model_path
        self.pmhc_model_path = config.pretrained_model.pmhc_model_path
        self.tcrab_model_path = config.pretrained_model.tcrab_model_path

        self.initialize_model()
    
    def initialize_model(self):
        """Initialize model weights from pretrained models."""
        self.MHCEncoder.load_state_dict(torch.load(self.mhc_model_path))
        self.EpitopeEncoder.load_state_dict(torch.load(self.epitope_model_path))
        self.TCRAEncoder.load_state_dict(torch.load(self.tcra_model_path))
        self.TCRBEncoder.load_state_dict(torch.load(self.tcrb_model_path))
        self.pMHCFusion.load_state_dict(torch.load(self.pmhc_model_path))
        self.TCRABFusion.load_state_dict(torch.load(self.tcrab_model_path))
        print("Pretrained models loaded.")
    
    def proceed_one_epoch(self, iteration, training=True, is_save=False):
        if training:
            self.MHCEncoder.train()
            self.EpitopeEncoder.train()
            self.TCRAEncoder.train()
            self.TCRBEncoder.train()
            self.pMHCFusion.train()
            self.TCRABFusion.train()
            self.Finetuner.train()
        else:
            self.MHCEncoder.eval()
            self.EpitopeEncoder.eval()
            self.TCRAEncoder.eval()
            self.TCRBEncoder.eval()
            self.pMHCFusion.eval()
            self.TCRABFusion.eval()
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

                tcra_fv = batch['tcra_fv_name']
                tcrb_fv = batch['tcrb_fv_name']
                tcra_cdr3 = batch['tcra_cdr3_name']
                tcrb_cdr3 = batch['tcrb_cdr3_name']

                tcrname = batch['tcrname']
                tcrmut = batch['tcrmut']

                for i in range(len(epitope_name)):
                    all_results.append({
                        'TCRname': tcrname[i],
                        'TCRmut': tcrmut[i],
                        'Epitope': epitope_name[i],
                        'MHCseq': mhc_name[i],
                        'TRA.CDR3': tcra_cdr3[i],
                        'TRB.CDR3': tcrb_cdr3[i],
                        'FV_alpha': tcra_fv[i],
                        'FV_beta': tcrb_fv[i],
                        'binder': label[i].item(),
                        'prediction': pred[i].item()
                    })

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc, auc, aupr, precision, recall, f1 = self.cal_metric(all_preds, all_labels)

        if is_save:
            return avg_loss / data_size, acc, auc, aupr, precision, recall, f1, all_results
        else:
            return avg_loss / data_size, acc, auc, aupr, precision, recall, f1
    
    def train_one_epoch(self, iteration):
        return self.proceed_one_epoch(iteration, training=True)
    
    def eval_one_epoch(self, iteration, is_save=False):
        return self.proceed_one_epoch(iteration, training=False, is_save=is_save)
    
    def fit(self, train_loader, val_loader, test_loader=None):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_loss, train_acc, train_auc, train_aupr, train_precision, train_recall, train_f1 = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_auc, val_aupr, val_precision, val_recall, val_f1 = self.eval_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train AUC: {train_auc:.3f}, Train AUPR: {train_aupr:.3f}')
            self.logger.info(f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val AUC: {val_auc:.3f}, Val AUPR: {val_aupr:.3f}, Val Precision: {val_precision:.3f}, Val Recall: {val_recall:.3f}, Val F1: {val_f1:.3f}')

            self.writer.add_scalar('Train/Loss', train_loss, epoch+1)
            self.writer.add_scalar('Valid/Loss', val_loss, epoch+1)

            if train_aupr > self.best_train_aupr:
                self.best_train_aupr = train_aupr
                self.save_model(model_type='best_train')

            if val_aupr > self.best_val_aupr:
                self.best_val_aupr = val_aupr
                self.save_model(model_type='best_val')

            if test_loader is not None:
                self.test(test_loader, 'best_val')

        self.writer.close()
    
    def save_model(self, model_type):
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type}.pt"
        pmhc_model_path = f"{self.save_dir}/pMHCFusion_{model_type}.pt"
        tcrab_model_path = f"{self.save_dir}/TCRABFusion_{model_type}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type}.pt"
        
        torch.save(self.TCRAEncoder.state_dict(), tcra_model_path)
        torch.save(self.MHCEncoder.state_dict(), mhc_model_path)
        torch.save(self.EpitopeEncoder.state_dict(), epitope_model_path)
        torch.save(self.TCRBEncoder.state_dict(), tcrb_model_path)
        torch.save(self.pMHCFusion.state_dict(), pmhc_model_path)
        torch.save(self.TCRABFusion.state_dict(), tcrab_model_path)
        torch.save(self.Finetuner.state_dict(), finetuner_model_path)
        
        self.logger.info(f"Saved model to {model_type}")

    def load_model(self, epitope_model_path=None, 
                    tcra_model_path=None, 
                    tcrb_model_path=None, 
                    mhc_model_path=None, 
                    pmhc_model_path=None, 
                    tcrab_model_path=None,
                    finetuner_model_path=None):
        self.EpitopeEncoder.load_state_dict(torch.load(epitope_model_path))
        self.TCRAEncoder.load_state_dict(torch.load(tcra_model_path))
        self.TCRBEncoder.load_state_dict(torch.load(tcrb_model_path))
        self.MHCEncoder.load_state_dict(torch.load(mhc_model_path))
        self.pMHCFusion.load_state_dict(torch.load(pmhc_model_path))
        self.TCRABFusion.load_state_dict(torch.load(tcrab_model_path))
        self.Finetuner.load_state_dict(torch.load(finetuner_model_path))
    
    def test(self, test_loader, model_type, is_save=False, task='None'):
        self.configure_logger(self.log_dir, 'testing')
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type}.pt"
        pmhc_model_path = f"{self.save_dir}/pMHCFusion_{model_type}.pt"
        tcrab_model_path = f"{self.save_dir}/TCRABFusion_{model_type}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type}.pt"
        self.load_model(epitope_model_path, tcra_model_path, tcrb_model_path, mhc_model_path, pmhc_model_path, tcrab_model_path, finetuner_model_path)

        if is_save:
            test_loss, test_acc, test_auc, test_aupr, test_precision, test_recall, test_f1, test_results = self.eval_one_epoch(test_loader, is_save)
            self.logger.info(f'Model ({model_type}) Test Loss: {test_loss:.3f}')
            self.logger.info(f'Model ({model_type}) Test Acc: {test_acc:.3f}, Test AUC: {test_auc:.3f}, Test AUPR: {test_aupr:.3f}')
            self.logger.info(f'Model ({model_type}) Test Precision: {test_precision:.3f}, Test Recall: {test_recall:.3f}, Test F1: {test_f1:.3f}')
            results_df = pd.DataFrame(test_results)
            results_df.to_csv(f'{task}_{model_type}.csv', mode='a', header=not os.path.exists('inference_results.csv'), index=False)
        else:
            test_loss, test_acc, test_auc, test_aupr, test_precision, test_recall, test_f1 = self.eval_one_epoch(test_loader, is_save)
            self.logger.info(f'Model ({model_type}) Test Loss: {test_loss:.3f}')
            self.logger.info(f'Model ({model_type}) Test Acc: {test_acc:.3f}, Test AUC: {test_auc:.3f}, Test AUPR: {test_aupr:.3f}')
            self.logger.info(f'Model ({model_type}) Test Precision: {test_precision:.3f}, Test Recall: {test_recall:.3f}, Test F1: {test_f1:.3f}')
    

    def compute_loss(self, batch, training):
        epitope_seq = batch['epitope_seq'].to(self.device)
        epitope_emb = batch['epitope_emb'].to(self.device)
        epitope_mask = batch['epitope_mask'].to(self.device)
        tcra_cdr3 = batch['tcra_cdr3'].to(self.device)
        tcra_fv = batch['tcra_fv'].to(self.device)
        tcra_cdr3_start_index = batch['tcra_cdr3_start_index'].to(self.device)
        tcra_emb = batch['tcra_emb'].to(self.device)
        tcra_mask = batch['tcra_mask'].to(self.device)
        tcrb_cdr3 = batch['tcrb_cdr3'].to(self.device)
        tcrb_fv = batch['tcrb_fv'].to(self.device)
        tcrb_cdr3_start_index = batch['tcrb_cdr3_start_index'].to(self.device)
        tcrb_emb = batch['tcrb_emb'].to(self.device)
        tcrb_mask = batch['tcrb_mask'].to(self.device)
        mhc = batch['mhc'].to(self.device)
        mhc_mask = batch['mhc_mask'].to(self.device)
        label = batch['label'].to(self.device)

        epitope_name = batch['epitope_name']
        tcrb_cdr3_name = batch['tcrb_cdr3_name']
        mhc_name = batch['mhc_name']

        if training:
            mhc_emb, mhc_pooling, mhc_mask = self.MHCEncoder(mhc, mhc_mask)
            epi_emb, epitope_pooling, epitope_mask = self.EpitopeEncoder(epitope_seq, epitope_emb, epitope_mask)
            pmhc, _, pmhc_mask, logit_scale = self.pMHCFusion(mhc_emb, epi_emb, mhc_mask, epitope_mask)

            tcra_emb, tcra_pooling, tcra_mask, _ = self.TCRAEncoder(tcra_cdr3, tcra_fv, tcra_emb, tcra_mask, tcra_cdr3_start_index)
            tcrb_emb, tcrb_pooling, tcrb_mask, _ = self.TCRBEncoder(tcrb_cdr3, tcrb_fv, tcrb_emb, tcrb_mask, tcrb_cdr3_start_index)
            tcrab, _, tcrab_mask, logit_scale = self.TCRABFusion(tcra_emb, tcrb_emb, tcra_mask, tcrb_mask)

            pred = self.Finetuner(tcrab, pmhc, tcrab_mask, pmhc_mask)
        
        else:
            with torch.no_grad():
                mhc_emb, mhc_pooling, mhc_mask = self.MHCEncoder(mhc, mhc_mask)
                epi_emb, epitope_pooling, epitope_mask = self.EpitopeEncoder(epitope_seq, epitope_emb, epitope_mask)
                pmhc, _, pmhc_mask, logit_scale = self.pMHCFusion(mhc_emb, epi_emb, mhc_mask, epitope_mask)

                tcra_emb, tcra_pooling, tcra_mask, _ = self.TCRAEncoder(tcra_cdr3, tcra_fv, tcra_emb, tcra_mask, tcra_cdr3_start_index)
                tcrb_emb, tcrb_pooling, tcrb_mask, _ = self.TCRBEncoder(tcrb_cdr3, tcrb_fv, tcrb_emb, tcrb_mask, tcrb_cdr3_start_index)
                tcrab, _, tcrab_mask, logit_scale = self.TCRABFusion(tcra_emb, tcrb_emb, tcra_mask, tcrb_mask)

                pred = self.Finetuner(tcrab, pmhc, tcrab_mask, pmhc_mask)
        
        loss = self.loss_fn(pred, label)
        return loss, pred, label
        
    
    def cal_metric(self, all_preds, all_labels):
        
        # Calculate metrics
        acc = accuracy_score(all_labels, np.round(all_preds))
        auc = roc_auc_score(all_labels, all_preds)
        aupr = average_precision_score(all_labels, all_preds)
        precision = precision_score(all_labels, np.round(all_preds))
        recall = recall_score(all_labels, np.round(all_preds))
        f1 = f1_score(all_labels, np.round(all_preds))
        return acc, auc, aupr, precision, recall, f1