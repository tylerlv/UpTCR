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

class StructureTrainer(BaseTrainer):
    def __init__(self, config):
        super(StructureTrainer, self).__init__(config)

        self.MHCEncoder = MHCEncoder(**config.MHCEncoder)
        self.EpitopeEncoder = EpitopeEncoder(**config.EpitopeEncoder)
        self.TCRAEncoder = TCRAEncoder(**config.TCRAEncoder)
        self.TCRBEncoder = TCRBEncoder(**config.TCRBEncoder)
        self.Finetuner = Finetuner_Structure(**config.StructureFinetuner)

        self.MHCEncoder.to(self.device)
        self.EpitopeEncoder.to(self.device)
        self.TCRAEncoder.to(self.device)
        self.TCRBEncoder.to(self.device)
        self.Finetuner.to(self.device)

        self.optimizer, self.scheduler = self.configure_optimizer(
            list(self.Finetuner.parameters()) + 
            list(self.MHCEncoder.parameters())+
            list(self.EpitopeEncoder.parameters())+
            list(self.TCRAEncoder.parameters())+
            list(self.TCRBEncoder.parameters()), config.training)

        self.epitope_model_path = config.pretrained_model.epitope_model_path
        self.tcra_model_path = config.pretrained_model.tcra_model_path
        self.tcrb_model_path = config.pretrained_model.tcrb_model_path
        self.mhc_model_path = config.pretrained_model.mhc_model_path
        
        self.initialize_model()

        self.best_val_loss = 1e10
        self.best_train_loss = 1e10

        self.vis_dir = config.training.vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def initialize_model(self):
        """Initialize model weights from pretrained models."""
        self.MHCEncoder.load_state_dict(torch.load(self.mhc_model_path))
        self.EpitopeEncoder.load_state_dict(torch.load(self.epitope_model_path))
        self.TCRAEncoder.load_state_dict(torch.load(self.tcra_model_path))
        self.TCRBEncoder.load_state_dict(torch.load(self.tcrb_model_path))
        print("Pretrained models loaded.")

    def get_loss(self, pred, labels):
        dist = labels['dist'].to(self.device)
        mask = labels['mask'].to(self.device)
        dist_loss = F.mse_loss(pred[..., 0], dist, reduction='none')
        dist_loss = dist_loss * mask / (dist + 0.01)

        dist_loss = torch.sum(dist_loss) / torch.sum(mask)
        return dist_loss
    
    def get_scores(self, pred, labels):
        dist, mask = labels
        avg_metrics_dist, metrics_dist = get_scores_dist(dist, pred, mask)
        return avg_metrics_dist, metrics_dist
    
    def proceed_one_epoch(self, iteration, training=True, return_pred=False):
        if training:
            self.MHCEncoder.train()
            self.EpitopeEncoder.train()
            self.TCRAEncoder.train()
            self.TCRBEncoder.train()
            self.Finetuner.train()
        else:
            self.MHCEncoder.eval()
            self.EpitopeEncoder.eval()
            self.TCRAEncoder.eval()
            self.TCRBEncoder.eval()
            self.Finetuner.eval()
        
        avg_loss = 0
        data_size = 0

        pred_tcra_mhc_dist = []
        pred_tcrb_mhc_dist = []
        pred_tcra_tcrb_dist = []
        pred_epi_tcra_dist = []
        pred_epi_tcrb_dist = []
        pred_epi_mhc_dist = []

        label_tcra_mhc_dist = []
        label_tcrb_mhc_dist = []
        label_tcra_tcrb_dist = []
        label_epi_tcra_dist = []
        label_epi_tcrb_dist = []
        label_epi_mhc_dist = []

        mask_tcra_mhc_dist = []
        mask_tcrb_mhc_dist = []
        mask_tcra_tcrb_dist = []
        mask_epi_tcra_dist = []
        mask_epi_tcrb_dist = []
        mask_epi_mhc_dist = []

        for batch in tqdm(iteration):
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

            epitope_name = batch['epitope_name']
            tcrb_cdr3_name = batch['tcrb_cdr3_name']
            mhc_name = batch['mhc_name']

            pdbcode = batch['pdb_code']

            label_tcra_mhc_dist_tmp = batch['tcra_mhc_dist']
            label_tcrb_mhc_dist_tmp = batch['tcrb_mhc_dist']
            label_tcra_tcrb_dist_tmp = batch['tcra_tcrb_dist']
            label_epi_tcra_dist_tmp = batch['epi_tcra_dist']
            label_epi_tcrb_dist_tmp = batch['epi_tcrb_dist']
            label_epi_mhc_dist_tmp = batch['epi_mhc_dist']

            mhc_emb1, mhc_pooling, mhc_mask1 = self.MHCEncoder(mhc, mhc_mask)
            epi_emb1, epi_pooling, epi_mask1 = self.EpitopeEncoder(epitope_seq, epitope_emb, epitope_mask)
            tcra_emb1, tcra_pooling, tcra_mask1, _ = self.TCRAEncoder(tcra_cdr3, tcra_fv, tcra_emb, tcra_mask, tcra_cdr3_start_index)
            tcrb_emb1, tcrb_pooling, tcrb_mask1, _ = self.TCRBEncoder(tcrb_cdr3, tcrb_fv, tcrb_emb, tcrb_mask, tcrb_cdr3_start_index)

            tcra_mhc_dist, tcrb_mhc_dist, tcra_tcrb_dist, epi_tcra_dist, epi_tcrb_dist, epi_mhc_dist = self.Finetuner(tcra_emb1, tcrb_emb1, epi_emb1, mhc_emb1, tcra_mask1, tcrb_mask1, epi_mask1, mhc_mask1)
            
            tcra_mhc_loss = self.get_loss(tcra_mhc_dist, label_tcra_mhc_dist_tmp)
            tcrb_mhc_loss = self.get_loss(tcrb_mhc_dist, label_tcrb_mhc_dist_tmp)
            tcra_tcrb_loss = self.get_loss(tcra_tcrb_dist, label_tcra_tcrb_dist_tmp)
            epi_tcra_loss = self.get_loss(epi_tcra_dist, label_epi_tcra_dist_tmp)
            epi_tcrb_loss = self.get_loss(epi_tcrb_dist, label_epi_tcrb_dist_tmp)
            epi_mhc_loss = self.get_loss(epi_mhc_dist, label_epi_mhc_dist_tmp)

            loss = tcra_mhc_loss + tcrb_mhc_loss + tcra_tcrb_loss + epi_tcra_loss + epi_tcrb_loss + epi_mhc_loss
            avg_loss += loss.item()
            data_size += 1

            pred_tcra_mhc_dist.extend(tcra_mhc_dist[:,:,:,0].detach().cpu().numpy().tolist())
            pred_tcrb_mhc_dist.extend(tcrb_mhc_dist[:,:,:,0].detach().cpu().numpy().tolist())
            pred_tcra_tcrb_dist.extend(tcra_tcrb_dist[:,:,:,0].detach().cpu().numpy().tolist())
            pred_epi_tcra_dist.extend(epi_tcra_dist[:,:,:,0].detach().cpu().numpy().tolist())
            pred_epi_tcrb_dist.extend(epi_tcrb_dist[:,:,:,0].detach().cpu().numpy().tolist())
            pred_epi_mhc_dist.extend(epi_mhc_dist[:,:,:,0].detach().cpu().numpy().tolist())

            label_tcra_mhc_dist.extend(label_tcra_mhc_dist_tmp['dist'].detach().cpu().numpy().tolist())
            label_tcrb_mhc_dist.extend(label_tcrb_mhc_dist_tmp['dist'].detach().cpu().numpy().tolist())
            label_tcra_tcrb_dist.extend(label_tcra_tcrb_dist_tmp['dist'].detach().cpu().numpy().tolist())
            label_epi_tcra_dist.extend(label_epi_tcra_dist_tmp['dist'].detach().cpu().numpy().tolist())
            label_epi_tcrb_dist.extend(label_epi_tcrb_dist_tmp['dist'].detach().cpu().numpy().tolist())
            label_epi_mhc_dist.extend(label_epi_mhc_dist_tmp['dist'].detach().cpu().numpy().tolist())

            mask_tcra_mhc_dist.extend(label_tcra_mhc_dist_tmp['mask'].detach().cpu().numpy().tolist())
            mask_tcrb_mhc_dist.extend(label_tcrb_mhc_dist_tmp['mask'].detach().cpu().numpy().tolist())
            mask_tcra_tcrb_dist.extend(label_tcra_tcrb_dist_tmp['mask'].detach().cpu().numpy().tolist())
            mask_epi_tcra_dist.extend(label_epi_tcra_dist_tmp['mask'].detach().cpu().numpy().tolist())
            mask_epi_tcrb_dist.extend(label_epi_tcrb_dist_tmp['mask'].detach().cpu().numpy().tolist())
            mask_epi_mhc_dist.extend(label_epi_mhc_dist_tmp['mask'].detach().cpu().numpy().tolist())

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        avg_loss /= data_size
        tcra_mhc_score, tcra_mhc_score_sample = self.get_scores(pred_tcra_mhc_dist, [label_tcra_mhc_dist, mask_tcra_mhc_dist])
        tcrb_mhc_score, tcrb_mhc_score_sample = self.get_scores(pred_tcrb_mhc_dist, [label_tcrb_mhc_dist, mask_tcrb_mhc_dist])
        tcra_tcrb_score, tcra_tcrb_score_sample = self.get_scores(pred_tcra_tcrb_dist, [label_tcra_tcrb_dist, mask_tcra_tcrb_dist])
        epi_tcra_score, epi_tcra_score_sample = self.get_scores(pred_epi_tcra_dist, [label_epi_tcra_dist, mask_epi_tcra_dist])
        epi_tcrb_score, epi_tcrb_score_sample = self.get_scores(pred_epi_tcrb_dist, [label_epi_tcrb_dist, mask_epi_tcrb_dist])
        epi_mhc_score, epi_mhc_score_sample = self.get_scores(pred_epi_mhc_dist, [label_epi_mhc_dist, mask_epi_mhc_dist])

        if return_pred:
            return {
                'tcra_mhc': [pred_tcra_mhc_dist, label_tcra_mhc_dist, mask_tcra_mhc_dist],
                'tcrb_mhc': [pred_tcrb_mhc_dist, label_tcrb_mhc_dist, mask_tcrb_mhc_dist],
                'tcra_tcrb': [pred_tcra_tcrb_dist, label_tcra_tcrb_dist, mask_tcra_tcrb_dist],
                'epi_tcra': [pred_epi_tcra_dist, label_epi_tcra_dist, mask_epi_tcra_dist],
                'epi_tcrb': [pred_epi_tcrb_dist, label_epi_tcrb_dist, mask_epi_tcrb_dist],
                'epi_mhc': [pred_epi_mhc_dist, label_epi_mhc_dist, mask_epi_mhc_dist]
            }
        else:
            return avg_loss, tcra_mhc_score[0], tcrb_mhc_score[0], tcra_tcrb_score[0], epi_tcra_score[0], epi_tcrb_score[0], epi_mhc_score[0]
   
    
    def train_one_epoch(self, iteration):
        return self.proceed_one_epoch(iteration, training=True)
    
    def eval_one_epoch(self, iteration, return_pred=False):
        return self.proceed_one_epoch(iteration, training=False, return_pred=return_pred)
    
    def fit(self, train_loader, val_loader, fold=False):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_loss, train_am_corr, train_bm_corr, train_ab_corr, train_pa_corr, train_pb_corr, train_pm_corr = self.train_one_epoch(train_loader)
            val_loss, val_am_corr, val_bm_corr, val_ab_corr, val_pa_corr, val_pb_corr, val_pm_corr = self.eval_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train Loss: {train_loss:.3f}, AM Corr: {train_am_corr:.3f}, BM Corr: {train_bm_corr:.3f}, AB Corr: {train_ab_corr:.3f}, PA Corr: {train_pa_corr:.3f}, PB Corr: {train_pb_corr:.3f}, PM Corr: {train_pm_corr:.3f}')
            self.logger.info(f'Val Loss: {val_loss:.3f}, AM Corr: {val_am_corr:.3f}, BM Corr: {val_bm_corr:.3f}, AB Corr: {val_ab_corr:.3f}, PA Corr: {val_pa_corr:.3f}, PB Corr: {val_pb_corr:.3f}, PM Corr: {val_pm_corr:.3f}')

            self.writer.add_scalar('Train/Loss', train_loss, epoch+1)
            self.writer.add_scalar('Valid/Loss', val_loss, epoch+1)

            if train_loss <= self.best_train_loss:
                self.best_train_loss = train_loss
                if fold:
                    self.save_model(model_type=f'fold{fold}_best_train')
                else:
                    self.save_model(model_type='best_train')

            if val_loss <= self.best_val_loss:
                self.best_val_loss = val_loss
                if fold:
                    self.save_model(model_type=f'fold{fold}_best_val')
                else:
                    self.save_model(model_type='best_val')

        self.writer.close()
    
    def save_model(self, model_type):
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type}.pt"
        
        torch.save(self.TCRAEncoder.state_dict(), tcra_model_path)
        torch.save(self.MHCEncoder.state_dict(), mhc_model_path)
        torch.save(self.EpitopeEncoder.state_dict(), epitope_model_path)
        torch.save(self.TCRBEncoder.state_dict(), tcrb_model_path)
        torch.save(self.Finetuner.state_dict(), finetuner_model_path)
        
        self.logger.info(f"Saved model to {model_type}")
    
    def load_model(self, epitope_model_path=None, 
                    tcra_model_path=None, 
                    tcrb_model_path=None, 
                    mhc_model_path=None, 
                    finetuner_model_path=None):
        self.EpitopeEncoder.load_state_dict(torch.load(epitope_model_path))
        self.TCRAEncoder.load_state_dict(torch.load(tcra_model_path))
        self.TCRBEncoder.load_state_dict(torch.load(tcrb_model_path))
        self.MHCEncoder.load_state_dict(torch.load(mhc_model_path))
        self.Finetuner.load_state_dict(torch.load(finetuner_model_path))
    
    def test(self, test_loader, model_type):
        self.configure_logger(self.log_dir, 'testing')
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type}.pt"
        self.load_model(epitope_model_path, tcra_model_path, tcrb_model_path, mhc_model_path, finetuner_model_path)

        result = self.eval_one_epoch(test_loader, return_pred=True)
        self.vis(*result['epi_tcrb'], 'epi_tcrb')
        self.vis(*result['epi_tcra'], 'epi_tcra')
        self.vis(*result['epi_mhc'], 'epi_mhc')
        self.vis(*result['tcra_tcrb'], 'tcra_tcrb')
        self.vis(*result['tcra_mhc'], 'tcra_mhc')
        self.vis(*result['tcrb_mhc'], 'tcrb_mhc')
    
    def test_kfold(self, test_loader1, model_type1,
                         test_loader2, model_type2,
                         test_loader3, model_type3,
                         test_loader4, model_type4,
                         test_loader5, model_type5):
        self.configure_logger(self.log_dir, 'testing')

        print("fold 1")
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type1}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type1}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type1}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type1}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type1}.pt"
        self.load_model(epitope_model_path, tcra_model_path, tcrb_model_path, mhc_model_path, finetuner_model_path)
        result1 = self.eval_one_epoch(test_loader1, return_pred=True)
        
        print("fold 2")
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type2}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type2}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type2}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type2}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type2}.pt"
        self.load_model(epitope_model_path, tcra_model_path, tcrb_model_path, mhc_model_path, finetuner_model_path)
        result2 = self.eval_one_epoch(test_loader2, return_pred=True)
        
        print("fold 3")
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type3}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type3}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type3}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type3}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type3}.pt"
        self.load_model(epitope_model_path, tcra_model_path, tcrb_model_path, mhc_model_path, finetuner_model_path)
        result3 = self.eval_one_epoch(test_loader3, return_pred=True)
        
        print("fold 4")
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type4}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type4}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type4}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type4}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type4}.pt"
        self.load_model(epitope_model_path, tcra_model_path, tcrb_model_path, mhc_model_path, finetuner_model_path)
        result4 = self.eval_one_epoch(test_loader4, return_pred=True)

        print("fold 5")
        mhc_model_path = f"{self.save_dir}/MHCEncoder_{model_type5}.pt"
        epitope_model_path = f"{self.save_dir}/EpitopeEncoder_{model_type5}.pt"
        tcra_model_path = f"{self.save_dir}/TCRAEncoder{model_type5}.pt"
        tcrb_model_path = f"{self.save_dir}/TCRBEncoder_{model_type5}.pt"
        finetuner_model_path = f"{self.save_dir}/Finetuner_{model_type5}.pt"
        self.load_model(epitope_model_path, tcra_model_path, tcrb_model_path, mhc_model_path, finetuner_model_path)
        result5 = self.eval_one_epoch(test_loader5, return_pred=True)

        self.vis_kfold(*result1['epi_tcrb'], *result2['epi_tcrb'], *result3['epi_tcrb'], *result4['epi_tcrb'], *result5['epi_tcrb'], 'epi_tcrb')
        self.vis_kfold(*result1['epi_tcra'], *result2['epi_tcra'], *result3['epi_tcra'], *result4['epi_tcra'], *result5['epi_tcra'], 'epi_tcra')
        self.vis_kfold(*result1['epi_mhc'], *result2['epi_mhc'], *result3['epi_mhc'], *result4['epi_mhc'], *result5['epi_mhc'], 'epi_mhc')
        self.vis_kfold(*result1['tcra_tcrb'], *result2['tcra_tcrb'], *result3['tcra_tcrb'], *result4['tcra_tcrb'], *result5['tcra_tcrb'], 'tcra_tcrb')
        self.vis_kfold(*result1['tcra_mhc'], *result2['tcra_mhc'], *result3['tcra_mhc'], *result4['tcra_mhc'], *result5['tcra_mhc'], 'tcra_mhc')
        self.vis_kfold(*result1['tcrb_mhc'], *result2['tcrb_mhc'], *result3['tcrb_mhc'], *result4['tcrb_mhc'], *result5['tcrb_mhc'], 'tcrb_mhc')


    def vis_kfold(self, pred_dist1, label_dist1, mask_dist1,
                        pred_dist2, label_dist2, mask_dist2,
                        pred_dist3, label_dist3, mask_dist3,
                        pred_dist4, label_dist4, mask_dist4,
                        pred_dist5, label_dist5, mask_dist5,
                        task):
        pred_dist1 = np.array(pred_dist1)
        label_dist1 = np.array(label_dist1)
        mask_dist1 = np.array(mask_dist1)

        pred_dist2 = np.array(pred_dist2)
        label_dist2 = np.array(label_dist2)
        mask_dist2 = np.array(mask_dist2)

        pred_dist3 = np.array(pred_dist3)
        label_dist3 = np.array(label_dist3)
        mask_dist3 = np.array(mask_dist3)

        pred_dist4 = np.array(pred_dist4)
        label_dist4 = np.array(label_dist4)
        mask_dist4 = np.array(mask_dist4)

        pred_dist5 = np.array(pred_dist5)
        label_dist5 = np.array(label_dist5)
        mask_dist5 = np.array(mask_dist5)

        pred_dist = np.concatenate((pred_dist1, pred_dist2, pred_dist3, pred_dist4, pred_dist5), axis=0)
        label_dist = np.concatenate((label_dist1, label_dist2, label_dist3, label_dist4, label_dist5), axis=0)
        mask_dist = np.concatenate((mask_dist1, mask_dist2, mask_dist3, mask_dist4, mask_dist5), axis=0)

        self.vis(pred_dist, label_dist, mask_dist, task)
   
   
    
    def vis(self, pred_dist, label_dist, mask_dist, task):
        pred_dist = np.array(pred_dist)
        label_dist = np.array(label_dist)
        mask_dist = np.array(mask_dist)

        num_pairs = pred_dist.shape[0]

        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 20

        if task == 'epi_tcrb':
            target_shape = (15, 118)
        elif task == 'epi_tcra':
            target_shape = (15, 117)
        elif task == 'epi_mhc':
            target_shape = (15, 34)
        elif task == 'tcra_tcrb':
            target_shape = (117, 118)
        elif task == 'tcra_mhc':
            target_shape = (117, 34)
        elif task == 'tcrb_mhc':
            target_shape = (118, 34)
        else:
            raise ValueError('Invalid task')

        average_pred_matrix = None
        count = 0

        for i in range(num_pairs):
            matrix = pred_dist[i]
            mask = mask_dist[i]
            
            first_false_row = np.argmax(~mask.any(axis=1))
            first_false_col = np.argmax(~mask.any(axis=0))

            if first_false_col == 0:
                extracted_matrix = matrix[:first_false_row, :]
            else:
                extracted_matrix = matrix[:first_false_row, :first_false_col]

            matrix = extracted_matrix

            original_shape0, original_shape1 = matrix.shape

            x = np.linspace(0, 1, original_shape1)
            y = np.linspace(0, 1, original_shape0)
            f = interp2d(x, y, matrix, kind='linear')

            x_new = np.linspace(0, 1, target_shape[1])
            y_new = np.linspace(0, 1, target_shape[0])

            resized_matrix = f(x_new, y_new)

            min_val = np.min(resized_matrix)
            max_val = np.max(resized_matrix)
            resized_matrix = 1 - (resized_matrix - min_val) / (max_val - min_val)

            if average_pred_matrix is None:
                average_pred_matrix = resized_matrix
            else:
                average_pred_matrix += resized_matrix

            count += 1
        if count > 0:
            average_pred_matrix /= count
        
        height, width = average_pred_matrix.shape
        figsize = (width / 4, height / 4)
        plt.figure(figsize=figsize)

        plt.imshow(average_pred_matrix, cmap='magma', interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.vis_dir}/{task}_average_pred_dist_heatmap.png", dpi=300)
        plt.savefig(f"{self.vis_dir}/{task}_average_pred_dist_heatmap.pdf", dpi=300)
        plt.close()

        average_label_matrix = None
        count = 0

        for i in range(num_pairs):
            matrix = label_dist[i]
            mask = mask_dist[i]

            first_false_row = np.argmax(~mask.any(axis=1))
            first_false_col = np.argmax(~mask.any(axis=0))

            if first_false_col == 0:
                extracted_matrix = matrix[:first_false_row, :]
            else:
                extracted_matrix = matrix[:first_false_row, :first_false_col]

            matrix = extracted_matrix

            original_shape0, original_shape1 = matrix.shape

            x = np.linspace(0, 1, original_shape1)
            y = np.linspace(0, 1, original_shape0)
            f = interp2d(x, y, matrix, kind='linear')

            x_new = np.linspace(0, 1, target_shape[1])
            y_new = np.linspace(0, 1, target_shape[0])

            resized_matrix = f(x_new, y_new)

            min_val = np.min(resized_matrix)
            max_val = np.max(resized_matrix)
            resized_matrix = 1 - (resized_matrix - min_val) / (max_val - min_val)

            if average_label_matrix is None:
                average_label_matrix = resized_matrix
            else:
                average_label_matrix += resized_matrix
            
            count += 1

        if count > 0:
            average_label_matrix /= count
        height, width = average_label_matrix.shape
        figsize = (width / 4, height / 4)
        plt.figure(figsize=figsize)

        plt.imshow(average_label_matrix, cmap='viridis', interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.vis_dir}/{task}_average_true_dist_heatmap.png", dpi=300)
        plt.savefig(f"{self.vis_dir}/{task}_average_true_dist_heatmap.pdf", dpi=300)
        plt.close()

        pred_flat = average_pred_matrix.flatten()
        label_flat = average_label_matrix.flatten()

        valid_indices = np.where((pred_flat >= 0) & (label_flat >= 0))
        pred_flat = pred_flat[valid_indices]
        label_flat = label_flat[valid_indices]

        if len(pred_flat) > 0 and len(label_flat) > 0:
            pearson_corr, _ = pearsonr(pred_flat, label_flat)
        else:
            pearson_corr = np.nan

        plt.figure(figsize=(8, 6))
        plt.scatter(pred_flat, label_flat, alpha=0.6, edgecolors='w', s=100)
        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.text(0.5, 0.9, f'Pearson Correlation: {pearson_corr:.2f}', 
         horizontalalignment='center', verticalalignment='center', 
        )

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{self.vis_dir}/{task}_pearson.png", dpi=300)
        plt.close()