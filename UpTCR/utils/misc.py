import random

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, median_absolute_error
from scipy.stats import pearsonr
from easydict import EasyDict
import yaml


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def get_scores_dist(y_true, y_pred, y_mask):
    coef, mae, mape = [], [], []
    for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):

        y_true_ = np.array(y_true_)
        y_pred_ = np.array(y_pred_)
        y_mask_ = np.array(y_mask_)

        y_true_ = y_true_[y_mask_.astype('bool')]
        y_pred_ = y_pred_[y_mask_.astype('bool')]
        try:
            # print("ture", y_true_)
            # print("pred", y_pred_)
            coef_, _ = pearsonr(y_true_, y_pred_)
        except Exception:
            coef_ = np.nan
        coef.append(coef_)

        mae_ = median_absolute_error(y_true_, y_pred_)
        mae.append(mae_)

        mape_ = np.median(np.abs((y_true_ - y_pred_) / y_true_))
        mape.append(mape_)
        
    avg_coef = np.nanmean(coef)
    avg_mae = np.mean(mae)
    avg_mape = np.mean(mape)
    return [avg_coef, avg_mae, avg_mape], [coef, mae, mape]