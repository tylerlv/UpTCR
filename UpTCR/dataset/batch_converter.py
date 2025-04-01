import torch
import numpy as np

from ..utils.encoding import mhc_encoding, epitope_encoding
from ..utils.encoding import tcra_fv_encoding, tcrb_fv_encoding, tcra_cdr3_encoding, tcrb_cdr3_encoding

class TCRApBatchConverter(object):
    def __init__(self, max_TCRA_fv, max_TCRA_cdr, max_epitope_len=15):
        self.max_epitope_len = max_epitope_len
        self.max_TCRA_fv = max_TCRA_fv
        self.max_TCRA_cdr = max_TCRA_cdr
    
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        epitope_seqs = []
        epitope_embs = []
        epitope_masks = []

        tcra_cdr3_seqs = []
        tcra_fv_seqs = []
        tcra_embs = []
        tcra_cdr3_start_indexs = []
        tcra_masks = []

        epitope_names = []

        for item in raw_batch:
            epitope_seq = item['epitope_seq']
            epitope_emb = item['pep_emb']
            tcra_cdr3_seq = item['cdr3_seq']
            tcra_fv_seq = item['fv_seq']
            tcra_emb = item['tcra_emb']
            tcra_cdr3_start_index = item['cdr3_start_index']

            epitope_names.append(epitope_seq)

            encoded_epitope, mask_epitope = epitope_encoding(epitope_seq, max_seq_len=self.max_epitope_len)
            epitope_seqs.append(torch.tensor(encoded_epitope, dtype=torch.float32))
            epitope_masks.append(torch.tensor(mask_epitope, dtype=torch.float32))

            epitope_embs.append(epitope_emb.clone().detach())

            encoded_tcra_cdr3 = tcra_cdr3_encoding(tcra_cdr3_seq, max_seq_len=self.max_TCRA_cdr)
            encoded_tcra_fv, mask_tcra = tcra_fv_encoding(tcra_fv_seq, max_seq_len=self.max_TCRA_fv)

            tcra_cdr3_seqs.append(torch.tensor(encoded_tcra_cdr3, dtype=torch.float32))
            tcra_fv_seqs.append(torch.tensor(encoded_tcra_fv, dtype=torch.float32))
            tcra_embs.append(tcra_emb.clone().detach())
            tcra_masks.append(torch.tensor(mask_tcra, dtype=torch.float32))
            tcra_cdr3_start_indexs.append(torch.tensor(tcra_cdr3_start_index, dtype=torch.float32))

        epitope_seqs_tensor = torch.stack(epitope_seqs)
        epitope_embs_tensor = torch.stack(epitope_embs)
        epitope_masks_tensor = torch.stack(epitope_masks)

        tcra_cdr3_seqs_tensor = torch.stack(tcra_cdr3_seqs)
        tcra_fv_seqs_tensor = torch.stack(tcra_fv_seqs)
        tcra_embs_tensor = torch.stack(tcra_embs)
        tcra_cdr3_start_indexs_tensor = torch.stack(tcra_cdr3_start_indexs)
        tcra_masks_tensor = torch.stack(tcra_masks)

        data = {}
        data['epitope_name'] = epitope_names
        data['epitope_seq'] = epitope_seqs_tensor
        data['epitope_emb'] = epitope_embs_tensor
        data['epitope_mask'] = epitope_masks_tensor
        data['tcra_cdr3'] = tcra_cdr3_seqs_tensor
        data['tcra_fv'] = tcra_fv_seqs_tensor
        data['tcra_cdr3_start_index'] = tcra_cdr3_start_indexs_tensor
        data['tcra_emb'] = tcra_embs_tensor
        data['tcra_mask'] = tcra_masks_tensor
        return data

class TCRBpBatchConverter(object):
    def __init__(self, max_TCRB_fv, max_TCRB_cdr, max_epitope_len=15):
        self.max_epitope_len = max_epitope_len
        self.max_TCRB_fv = max_TCRB_fv
        self.max_TCRB_cdr = max_TCRB_cdr
    
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        epitope_seqs = []
        epitope_masks = []
        epitope_embs = []
        epitope_names = []

        tcrb_cdr3_seqs = []
        tcrb_fv_seqs = []
        tcrb_embs = []
        tcrb_cdr3_start_indexs = []
        tcrb_masks = []

        for item in raw_batch:
            epitope_seq = item['epitope_seq']
            epitope_emb = item['pep_emb']
            tcrb_cdr3_seq = item['cdr3_seq']
            tcrb_fv_seq = item['fv_seq']
            tcrb_emb = item['tcrb_emb']
            tcrb_cdr3_start_index = item['cdr3_start_index']
            epitope_names.append(epitope_seq)

            encoded_epitope, mask_epitope = epitope_encoding(epitope_seq, max_seq_len=self.max_epitope_len)
            epitope_seqs.append(torch.tensor(encoded_epitope, dtype=torch.float32))
            epitope_masks.append(torch.tensor(mask_epitope, dtype=torch.float32))

            epitope_embs.append(epitope_emb.clone().detach())

            encoded_tcrb_cdr3 = tcrb_cdr3_encoding(tcrb_cdr3_seq, max_seq_len=self.max_TCRB_cdr)
            encoded_tcrb_fv, mask_tcrb = tcrb_fv_encoding(tcrb_fv_seq, max_seq_len=self.max_TCRB_fv)

            tcrb_cdr3_seqs.append(torch.tensor(encoded_tcrb_cdr3, dtype=torch.float32))
            tcrb_fv_seqs.append(torch.tensor(encoded_tcrb_fv, dtype=torch.float32))
            tcrb_embs.append(tcrb_emb.clone().detach())
            tcrb_masks.append(torch.tensor(mask_tcrb, dtype=torch.float32))
            tcrb_cdr3_start_indexs.append(torch.tensor(tcrb_cdr3_start_index, dtype=torch.float32))
        
        epitope_seqs_tensor = torch.stack(epitope_seqs)
        epitope_embs_tensor = torch.stack(epitope_embs)
        epitope_masks_tensor = torch.stack(epitope_masks)

        tcrb_cdr3_seqs_tensor = torch.stack(tcrb_cdr3_seqs)
        tcrb_fv_seqs_tensor = torch.stack(tcrb_fv_seqs)
        tcrb_embs_tensor = torch.stack(tcrb_embs)
        tcrb_cdr3_start_indexs_tensor = torch.stack(tcrb_cdr3_start_indexs)
        tcrb_masks_tensor = torch.stack(tcrb_masks)
        
        data = {}
        data['epitope_name'] = epitope_names
        data['epitope_seq'] = epitope_seqs_tensor
        data['epitope_emb'] = epitope_embs_tensor
        data['epitope_mask'] = epitope_masks_tensor
        data['tcrb_cdr3'] = tcrb_cdr3_seqs_tensor
        data['tcrb_fv'] = tcrb_fv_seqs_tensor
        data['tcrb_cdr3_start_index'] = tcrb_cdr3_start_indexs_tensor
        data['tcrb_emb'] = tcrb_embs_tensor
        data['tcrb_mask'] = tcrb_masks_tensor
        return data


class TCRABpBatchConverter(object):

    def __init__(self, max_TCRA_fv, max_TCRA_cdr, max_TCRB_fv, max_TCRB_cdr, max_epitope_len=15):
        self.max_epitope_len = max_epitope_len
        self.max_TCRA_fv = max_TCRA_fv
        self.max_TCRA_cdr = max_TCRA_cdr
        self.max_TCRB_fv = max_TCRB_fv
        self.max_TCRB_cdr = max_TCRB_cdr
    
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        epitope_seqs = []
        epitope_embs = []
        epitope_masks = []
        epitope_names = []

        tcra_cdr3_seqs = []
        tcra_fv_seqs = []
        tcra_embs = []
        tcra_cdr3_start_indexs = []
        tcra_masks = []

        tcrb_cdr3_seqs = []
        tcrb_fv_seqs = []
        tcrb_embs = []
        tcrb_cdr3_start_indexs = []
        tcrb_masks = []

        for item in raw_batch:
            epitope_seq = item['epitope_seq']
            epitope_emb = item['pep_emb']
            tcra_cdr3_seq = item['tcra_cdr3_seq']
            tcra_fv_seq = item['tcra_fv_seq']
            tcra_emb = item['tcra_emb']
            tcra_cdr3_start_index = item['tcra_cdr3_start_index']
            tcrb_cdr3_seq = item['tcrb_cdr3_seq']
            tcrb_fv_seq = item['tcrb_fv_seq']
            tcrb_emb = item['tcrb_emb']
            tcrb_cdr3_start_index = item['tcrb_cdr3_start_index']

            encoded_epitope, mask_epitope = epitope_encoding(epitope_seq, max_seq_len=self.max_epitope_len)
            epitope_seqs.append(torch.tensor(encoded_epitope, dtype=torch.float32))
            epitope_masks.append(torch.tensor(mask_epitope, dtype=torch.float32))
            epitope_embs.append(epitope_emb.clone().detach())
            epitope_names.append(epitope_seq)

            encoded_tcra_cdr3 = tcra_cdr3_encoding(tcra_cdr3_seq, max_seq_len=self.max_TCRA_cdr)
            encoded_tcra_fv, mask_tcra = tcra_fv_encoding(tcra_fv_seq, max_seq_len=self.max_TCRA_fv)

            encoded_tcrb_cdr3 = tcrb_cdr3_encoding(tcrb_cdr3_seq, max_seq_len=self.max_TCRB_cdr)
            encoded_tcrb_fv, mask_tcrb = tcrb_fv_encoding(tcrb_fv_seq, max_seq_len=self.max_TCRB_fv)

            tcra_cdr3_seqs.append(torch.tensor(encoded_tcra_cdr3, dtype=torch.float32))
            tcra_fv_seqs.append(torch.tensor(encoded_tcra_fv, dtype=torch.float32))
            tcra_embs.append(tcra_emb.clone().detach())
            tcra_masks.append(torch.tensor(mask_tcra, dtype=torch.float32))
            tcra_cdr3_start_indexs.append(torch.tensor(tcra_cdr3_start_index, dtype=torch.float32))

            tcrb_cdr3_seqs.append(torch.tensor(encoded_tcrb_cdr3, dtype=torch.float32))
            tcrb_fv_seqs.append(torch.tensor(encoded_tcrb_fv, dtype=torch.float32))
            tcrb_embs.append(tcrb_emb.clone().detach())
            tcrb_masks.append(torch.tensor(mask_tcrb, dtype=torch.float32))
            tcrb_cdr3_start_indexs.append(torch.tensor(tcrb_cdr3_start_index, dtype=torch.float32))

        epitope_seqs_tensor = torch.stack(epitope_seqs)
        epitope_embs_tensor = torch.stack(epitope_embs)
        epitope_masks_tensor = torch.stack(epitope_masks)

        tcra_cdr3_seqs_tensor = torch.stack(tcra_cdr3_seqs)
        tcra_fv_seqs_tensor = torch.stack(tcra_fv_seqs)
        tcra_embs_tensor = torch.stack(tcra_embs)
        tcra_cdr3_start_indexs_tensor = torch.stack(tcra_cdr3_start_indexs)
        tcra_masks_tensor = torch.stack(tcra_masks)

        tcrb_cdr3_seqs_tensor = torch.stack(tcrb_cdr3_seqs)
        tcrb_fv_seqs_tensor = torch.stack(tcrb_fv_seqs)
        tcrb_embs_tensor = torch.stack(tcrb_embs)
        tcrb_cdr3_start_indexs_tensor = torch.stack(tcrb_cdr3_start_indexs)
        tcrb_masks_tensor = torch.stack(tcrb_masks)
        
        data = {}
        data['epitope_seq'] = epitope_seqs_tensor
        data['epitope_emb'] = epitope_embs_tensor
        data['epitope_mask'] = epitope_masks_tensor
        data['epitope_name'] = epitope_names
        data['tcra_cdr3'] = tcra_cdr3_seqs_tensor
        data['tcra_fv'] = tcra_fv_seqs_tensor
        data['tcra_cdr3_start_index'] = tcra_cdr3_start_indexs_tensor
        data['tcra_emb'] = tcra_embs_tensor
        data['tcra_mask'] = tcra_masks_tensor
        data['tcrb_cdr3'] = tcrb_cdr3_seqs_tensor
        data['tcrb_fv'] = tcrb_fv_seqs_tensor
        data['tcrb_cdr3_start_index'] = tcrb_cdr3_start_indexs_tensor
        data['tcrb_emb'] = tcrb_embs_tensor
        data['tcrb_mask'] = tcrb_masks_tensor
        return data

class TCRApMHCBatchConverter(object):
    def __init__(self, max_TCRA_fv, max_TCRA_cdr, max_epitope_len=15, max_mhc_len=34):
        self.max_epitope_len = max_epitope_len
        self.max_TCRA_fv = max_TCRA_fv
        self.max_TCRA_cdr = max_TCRA_cdr
        self.max_mhc_len = max_mhc_len
    
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        epitope_seqs = []
        epitope_embs = []
        epitope_masks = []

        tcra_cdr3_seqs = []
        tcra_fv_seqs = []
        tcra_embs = []
        tcra_cdr3_start_indexs = []
        tcra_masks = []

        mhc_seqs = []
        mhc_masks = []
        epitope_mhc_names = []

        for item in raw_batch:
            epitope_seq = item['epitope_seq']
            epitope_emb = item['pep_emb']
            tcra_cdr3_seq = item['cdr3_seq']
            tcra_fv_seq = item['fv_seq']
            tcra_emb = item['tcra_emb']
            tcra_cdr3_start_index = item['cdr3_start_index']
            mhcseq = item['mhc']
            
            encoded_epitope, mask_epitope = epitope_encoding(epitope_seq, max_seq_len=self.max_epitope_len)
            epitope_seqs.append(torch.tensor(encoded_epitope, dtype=torch.float32))
            epitope_masks.append(torch.tensor(mask_epitope, dtype=torch.float32))
            epitope_embs.append(epitope_emb.clone().detach())

            epitope_mhc_names.append(epitope_seq+'-'+mhcseq)

            encoded_tcra_cdr3 = tcra_cdr3_encoding(tcra_cdr3_seq, max_seq_len=self.max_TCRA_cdr)
            encoded_tcra_fv, mask_tcra = tcra_fv_encoding(tcra_fv_seq, max_seq_len=self.max_TCRA_fv)

            tcra_cdr3_seqs.append(torch.tensor(encoded_tcra_cdr3, dtype=torch.float32))
            tcra_fv_seqs.append(torch.tensor(encoded_tcra_fv, dtype=torch.float32))
            tcra_embs.append(tcra_emb.clone().detach())
            tcra_masks.append(torch.tensor(mask_tcra, dtype=torch.float32))
            tcra_cdr3_start_indexs.append(torch.tensor(tcra_cdr3_start_index, dtype=torch.float32))

            encoded_mhc, mhc_mask = mhc_encoding(mhcseq, max_seq_len=self.max_mhc_len)
            mhc_seqs.append(torch.tensor(encoded_mhc, dtype=torch.float32))
            mhc_masks.append(torch.tensor(mhc_mask, dtype=torch.float32))

        epitope_seqs_tensor = torch.stack(epitope_seqs)
        epitope_embs_tensor = torch.stack(epitope_embs)
        epitope_masks_tensor = torch.stack(epitope_masks)

        tcra_cdr3_seqs_tensor = torch.stack(tcra_cdr3_seqs)
        tcra_fv_seqs_tensor = torch.stack(tcra_fv_seqs)
        tcra_embs_tensor = torch.stack(tcra_embs)
        tcra_cdr3_start_indexs_tensor = torch.stack(tcra_cdr3_start_indexs)
        tcra_masks_tensor = torch.stack(tcra_masks)

        mhc_seqs_tensor = torch.stack(mhc_seqs)
        mhc_masks_tensor = torch.stack(mhc_masks)

        data = {}
        data['epitope_seq'] = epitope_seqs_tensor
        data['epitope_emb'] = epitope_embs_tensor
        data['epitope_mask'] = epitope_masks_tensor
        data['epitope_mhc_name'] = epitope_mhc_names
        data['tcra_cdr3'] = tcra_cdr3_seqs_tensor
        data['tcra_fv'] = tcra_fv_seqs_tensor
        data['tcra_cdr3_start_index'] = tcra_cdr3_start_indexs_tensor
        data['tcra_emb'] = tcra_embs_tensor
        data['tcra_mask'] = tcra_masks_tensor
        data['mhc'] = mhc_seqs_tensor
        data['mhc_mask'] = mhc_masks_tensor
        return data

class TCRBpMHCBatchConverter(object):
    def __init__(self, max_TCRB_fv, max_TCRB_cdr, max_epitope_len=15, max_mhc_len=34):
        self.max_epitope_len = max_epitope_len
        self.max_TCRB_fv = max_TCRB_fv
        self.max_TCRB_cdr = max_TCRB_cdr
        self.max_mhc_len = max_mhc_len
    
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        epitope_seqs = []
        epitope_masks = []
        epitope_embs = []

        tcrb_cdr3_seqs = []
        tcrb_fv_seqs = []
        tcrb_embs = []
        tcrb_cdr3_start_indexs = []
        tcrb_masks = []

        mhc_seqs = []
        mhc_masks = []
        epitope_mhc_names = []

        for item in raw_batch:
            epitope_seq = item['epitope_seq']
            epitope_emb = item['pep_emb']
            tcrb_cdr3_seq = item['cdr3_seq']
            tcrb_fv_seq = item['fv_seq']
            tcrb_emb = item['tcrb_emb']
            tcrb_cdr3_start_index = item['cdr3_start_index']
            mhcseq = item['mhc']
            

            encoded_epitope, mask_epitope = epitope_encoding(epitope_seq, max_seq_len=self.max_epitope_len)
            epitope_seqs.append(torch.tensor(encoded_epitope, dtype=torch.float32))
            epitope_masks.append(torch.tensor(mask_epitope, dtype=torch.float32))
            epitope_embs.append(epitope_emb.clone().detach())

            epitope_mhc_names.append(epitope_seq+'-'+mhcseq)

            encoded_tcrb_cdr3 = tcrb_cdr3_encoding(tcrb_cdr3_seq, max_seq_len=self.max_TCRB_cdr)
            encoded_tcrb_fv, mask_tcrb = tcrb_fv_encoding(tcrb_fv_seq, max_seq_len=self.max_TCRB_fv)

            tcrb_cdr3_seqs.append(torch.tensor(encoded_tcrb_cdr3, dtype=torch.float32))
            tcrb_fv_seqs.append(torch.tensor(encoded_tcrb_fv, dtype=torch.float32))
            tcrb_embs.append(tcrb_emb.clone().detach())
            tcrb_masks.append(torch.tensor(mask_tcrb, dtype=torch.float32))
            tcrb_cdr3_start_indexs.append(torch.tensor(tcrb_cdr3_start_index, dtype=torch.float32))

            encoded_mhc, mhc_mask = mhc_encoding(mhcseq, max_seq_len=self.max_mhc_len)
            mhc_seqs.append(torch.tensor(encoded_mhc, dtype=torch.float32))
            mhc_masks.append(torch.tensor(mhc_mask, dtype=torch.float32))

        
        epitope_seqs_tensor = torch.stack(epitope_seqs)
        epitope_embs_tensor = torch.stack(epitope_embs)
        epitope_masks_tensor = torch.stack(epitope_masks)

        tcrb_cdr3_seqs_tensor = torch.stack(tcrb_cdr3_seqs)
        tcrb_fv_seqs_tensor = torch.stack(tcrb_fv_seqs)
        tcrb_embs_tensor = torch.stack(tcrb_embs)
        tcrb_cdr3_start_indexs_tensor = torch.stack(tcrb_cdr3_start_indexs)
        tcrb_masks_tensor = torch.stack(tcrb_masks)

        mhc_seqs_tensor = torch.stack(mhc_seqs)
        mhc_masks_tensor = torch.stack(mhc_masks)
        
        data = {}
        data['epitope_seq'] = epitope_seqs_tensor
        data['epitope_emb'] = epitope_embs_tensor
        data['epitope_mask'] = epitope_masks_tensor
        data['epitope_mhc_name'] = epitope_mhc_names
        data['tcrb_cdr3'] = tcrb_cdr3_seqs_tensor
        data['tcrb_fv'] = tcrb_fv_seqs_tensor
        data['tcrb_cdr3_start_index'] = tcrb_cdr3_start_indexs_tensor
        data['tcrb_emb'] = tcrb_embs_tensor
        data['tcrb_mask'] = tcrb_masks_tensor
        data['mhc'] = mhc_seqs_tensor
        data['mhc_mask'] = mhc_masks_tensor
        return data


class TCRABpMHCBatchConverter(object):

    def __init__(self, max_TCRA_fv, max_TCRA_cdr, max_TCRB_fv, max_TCRB_cdr, max_epitope_len=15, max_mhc_len=34):
        self.max_epitope_len = max_epitope_len
        self.max_TCRA_fv = max_TCRA_fv
        self.max_TCRA_cdr = max_TCRA_cdr
        self.max_TCRB_fv = max_TCRB_fv
        self.max_TCRB_cdr = max_TCRB_cdr
        self.max_mhc_len = max_mhc_len
    
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        epitope_seqs = []
        epitope_embs = []
        epitope_masks = []

        tcra_cdr3_seqs = []
        tcra_fv_seqs = []
        tcra_embs = []
        tcra_cdr3_start_indexs = []
        tcra_masks = []

        tcrb_cdr3_seqs = []
        tcrb_fv_seqs = []
        tcrb_embs = []
        tcrb_cdr3_start_indexs = []
        tcrb_masks = []

        mhc_seqs = []
        mhc_masks = []

        labels = []

        epitope_names = []
        tcrb_cdr3_names = []
        tcra_cdr3_names = []
        tcra_fv_names = []
        tcrb_fv_names = []
        mhc_names = []

        pdb_codes = []

        tcrnames = []
        tcrmuts = []

        for item in raw_batch:
            epitope_seq = item['epitope_seq']
            epitope_emb = item['pep_emb']
            tcra_cdr3_seq = item['tcra_cdr3_seq']
            tcra_fv_seq = item['tcra_fv_seq']
            tcra_emb = item['tcra_emb']
            tcra_cdr3_start_index = item['tcra_cdr3_start_index']
            tcrb_cdr3_seq = item['tcrb_cdr3_seq']
            tcrb_fv_seq = item['tcrb_fv_seq']
            tcrb_emb = item['tcrb_emb']
            tcrb_cdr3_start_index = item['tcrb_cdr3_start_index']
            mhcseq = item['mhc']
            label = item['label']

            epitope_name = item['epitope_seq']
            tcrb_cdr3_name = item['tcrb_cdr3_seq']
            tcra_cdr3_name = item['tcra_cdr3_seq']
            mhc_name = item['mhc']
            pdb_code = item['pdb_code']

            epitope_names.append(epitope_name)
            tcrb_cdr3_names.append(tcrb_cdr3_name)
            tcra_cdr3_names.append(tcra_cdr3_name)
            mhc_names.append(mhc_name)
            pdb_codes.append(pdb_code)

            tcrnames.append(item['tcrname'])
            tcrmuts.append(item['tcrmut'])

            tcra_fv_names.append(tcra_fv_seq)
            tcrb_fv_names.append(tcrb_fv_seq)

            encoded_epitope, mask_epitope = epitope_encoding(epitope_seq, max_seq_len=self.max_epitope_len)
            epitope_seqs.append(torch.tensor(encoded_epitope, dtype=torch.float32))
            epitope_masks.append(torch.tensor(mask_epitope, dtype=torch.float32))
            epitope_embs.append(epitope_emb.clone().detach())

            encoded_tcra_cdr3 = tcra_cdr3_encoding(tcra_cdr3_seq, max_seq_len=self.max_TCRA_cdr)
            encoded_tcra_fv, mask_tcra = tcra_fv_encoding(tcra_fv_seq, max_seq_len=self.max_TCRA_fv)
            encoded_tcrb_cdr3 = tcrb_cdr3_encoding(tcrb_cdr3_seq, max_seq_len=self.max_TCRB_cdr)
            encoded_tcrb_fv, mask_tcrb = tcrb_fv_encoding(tcrb_fv_seq, max_seq_len=self.max_TCRB_fv)

            tcra_cdr3_seqs.append(torch.tensor(encoded_tcra_cdr3, dtype=torch.float32))
            tcra_fv_seqs.append(torch.tensor(encoded_tcra_fv, dtype=torch.float32))
            tcra_embs.append(tcra_emb.clone().detach())
            tcra_masks.append(torch.tensor(mask_tcra, dtype=torch.float32))
            tcra_cdr3_start_indexs.append(torch.tensor(tcra_cdr3_start_index, dtype=torch.float32))

            tcrb_cdr3_seqs.append(torch.tensor(encoded_tcrb_cdr3, dtype=torch.float32))
            tcrb_fv_seqs.append(torch.tensor(encoded_tcrb_fv, dtype=torch.float32))
            tcrb_embs.append(tcrb_emb.clone().detach())
            tcrb_masks.append(torch.tensor(mask_tcrb, dtype=torch.float32))
            tcrb_cdr3_start_indexs.append(torch.tensor(tcrb_cdr3_start_index, dtype=torch.float32))

            encoded_mhc, mhc_mask = mhc_encoding(mhcseq, max_seq_len=self.max_mhc_len)
            mhc_seqs.append(torch.tensor(encoded_mhc, dtype=torch.float32))
            mhc_masks.append(torch.tensor(mhc_mask, dtype=torch.float32))
            labels.append(torch.tensor(label, dtype=torch.float32))

        epitope_seqs_tensor = torch.stack(epitope_seqs)
        epitope_embs_tensor = torch.stack(epitope_embs)
        epitope_masks_tensor = torch.stack(epitope_masks)

        tcra_cdr3_seqs_tensor = torch.stack(tcra_cdr3_seqs)
        tcra_fv_seqs_tensor = torch.stack(tcra_fv_seqs)
        tcra_embs_tensor = torch.stack(tcra_embs)
        tcra_cdr3_start_indexs_tensor = torch.stack(tcra_cdr3_start_indexs)
        tcra_masks_tensor = torch.stack(tcra_masks)

        tcrb_cdr3_seqs_tensor = torch.stack(tcrb_cdr3_seqs)
        tcrb_fv_seqs_tensor = torch.stack(tcrb_fv_seqs)
        tcrb_embs_tensor = torch.stack(tcrb_embs)
        tcrb_cdr3_start_indexs_tensor = torch.stack(tcrb_cdr3_start_indexs)
        tcrb_masks_tensor = torch.stack(tcrb_masks)

        mhc_seqs_tensor = torch.stack(mhc_seqs)
        mhc_masks_tensor = torch.stack(mhc_masks)

        labels_tensor = torch.stack(labels)
        
        data = {}
        data['epitope_seq'] = epitope_seqs_tensor
        data['epitope_emb'] = epitope_embs_tensor
        data['epitope_mask'] = epitope_masks_tensor
        data['tcra_cdr3'] = tcra_cdr3_seqs_tensor
        data['tcra_fv'] = tcra_fv_seqs_tensor
        data['tcra_cdr3_start_index'] = tcra_cdr3_start_indexs_tensor
        data['tcra_emb'] = tcra_embs_tensor
        data['tcra_mask'] = tcra_masks_tensor
        data['tcrb_cdr3'] = tcrb_cdr3_seqs_tensor
        data['tcrb_fv'] = tcrb_fv_seqs_tensor
        data['tcrb_cdr3_start_index'] = tcrb_cdr3_start_indexs_tensor
        data['tcrb_emb'] = tcrb_embs_tensor
        data['tcrb_mask'] = tcrb_masks_tensor
        data['mhc'] = mhc_seqs_tensor
        data['mhc_mask'] = mhc_masks_tensor
        data['label'] = labels_tensor

        data['epitope_name'] = epitope_names
        data['tcrb_cdr3_name'] = tcrb_cdr3_names
        data['tcra_cdr3_name'] = tcra_cdr3_names
        data['mhc_name'] = mhc_names
        data['pdb_code'] = pdb_codes

        data['tcrname'] = tcrnames
        data['tcrmut'] = tcrmuts

        data['tcra_fv_name'] = tcra_fv_names
        data['tcrb_fv_name'] = tcrb_fv_names
        return data


class pMHC_finetune_BatchConverter(object):
    
    def __init__(self, max_epitope_len=15, max_mhc_len=34):
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len
    
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        mhc_seqs = []
        mhc_masks = []
        embeddings = []
        epitopes = []
        masks = []
        affinitys = []
        mhc_names = []
        pep_names = []

        for item in raw_batch:
            mhcseq = item['mhc']
            mhcname = item['mhc']
            epitope = item['epitope']
            embedding = item['pep_emb']
            affinity = item['Score']

            encoded_mhc, mhc_mask = mhc_encoding(mhcseq, max_seq_len=self.max_mhc_len)
            mhc_seqs.append(torch.tensor(encoded_mhc, dtype=torch.float32))
            mhc_masks.append(torch.tensor(mhc_mask, dtype=torch.float32))
            mhc_names.append(mhcname)
            pep_names.append(epitope)
            

            encoded_epitope, mask = epitope_encoding(epitope, max_seq_len=self.max_epitope_len)
            epitopes.append(torch.tensor(encoded_epitope, dtype=torch.float32))

            embeddings.append(embedding.clone().detach())

            masks.append(torch.tensor(mask, dtype=torch.float32))

            affinitys.append(torch.tensor(affinity, dtype=torch.float32))


        mhc_seqs_tensor = torch.stack(mhc_seqs)
        mhc_masks_tensor = torch.stack(mhc_masks)
        epitopes_tensor = torch.stack(epitopes)
        embeddings_tensor = torch.stack(embeddings)
        masks_tensor = torch.stack(masks)
        affinitys_tensor = torch.stack(affinitys)

        data = {}
        data['mhc_name'] = mhc_names
        data['mhc'] = mhc_seqs_tensor
        data['mhc_mask'] = mhc_masks_tensor
        data['epitope_seq'] = epitopes_tensor
        data['epitope_emb'] = embeddings_tensor
        data['epitope_mask'] = masks_tensor
        data['label'] = affinitys_tensor
        data['epitope_name'] = pep_names

        return data

def create_mask(matrix, x_max, y_max):
    """
    Create a mask matrix based on the input matrix dimensions.

    Parameters:
    - matrix: 2D numpy array (x, y)
    - x_max: int, maximum number of rows for the mask
    - y_max: int, maximum number of columns for the mask

    Returns:
    - mask: 2D numpy array (x_max, y_max) with 1s at the positions of the input matrix and 0s elsewhere
    """
    # Get the shape of the input matrix
    x, y = matrix.shape
    
    # Initialize the mask with zeros
    mask = np.zeros((x_max, y_max), dtype=int)
    
    # Set the corresponding positions in the mask to 1
    mask[:x, :y] = 1
    
    return mask

def create_dist(matrix, x_max, y_max):
    """
    Create a mask matrix based on the input matrix dimensions, filling with the values from the input matrix.

    Parameters:
    - matrix: 2D numpy array (x, y)
    - x_max: int, maximum number of rows for the mask
    - y_max: int, maximum number of columns for the mask

    Returns:
    - mask: 2D numpy array (x_max, y_max) filled with values from the input matrix at the corresponding positions
    """
    # Get the shape of the input matrix
    x, y = matrix.shape
    
    # Initialize the mask with zeros
    dist = np.zeros((x_max, y_max), dtype=matrix.dtype)  # Use the same dtype as the input matrix

    max_distance = np.max(matrix)
    min_distance = np.min(matrix)

    reversed_distances = max_distance - matrix + min_distance

    normalized_distances = (reversed_distances - np.min(reversed_distances)) / (np.max(reversed_distances) - np.min(reversed_distances))
    
    # Set the corresponding positions in the mask to the values from the input matrix
    dist[:x, :y] = matrix
    
    return dist

class StructureBatchConverter(object):

    def __init__(self, max_TCRA_fv, max_TCRA_cdr, max_TCRB_fv, max_TCRB_cdr, max_epitope_len=15, max_mhc_len=34):
        self.max_epitope_len = max_epitope_len
        self.max_TCRA_fv = max_TCRA_fv
        self.max_TCRA_cdr = max_TCRA_cdr
        self.max_TCRB_fv = max_TCRB_fv
        self.max_TCRB_cdr = max_TCRB_cdr
        self.max_mhc_len = max_mhc_len
    
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        epitope_seqs = []
        epitope_embs = []
        epitope_masks = []

        tcra_cdr3_seqs = []
        tcra_fv_seqs = []
        tcra_embs = []
        tcra_cdr3_start_indexs = []
        tcra_masks = []

        tcrb_cdr3_seqs = []
        tcrb_fv_seqs = []
        tcrb_embs = []
        tcrb_cdr3_start_indexs = []
        tcrb_masks = []

        mhc_seqs = []
        mhc_masks = []

        labels = []

        epitope_names = []
        tcrb_cdr3_names = []
        tcra_cdr3_names = []
        mhc_names = []

        pdb_codes = []

        tcra_mhc_dists = []
        tcrb_mhc_dists = []
        tcra_tcrb_dists = []
        epi_tcra_dists = []
        epi_tcrb_dists = []
        epi_mhc_dists = []

        tcra_mhc_dist_masks = []
        tcrb_mhc_dist_masks = []
        tcra_tcrb_dist_masks = []
        epi_tcra_dist_masks = []
        epi_tcrb_dist_masks = []
        epi_mhc_dist_masks = []


        for item in raw_batch:
            epitope_seq = item['epitope_seq']
            epitope_emb = item['pep_emb']
            tcra_cdr3_seq = item['tcra_cdr3_seq']
            tcra_fv_seq = item['tcra_fv_seq']
            tcra_emb = item['tcra_emb']
            tcra_cdr3_start_index = item['tcra_cdr3_start_index']
            tcrb_cdr3_seq = item['tcrb_cdr3_seq']
            tcrb_fv_seq = item['tcrb_fv_seq']
            tcrb_emb = item['tcrb_emb']
            tcrb_cdr3_start_index = item['tcrb_cdr3_start_index']
            mhcseq = item['mhc']

            epitope_name = item['epitope_seq']
            tcrb_cdr3_name = item['tcrb_cdr3_seq']
            tcra_cdr3_name = item['tcra_cdr3_seq']
            mhc_name = item['mhc']
            pdb_code = item['pdb_code']

            labels = item['label']
            tcra_mhc_dist = labels['tcra_mhc']['dist']
            tcrb_mhc_dist = labels['tcrb_mhc']['dist']
            tcra_tcrb_dist = labels['tcra_b']['dist']
            epi_tcra_dist = labels['p_tcra']['dist']
            epi_tcrb_dist = labels['p_tcrb']['dist']
            epi_mhc_dist = labels['p_mhc']['dist']

            tcra_mhc_dist_mask = create_mask(tcra_mhc_dist, self.max_TCRA_fv, self.max_mhc_len)
            tcrb_mhc_dist_mask = create_mask(tcrb_mhc_dist, self.max_TCRB_fv, self.max_mhc_len)
            tcra_tcrb_dist_mask = create_mask(tcra_tcrb_dist, self.max_TCRA_fv, self.max_TCRB_fv)
            epi_tcra_dist_mask = create_mask(epi_tcra_dist, self.max_epitope_len, self.max_TCRA_fv)
            epi_tcrb_dist_mask = create_mask(epi_tcrb_dist, self.max_epitope_len, self.max_TCRB_fv)
            epi_mhc_dist_mask = create_mask(epi_mhc_dist, self.max_epitope_len, self.max_mhc_len)

            tcra_mhc_dists.append(torch.tensor(create_dist(tcra_mhc_dist, self.max_TCRA_fv, self.max_mhc_len), dtype=torch.float32))
            tcrb_mhc_dists.append(torch.tensor(create_dist(tcrb_mhc_dist, self.max_TCRB_fv, self.max_mhc_len), dtype=torch.float32))
            tcra_tcrb_dists.append(torch.tensor(create_dist(tcra_tcrb_dist, self.max_TCRA_fv, self.max_TCRB_fv), dtype=torch.float32))
            epi_tcra_dists.append(torch.tensor(create_dist(epi_tcra_dist, self.max_epitope_len, self.max_TCRA_fv), dtype=torch.float32))
            epi_tcrb_dists.append(torch.tensor(create_dist(epi_tcrb_dist, self.max_epitope_len, self.max_TCRB_fv), dtype=torch.float32))
            epi_mhc_dists.append(torch.tensor(create_dist(epi_mhc_dist, self.max_epitope_len, self.max_mhc_len), dtype=torch.float32))

            tcra_mhc_dist_masks.append(torch.tensor(tcra_mhc_dist_mask, dtype=torch.bool))
            tcrb_mhc_dist_masks.append(torch.tensor(tcrb_mhc_dist_mask, dtype=torch.bool))
            tcra_tcrb_dist_masks.append(torch.tensor(tcra_tcrb_dist_mask, dtype=torch.bool))
            epi_tcra_dist_masks.append(torch.tensor(epi_tcra_dist_mask, dtype=torch.bool))
            epi_tcrb_dist_masks.append(torch.tensor(epi_tcrb_dist_mask, dtype=torch.bool))
            epi_mhc_dist_masks.append(torch.tensor(epi_mhc_dist_mask, dtype=torch.bool))

            epitope_names.append(epitope_name)
            tcrb_cdr3_names.append(tcrb_cdr3_name)
            tcra_cdr3_names.append(tcra_cdr3_name)
            mhc_names.append(mhc_name)
            pdb_codes.append(pdb_code)

            encoded_epitope, mask_epitope = epitope_encoding(epitope_seq, max_seq_len=self.max_epitope_len)
            epitope_seqs.append(torch.tensor(encoded_epitope, dtype=torch.float32))
            epitope_masks.append(torch.tensor(mask_epitope, dtype=torch.float32))
            epitope_embs.append(epitope_emb.clone().detach())

            encoded_tcra_cdr3 = tcra_cdr3_encoding(tcra_cdr3_seq, max_seq_len=self.max_TCRA_cdr)
            encoded_tcra_fv, mask_tcra = tcra_fv_encoding(tcra_fv_seq, max_seq_len=self.max_TCRA_fv)
            encoded_tcrb_cdr3 = tcrb_cdr3_encoding(tcrb_cdr3_seq, max_seq_len=self.max_TCRB_cdr)
            encoded_tcrb_fv, mask_tcrb = tcrb_fv_encoding(tcrb_fv_seq, max_seq_len=self.max_TCRB_fv)

            tcra_cdr3_seqs.append(torch.tensor(encoded_tcra_cdr3, dtype=torch.float32))
            tcra_fv_seqs.append(torch.tensor(encoded_tcra_fv, dtype=torch.float32))
            tcra_embs.append(tcra_emb.clone().detach())
            tcra_masks.append(torch.tensor(mask_tcra, dtype=torch.float32))
            tcra_cdr3_start_indexs.append(torch.tensor(tcra_cdr3_start_index, dtype=torch.float32))

            tcrb_cdr3_seqs.append(torch.tensor(encoded_tcrb_cdr3, dtype=torch.float32))
            tcrb_fv_seqs.append(torch.tensor(encoded_tcrb_fv, dtype=torch.float32))
            tcrb_embs.append(tcrb_emb.clone().detach())
            tcrb_masks.append(torch.tensor(mask_tcrb, dtype=torch.float32))
            tcrb_cdr3_start_indexs.append(torch.tensor(tcrb_cdr3_start_index, dtype=torch.float32))

            encoded_mhc, mhc_mask = mhc_encoding(mhcseq, max_seq_len=self.max_mhc_len)
            mhc_seqs.append(torch.tensor(encoded_mhc, dtype=torch.float32))
            mhc_masks.append(torch.tensor(mhc_mask, dtype=torch.float32))

        epitope_seqs_tensor = torch.stack(epitope_seqs)
        epitope_embs_tensor = torch.stack(epitope_embs)
        epitope_masks_tensor = torch.stack(epitope_masks)

        tcra_cdr3_seqs_tensor = torch.stack(tcra_cdr3_seqs)
        tcra_fv_seqs_tensor = torch.stack(tcra_fv_seqs)
        tcra_embs_tensor = torch.stack(tcra_embs)
        tcra_cdr3_start_indexs_tensor = torch.stack(tcra_cdr3_start_indexs)
        tcra_masks_tensor = torch.stack(tcra_masks)

        tcrb_cdr3_seqs_tensor = torch.stack(tcrb_cdr3_seqs)
        tcrb_fv_seqs_tensor = torch.stack(tcrb_fv_seqs)
        tcrb_embs_tensor = torch.stack(tcrb_embs)
        tcrb_cdr3_start_indexs_tensor = torch.stack(tcrb_cdr3_start_indexs)
        tcrb_masks_tensor = torch.stack(tcrb_masks)

        mhc_seqs_tensor = torch.stack(mhc_seqs)
        mhc_masks_tensor = torch.stack(mhc_masks)

        tcra_mhc_dists_tensor = torch.stack(tcra_mhc_dists)
        tcrb_mhc_dists_tensor = torch.stack(tcrb_mhc_dists)
        tcra_tcrb_dists_tensor = torch.stack(tcra_tcrb_dists)
        epi_tcra_dists_tensor = torch.stack(epi_tcra_dists)
        epi_tcrb_dists_tensor = torch.stack(epi_tcrb_dists)
        epi_mhc_dists_tensor = torch.stack(epi_mhc_dists)

        tcra_mhc_dist_masks_tensor = torch.stack(tcra_mhc_dist_masks)
        tcrb_mhc_dist_masks_tensor = torch.stack(tcrb_mhc_dist_masks)
        tcra_tcrb_dist_masks_tensor = torch.stack(tcra_tcrb_dist_masks)
        epi_tcra_dist_masks_tensor = torch.stack(epi_tcra_dist_masks)
        epi_tcrb_dist_masks_tensor = torch.stack(epi_tcrb_dist_masks)
        epi_mhc_dist_masks_tensor = torch.stack(epi_mhc_dist_masks)
        
        data = {}
        data['epitope_seq'] = epitope_seqs_tensor
        data['epitope_emb'] = epitope_embs_tensor
        data['epitope_mask'] = epitope_masks_tensor
        data['tcra_cdr3'] = tcra_cdr3_seqs_tensor
        data['tcra_fv'] = tcra_fv_seqs_tensor
        data['tcra_cdr3_start_index'] = tcra_cdr3_start_indexs_tensor
        data['tcra_emb'] = tcra_embs_tensor
        data['tcra_mask'] = tcra_masks_tensor
        data['tcrb_cdr3'] = tcrb_cdr3_seqs_tensor
        data['tcrb_fv'] = tcrb_fv_seqs_tensor
        data['tcrb_cdr3_start_index'] = tcrb_cdr3_start_indexs_tensor
        data['tcrb_emb'] = tcrb_embs_tensor
        data['tcrb_mask'] = tcrb_masks_tensor
        data['mhc'] = mhc_seqs_tensor
        data['mhc_mask'] = mhc_masks_tensor

        data['epitope_name'] = epitope_names
        data['tcrb_cdr3_name'] = tcrb_cdr3_names
        data['tcra_cdr3_name'] = tcra_cdr3_names
        data['mhc_name'] = mhc_names
        data['pdb_code'] = pdb_codes

        data['tcra_mhc_dist'] = {
            'dist': tcra_mhc_dists_tensor,
            'mask': tcra_mhc_dist_masks_tensor}
        
        data['tcrb_mhc_dist'] = {
            'dist': tcrb_mhc_dists_tensor,
            'mask': tcrb_mhc_dist_masks_tensor}
        
        data['tcra_tcrb_dist'] = {
            'dist': tcra_tcrb_dists_tensor,
            'mask': tcra_tcrb_dist_masks_tensor}
        
        data['epi_tcra_dist'] = {
            'dist': epi_tcra_dists_tensor,
            'mask': epi_tcra_dist_masks_tensor}
        
        data['epi_tcrb_dist'] = {
            'dist': epi_tcrb_dists_tensor,
            'mask': epi_tcrb_dist_masks_tensor}
        
        data['epi_mhc_dist'] = {
            'dist': epi_mhc_dists_tensor,
            'mask': epi_mhc_dist_masks_tensor}

        return data