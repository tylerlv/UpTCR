import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(WeightedContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, mhc, pos, neg, pos_weight, neg_weight):
        pos_distance = torch.norm(mhc - pos, dim=1)
        neg_distance = torch.norm(mhc - neg, dim=1)

        weighted_pos_loss = pos_weight * pos_distance
        weighted_neg_loss = (1-neg_weight) * torch.clamp(self.margin - neg_distance, min=0)  # 通过 margin 调整
        total_loss = torch.mean(weighted_pos_loss + weighted_neg_loss)

        return total_loss 

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def cal_similarity(self, domainA, domainB, logit_scale):
        domainA = F.normalize(domainA, dim=-1)
        domainB = F.normalize(domainB, dim=-1)
        return logit_scale * domainA @ domainB.T

    def get_logits(self, domainA, domainB, logit_scale):
        logits_domainA = self.cal_similarity(domainA, domainB, logit_scale)
        logits_domainB = self.cal_similarity(domainB, domainA, logit_scale)
        return logits_domainA, logits_domainB
    
    def cal_clip_loss(self, domainA, domainB, logit_scale):
        device = domainA.device
        logits_domainA, logits_domainB = self.get_logits(domainA, domainB, logit_scale)
        labels = torch.arange(logits_domainA.shape[0], device=device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_domainA, labels) +
            F.cross_entropy(logits_domainB, labels)
        ) / 2
        return total_loss
    
    def forward(self, domainA, domainB, logit_scale):
        return self.cal_clip_loss(domainA, domainB, logit_scale)

class SoftClipLoss(ClipLoss):
    def __init__(self, beta=0.3, lambda_=0.1, mu=0.1):
        super(SoftClipLoss, self).__init__()
        self.beta = beta
        self.lambda_ = lambda_
        self.mu = mu

    def get_intramodel_logits(self, domain, logit_scale):
        logits_domain = self.cal_similarity(domain, domain, logit_scale)
        return logits_domain

    def cal_soft_loss(self, domainA, domainB, logit_scale):
        A_intramodel_logit = self.get_intramodel_logits(domainA, logit_scale)
        B_intramodel_logit = self.get_intramodel_logits(domainB, logit_scale)
        device = domainA.device
        labels = torch.arange(domainA.shape[0], device=device, dtype=torch.long)

        A_soften_logit = (1 - self.beta) * labels + self.beta * A_intramodel_logit
        B_soften_logit = (1 - self.beta) * labels + self.beta * B_intramodel_logit

        logits_domainA, logits_domainB = self.get_logits(domainA, domainB, logit_scale)

        KL_loss_A = F.kl_div(F.log_softmax(logits_domainA, dim=1), F.softmax(A_soften_logit, dim=1) + 1e-12, reduction='batchmean')
        KL_loss_B = F.kl_div(F.log_softmax(logits_domainB, dim=1), F.softmax(B_soften_logit, dim=1) + 1e-12, reduction='batchmean')

        return (KL_loss_A + KL_loss_B) / 2

    def cal_re_soft_loss(self, domainA, domainB, logit_scale):
        A_intramodel_logit = self.get_intramodel_logits(domainA, logit_scale)
        B_intramodel_logit = self.get_intramodel_logits(domainB, logit_scale)
        device = domainA.device
        labels = torch.arange(domainA.shape[0], device=device, dtype=torch.long)

        A_soften_logit = (1 - self.beta) * labels + self.beta * A_intramodel_logit
        B_soften_logit = (1 - self.beta) * labels + self.beta * B_intramodel_logit

        logits_domainA, logits_domainB = self.get_logits(domainA, domainB, logit_scale)

        mask = ~torch.eye(domainA.shape[0], device=device).bool()
        logits_domainA = logits_domainA[mask].view(domainA.shape[0], -1)
        logits_domainB = logits_domainB[mask].view(domainA.shape[0], -1)
        A_soften_logit = A_soften_logit[mask].view(domainA.shape[0], -1)
        B_soften_logit = B_soften_logit[mask].view(domainA.shape[0], -1)

        KL_loss_A = F.kl_div(F.log_softmax(logits_domainA, dim=1), F.softmax(A_soften_logit, dim=1) + 1e-12, reduction='batchmean')
        KL_loss_B = F.kl_div(F.log_softmax(logits_domainB, dim=1), F.softmax(B_soften_logit, dim=1) + 1e-12, reduction='batchmean')

        return (KL_loss_A + KL_loss_B) / 2

    def cal_softclip_loss(self, domainA, domainB, logit_scale):
        soft_loss = self.cal_soft_loss(domainA, domainB, logit_scale)
        re_soft_loss = self.cal_re_soft_loss(domainA, domainB, logit_scale)
        clip_loss = self.cal_clip_loss(domainA, domainB, logit_scale)
    
        softclip_loss = clip_loss + self.lambda_ * soft_loss + self.mu * re_soft_loss
        return softclip_loss, clip_loss

    def forward(self, domainA, domainB, logit_scale):
        return self.cal_softclip_loss(domainA, domainB, logit_scale)

class MultilabelContrastiveLoss(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, loss_type=None):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super().__init__()
        self.loss_type = loss_type

        self.kl_loss_func = nn.KLDivLoss(reduction="mean")
        self.bce_loss_func = nn.BCELoss(reduction="none")

        self.n_clusters = 4   # for KmeansBalanceBCE


        
    def __kl_criterion(self, logit, label):
        # batchsize = logit.shape[0]
        probs1 = F.log_softmax(logit, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.kl_loss_func(probs1, probs2)
        return loss

    def __cwcl_criterion(self, logit, label):
        logprob = torch.log_softmax(logit, 1)
        loss_per = (label * logprob).sum(1) / (label.sum(1)+1e-6)
        loss = -loss_per.mean()
        return loss
    
    def __infonce_nonvonventional_criterion(self, logit, label):
        logprob = torch.log_softmax(logit, 1)
        loss_per = (label * logprob).sum(1)
        loss = -loss_per.mean()
        return loss
        

    def forward(self, logits, gt_matrix):
        gt_matrix = gt_matrix.to(logits.device)

        if self.loss_type == "KL":
            loss_i = self.__kl_criterion(logits, gt_matrix)
            loss_t = self.__kl_criterion(logits.t(), gt_matrix.t())
            return (loss_i + loss_t) * 0.5
        elif self.loss_type == "BCE":
            # print("BCE is called")
            probs1 = torch.sigmoid(logits)
            probs2 = gt_matrix # torch.sigmoid(gt_matrix)
            bce_loss = self.bce_loss_func(probs1, probs2)
            loss = bce_loss.mean()
            # loss = self.__general_cl_criterion(logits, gt_matrix, "BCE", 
                                            #    use_norm=False, use_negwgt=False)
            return loss
        
        elif self.loss_type in ["BalanceBCE", "WBCE"]:
            probs1 = torch.sigmoid(logits)
            probs2 = gt_matrix # torch.sigmoid(gt_matrix)

            loss_matrix = - probs2 * torch.log(probs1+1e-6) - (1-probs2) * torch.log(1-probs1+1e-6)
            
            pos_mask = (probs2>0.5).detach()
            neg_mask = ~pos_mask

            loss_pos = torch.where(pos_mask, loss_matrix, torch.tensor(0.0, device=probs1.device)).sum()
            loss_neg = torch.where(neg_mask, loss_matrix, torch.tensor(0.0, device=probs1.device)).sum()
            
            loss_pos /= (pos_mask.sum()+1e-6)
            loss_neg /= (neg_mask.sum()+1e-6)

            return (loss_pos+loss_neg)/2

        
        elif self.loss_type in ["NCE", "InfoNCE"]:
            loss_i = self.__infonce_nonvonventional_criterion(logits, gt_matrix)
            loss_t = self.__infonce_nonvonventional_criterion(logits.t(), gt_matrix.t())
            return (loss_i + loss_t) * 0.5

        elif self.loss_type == "MSE":
            probs = torch.sigmoid(logits)
            return F.mse_loss(probs, gt_matrix)
        
        elif self.loss_type == "CWCL":
            loss_i = self.__cwcl_criterion(logits, gt_matrix)
            loss_t = self.__cwcl_criterion(logits.t(), gt_matrix.t())
            return (loss_i + loss_t) * 0.5

        else:
            raise NotImplementedError(self.loss_type)

def test_losses():
    batch_size = 32
    feature_dim = 128
    logit_scale = torch.tensor(0.07, requires_grad=True)

    domainA = torch.randn(batch_size, feature_dim)
    domainB = torch.randn(batch_size, feature_dim)

    clip_loss_fn = ClipLoss()
    soft_clip_loss_fn = SoftClipLoss(beta=0.3, lambda_=1.0, mu=0.5)

    clip_loss = clip_loss_fn(domainA, domainB, logit_scale)
    soft_clip_loss = soft_clip_loss_fn(domainA, domainB, logit_scale)

    print("Clip Loss:", clip_loss.item())
    print("Soft Clip Loss:", soft_clip_loss.item())

if __name__ == "__main__":
    test_losses()
    