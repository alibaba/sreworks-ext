import torch
import torch.nn as nn


class MarginLoss(nn.Module):

    def __init__(self, margin=0.03):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, pred, label):
        batch, label_num = label.shape
        
        label_sort, index_sort = torch.sort(label, dim=-1, descending=True)
        pred_sorted_by_true = pred.gather(dim=1, index=index_sort)
        
        pred_dis = pred_sorted_by_true.unsqueeze(2) - pred_sorted_by_true.unsqueeze(1)
        label_dis =  label_sort.unsqueeze(2) - label_sort.unsqueeze(1)
        
        mask = torch.triu(torch.ones(label_num, label_num), diagonal=2) + torch.tril(torch.ones(label_num, label_num), diagonal=0)
        mask = mask.to(torch.bool).to(label.device)
        dis_dis = self.margin + label_dis - pred_dis
        dis_dis_mask = dis_dis.masked_fill(mask, 0)
        loss = torch.relu(dis_dis_mask)

        return loss.mean()

class ListnetLoss(nn.Module):

    def __init__(self):
        super(ListnetLoss, self).__init__()

    def forward(self, pred, label):

        
        top1_target = torch.softmax(label, dim=-1)
        top1_predict = torch.softmax(pred, dim=-1)
        return torch.mean(-torch.sum(top1_target * torch.log(top1_predict)))

class ListMleLoss(nn.Module):

    def __init__(self):
        super(ListMleLoss, self).__init__()

    def forward(self, y_pred, y_true, k=None):
        if k is not None:
            sublist_indices = (y_pred.shape[1] * torch.rand(size=k)).long()
            y_pred = y_pred[:, sublist_indices] 
            y_true = y_true[:, sublist_indices] 
    
        _, indices = y_true.sort(descending=True, dim=-1)
    
        pred_sorted_by_true = y_pred.gather(dim=1, index=indices)
    
        cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    
        listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true
    
        return listmle_loss.sum(dim=1).mean()

class ThresholdLoss(nn.Module):

    def __init__(self, threshold=0.05, margin_left=0.03, margin_right=0.03):
        super(ThresholdLoss, self).__init__()
        self.threshold = threshold
        self.margin_left = margin_left
        self.margin_right = margin_right

    def forward(self, pred, label):

        
        sign = ((label - self.threshold) + 1e-6) / torch.abs((label - self.threshold) + 1e-6)
        sign = sign.detach()
        
        ts_loss = (0.5 - 0.5 * sign) * (pred - self.threshold + self.margin_left) + (0.5 + 0.5 * sign) * (self.threshold - pred + self.margin_right)

        loss = torch.relu(ts_loss)
        return loss.mean()


