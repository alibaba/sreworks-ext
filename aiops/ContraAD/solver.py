import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import json
from utils.utils import *
# from model.DCdetector import DCdetector
from accelerate import Accelerator

accelerator = Accelerator()

# from model.ChannelSim import *
# from model.ChannelTransformer import ChannelTransformer2D
from model.PointAttention import *
from data_factory.data_loader import get_loader_segment
from einops import rearrange,reduce,repeat
from sklearn.metrics import precision_recall_fscore_support

import warnings

warnings.filterwarnings("ignore")

def normalize(x,method='softmax'):
    # x shape : batch, size, channel
    # min-max normalization
    if len(x.shape) <3:
        x = x.unsqueeze(dim=-1)
    b,w,c = x.shape
    min_vals,_ = torch.min(x,dim=1) # batch,channel
    max_vals,_ = torch.max(x,dim=1) # batch,channel
    mean_vals = torch.mean(x,dim=1)
    std_vals = torch.std(x,dim=1)
    if method == 'min-max':
        min_vals = repeat(min_vals,'b c -> b w c',w=w)
        max_vals = repeat(max_vals,'b c -> b w c',w=w)
        x = (x - min_vals) / (max_vals - min_vals + 1e-8)
    # z-score normalization
    elif method == 'z-score':
        mean_vals = repeat(mean_vals,'b c -> b w c',w=w)
        std_vals = repeat(std_vals,'b c -> b w c',w=w)
        x = torch.abs((x - mean_vals) / (std_vals + 1e-8))
    # softmax normalization
    elif method == 'softmax':
        x = F.softmax(x,dim=1)
    else :
        raise ValueError('Unknown normalization method')
    if c ==1:
        x = x.squeeze(dim=-1)
    return x


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0,win_size=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.win_size = win_size

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif (
            score < self.best_score + self.delta
            or score2 < self.best_score2 + self.delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"early_stopped")
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        accelerator.wait_for_everyone()
        model = accelerator.unwrap_model(model)
        accelerator.save(
            model.state_dict(),
            os.path.join(path, str(self.dataset) + f"_checkpoint_{self.win_size}.pth"),
        )
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


def cdist(x):
    return torch.cdist(x, x)
attns_energy_collect = []
test_labels_collect=[]
vali_loss_collect = []

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(
            self.index,
            "dataset/" + self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="train",
            dataset=self.dataset,
        )
        self.vali_loader = get_loader_segment(
            self.index,
            "dataset/" + self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="val",
            dataset=self.dataset,
        )
        self.test_loader = get_loader_segment(
            self.index,
            "dataset/" + self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="test",
            dataset=self.dataset,
        )
        self.thre_loader = get_loader_segment(
            self.index,
            "dataset/" + self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode="thre",
            dataset=self.dataset,
        )
        print(f"{len(self.vali_loader)} , {len(self.thre_loader)}")
        self.device = accelerator.device  #torch.device(f"cuda:{str(self.gpu)}" if torch.cuda.is_available() else "cpu")

        self.build_model()

        self.loss_mode = 'z_score_clamp'
        self.soft = True
        self.soft_mode= 'min'

        if self.loss_fuc == "MAE":
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == "MSE":
            self.criterion = nn.MSELoss()
        # self.criterion = FeatureDistance()
        self.criterion = PointHingeLoss(mode=self.loss_mode ,soft=self.soft,soft_mode=self.soft_mode)
        if self.mode == 'train':
            print("train")
            self.model,self.optimizer,self.train_loader,self.vali_loader,self.test_loader,self.thre_loader=accelerator.prepare(
                self.model,self.optimizer,self.train_loader,self.vali_loader,self.test_loader,self.thre_loader
            )
        else:
            print("test")
            self.model,self.optimizer,self.train_loader,self.vali_loader,self.test_loader,self.thre_loader=accelerator.prepare(
                self.model,self.optimizer,self.train_loader,self.vali_loader,self.test_loader,self.thre_loader
            )
            # self.model.to(device=0)
            # self.model = accelerator.prepare(self.model)

    def build_model(self):

        self.model = PatchAttention(
            win_size=self.win_size,
            channel=self.input_c,
            depth=4,
            dim=256,
            dim_head=64,
            heads=4,
            attn_dropout=0.2,
            ff_mult=4,
            ff_dropout=0.2,
            hop=3,
            reduction='mean',
            use_RN=True,
            flash_attn=True,
        )
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        print("vali")
        for i, (input_data, _) in enumerate(vali_loader):
            input_data = input_data#.to(self.device)
            z_score = torch.sum(normalize(input_data),dim=-1).to(self.device)
            intra = self.model(input_data)
            loss,_ = self.criterion(intra,z_score)  # + self.criterion(inter).mean()
            all_losses = accelerator.gather(loss)
            vali_loss_collect.extend([ i.item() for i in all_losses])
            
        accelerator.wait_for_everyone()
        a,b = np.average(vali_loss_collect) ,np.average(loss_2)
        return a,b 
        

    def train(self):
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(
            patience=5, verbose=True, dataset_name=self.data_path,win_size=self.win_size
        )
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                # batch,win_size,channel
                iter_count += 1
                input_data = input_data#.to(self.device)
                z_score = torch.sum(normalize(input_data.detach()),dim=-1) # batch win_size
                intra = self.model(input_data)
                loss,_ = self.criterion(intra,z_score.detach())
                self.optimizer.zero_grad()
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()
                accelerator.backward(loss)
                self.optimizer.step()
            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("stopped")
                break
            print(
                "Epoch: {0}, Cost time: {1:.3f}s Vali Loss: {2:.3f} ".format(
                    epoch + 1, time.time() - epoch_time,0.0
                )
            )

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model = accelerator.unwrap_model(self.model)
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    str(self.model_save_path), str(self.data_path) + f"_checkpoint_{self.win_size}.pth"
                )
            )
        )
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input_data = input_data.to(self.device)
            intra = self.model(input_data)
            # inter_dis = cdist(inter) # b c c
            # intra_dis = cdist(intra)  # b w w
            # val,_ = intra_dis.sum(dim=1).min(dim=1)
            # val = repeat(val, "b -> b w", w=self.win_size)
            # metric = F.softmax(normalize(intra_dis.sum(dim=1)/val), dim=1)  # b w
            out = cal_metric(x=intra,z_score=None,mode=self.loss_mode ,soft=self.soft,soft_mode=self.soft_mode,model_mode='test')
            metric = F.softmax(out, dim=1)
            attens_energy = accelerator.gather_for_metrics((metric))
            attns_energy_collect.extend([item.detach().cpu().numpy() for item in attens_energy])
            # attens_energy.append(metric.detach().cpu().numpy())
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            attens_energy = attns_energy_collect.copy()
            attns_energy_collect.clear()
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input_data = input_data.to(self.device)
            intra = self.model(input_data)
            # metric = F.softmax(intra, dim=1)
            # intra_dis = cdist(intra)  # b w w
            # val,_ = intra_dis.sum(dim=1).min(dim=1)
            # val = repeat(val, "b -> b w", w=self.win_size)
            # metric = F.softmax(normalize(intra_dis.sum(dim=1)/val), dim=1)  # b w
            out = cal_metric(x=intra,z_score=None,mode=self.loss_mode ,soft=self.soft,soft_mode=self.soft_mode,model_mode='test')
            metric = F.softmax(out, dim=1)

            # attens_energy.append(metric.detach().cpu().numpy())
            attens_energy = accelerator.gather_for_metrics((metric))
            attns_energy_collect.extend([item.detach().cpu().numpy() for item in attens_energy])

            # self.attens_energy.extend([item.detach().cpu().numpy() for item in attens_energy])
            # attens_energy.append(metric.detach().cpu().numpy())
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            attens_energy = attns_energy_collect.copy()
            attns_energy_collect.clear()
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
     

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        point_labels = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input_data = input_data.to(self.device)
            intra = self.model(input_data)
            # metric = F.softmax(intra, dim=1)
            out = cal_metric(x=intra,z_score=None,mode=self.loss_mode ,soft=self.soft,soft_mode=self.soft_mode,model_mode='test')
            metric = F.softmax(out, dim=1)

            # val,_ = intra_dis.sum(dim=1).min(dim=1)
            # val = repeat(val, "b -> b w", w=self.win_size)
            # metric = F.softmax(normalize(intra_dis.sum(dim=1)/val), dim=1)  # b w
            # attens_energy.append(metric.detach().cpu().numpy())
            # test_labels.append(labels)
            attens_energy,test_labels = accelerator.gather_for_metrics((metric,labels))
            attns_energy_collect.extend([item.detach().cpu().numpy() for item in attens_energy])
            test_labels_collect.extend([item.detach().cpu().numpy() for item in test_labels])
            # test_labels = accelerator.gather(labels)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            attens_energy = attns_energy_collect.copy() #[item.detach().cpu().numpy() for item in attens_energy]
            test_labels =  test_labels_collect.copy() #[item.detach().cpu().numpy() for item in test_labels]
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

            test_energy = np.array(attens_energy)
            test_labels = np.array(test_labels)

            pred = (test_energy > thresh).astype(int)
            gt = test_labels.astype(int)
            print(len(gt),len(pred))

            anomaly_state = False
            for i in range(len(gt)):
                if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    pred[i] = 1

            pred = np.array(pred)
            gt = np.array(gt)

            from sklearn.metrics import precision_recall_fscore_support
            from sklearn.metrics import accuracy_score

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(
                gt, pred, average="binary"
            )
            result_dict = {
                "anomaly_ratio": self.anormly_ratio,
                "win_size": self.win_size,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
                "thre":thresh
            }
            print(result_dict)
            # if not os.path.exists(f"{self.dataset}.log"):
            with open(f"{self.dataset}.log",mode="a") as f:
                f.write(json.dumps(result_dict))
                f.write("\n")

            return accuracy, precision, recall, f_score
