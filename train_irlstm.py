import os
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import DataLoader
import dataloader
from tqdm import tqdm
from models import irlstm

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score # classification

class evaluate_DX():
    def __init__(self):
        self.preds = []
        self.targets_num = []
        self.probs = []
        self.targets_cat = []
        self.auc = 0
        self.N = 0

    def average(self):
        acc = accuracy_score(self.targets_num, self.preds)
        pre = precision_score(self.targets_num, self.preds, average="macro")
        rec = recall_score(self.targets_num, self.preds, average="macro")
        auc = roc_auc_score(self.targets_cat, self.probs, average="macro", multi_class='ovr')
        return acc, pre, rec, auc 
    
    def measure(self, logits, targets, mask):
        """
        logits : B x L x 3
        targets: B x L x 3
        mask   : B x L x 3
        """
        # print(logits.shape, targets.shape, mask.shape)
        probs = logits.softmax(-1)[mask].detach().cpu().numpy().reshape(-1, 3)
        targets_cat = targets[mask].detach().cpu().numpy().reshape(-1, 3)

        preds   = torch.argmax(logits[mask].reshape(-1, 3), dim=-1).detach().cpu().numpy()
        targets_num = torch.argmax(targets[mask].reshape(-1, 3), dim=-1).detach().cpu().numpy()
        

        if len(self.preds) == 0:
            self.preds = preds
            self.targets_num = targets_num
            self.probs = probs
            self.targets_cat = targets_cat
        else:
            self.preds       = np.concatenate((self.preds, preds), axis=0)
            self.targets_num = np.concatenate((self.targets_num, targets_num), axis=0)
            self.probs       = np.concatenate((self.probs, probs), axis=0)
            self.targets_cat = np.concatenate((self.targets_cat, targets_cat), axis=0)


def dxLoss(logits, target, mask):
    """
    logits: B x L x 3
    target: B x L x 3
    mask  : B x L x 3
    """
    num_classes =  logits.shape[-1]
    return torch.nn.functional.cross_entropy(logits[mask].reshape(-1, num_classes), target[mask].reshape(-1, num_classes), reduction='mean')

def bioLoss(logits, target, mask): 
    """
    y    : B x L x D_cs
    mask : B x L x D_cs
    """    
    return (abs(logits - target)[mask]).mean()

import torch.nn.functional as F
def strictly_increasing_loss(y_pred):
    """
    Enforces y_pred to be strictly increasing along the sequence dimension.

    Args:
    - y_pred (tensor): Shape (B, L, D), predicted sequence.

    Returns:
    - loss (tensor): Scalar loss value.
    """
    B, L, D = y_pred.shape
    loss = 0.0

    for f in range(D):
        if f < D -1: ## All features except the last one should decrease
            for i in range(L - 1):
                for j in range(i + 1, L):
                    diff = y_pred[:, j, f] - y_pred[:, i, f]  # Should be negative for decreasing values
                    loss += torch.mean(F.relu(diff))  # Penalize non-decreasing values
        elif f == D - 1: # ventricle should increase
            for i in range(L - 1):
                for j in range(i + 1, L):
                    diff = y_pred[:, i, f] - y_pred[:, j, f]  # Should be negative for increasing values
                    loss += torch.mean(F.relu(diff))  # Penalize non-increasing values

    return loss

def constraint_values(y_start, y_pred):
    """
    Enforces y_pred to be strictly increasing along the sequence dimension.

    Args:
    - y_pred (tensor): Shape (B, L, D), predicted sequence.

    Returns:
    - loss (tensor): Scalar loss value.
    """
    B, L, D = y_pred.shape
    loss = 0.0
    for l in range(L):
        value =  0.5 * (l+1) / L # torch.ones(B).to(y_start.device) *
        loss += torch.mean(F.relu(abs(y_pred[:, l] - y_start)-value))  # Penalize non-decreasing values

    return loss

if __name__=="__main__":

    data_plit = pd.read_csv("../dataset/data_ptid_train_val_test_split_longitudinal.csv")
    frame = pd.read_csv("../dataset/data_cleaned_for_longitudinal.csv")
    frame = frame[(frame["MRI"] == 1) * (frame["DX"].notnull())]
    ##### 
    sex_mapping = {'Male': 0., 'Female': 1.}
    frame["sex"] = frame["PTGENDER"].replace(sex_mapping)

    dx_mapping = {'Dementia': 1., 'MCI': 0.5, 'CN': 0.}
    frame["diagnosis"]= frame['DX'].replace(dx_mapping)
    ## Normalize
    frame["cerebral_cortex"]       = (frame["cerebral_cortex"] - 490000) / (750000- 490000)
    frame["hippocampus"]           = (frame["hippocampus"] - 5000) / (14000- 5000)
    frame["amygdala"]              = (frame["amygdala"] - 1700) / (5600- 1700)
    frame["cerebral_white_matter"] = (frame["cerebral_white_matter"] - 449000) / (751000- 449000)
    frame["lateral_ventricle"]     = (frame["lateral_ventricle"] - 13000) / (179000- 13000)

    print()
    frame["AGE"] = frame["AGE"] / 100.
    ###

    list_ptid_train = list(data_plit[data_plit["train"] == 1.]["PTID"])
    list_ptid_valid = list(data_plit[data_plit["val"] == 1.]["PTID"])

    frame_train = frame[frame["PTID"].isin(list_ptid_train)]
    frame_valid = frame[frame["PTID"].isin(list_ptid_valid)]


    train_set = dataloader.MedDataset(frame_train, length=20, mode="train")
    valid_set = dataloader.MedDataset(frame_valid, length=20, mode="valid")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader
    }
    #######################################################
    verbose = True
    log = print if verbose else lambda *x, **i: None
    np.random.seed(10)
    torch.manual_seed(10)
    ######################## LOAD MODEL  ###############################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = irlstm.IRLSTM(dim_hid=64, num_bio=5, num_classes=3, length_out=20, device=device)
    model.to(device)

    #################### TRAINING SETTING #############################
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    # if "IMAGE" in model_config.DATA_TYPE:
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=model_config.MILESTONES, gamma=model_config.GAMMA)
    ###################### TRAINING PHASE ###############################
    ###

    filename = "model.pt"
    data_training = []
    min_loss_dx = np.inf
    acc_max, mauc_max = 0, 0

    for epoch in range(100):
        loss_total_train, loss_impute_train, loss_dx_train = 0, 0, 0
        loss_total_valid, loss_impute_valid, loss_dx_valid = 0, 0, 0
        
        dx_metrics_train    = evaluate_DX()
        dx_metrics_valid    = evaluate_DX()
        model.train()
        for sample in tqdm(dataloaders['train']):

            data = sample['data'].to(device)
            mask_bio = sample['mask_bio'].to(device)
            mask_dx = sample['mask_dx'].to(device)
   
            pred_dx, pred_bio = model(data[:, 0].detach())
            ## Calculate loss
            loss_bio    = bioLoss(pred_bio, data[:, 1:, :5].detach(), mask_bio[:, 1:].detach()) ### get target data from 2nd timepoints
            loss_dx     = dxLoss(pred_dx, data[:, 1:, -3:].detach(), mask_dx[:, 1:].detach())
            loss_total  = loss_bio + loss_dx + 0.01 *strictly_increasing_loss(pred_bio) + 0.1 * constraint_values(data[:, 0, :5], pred_bio)

            loss_total_train += loss_total.item() 
            loss_impute_train += loss_bio.item()
            loss_dx_train += loss_dx.item()
            ### UPdate paras
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            ######################## 
            dx_metrics_train.measure(pred_dx, data[:, 1:, -3:], mask_dx[:, 1:])


        ###
        loss_dx_train     = loss_dx_train / len(dataloaders['train'])
        loss_impute_train = loss_impute_train / len(dataloaders['train'])
        loss_total_train  = loss_total_train / len(dataloaders['train'])

        acc_train, pre_train, rec_train, auc_train = dx_metrics_train.average()

        ######################################################################################## 
        model.eval()  
        with torch.no_grad():
            for sample in tqdm(dataloaders['valid']):

                data = sample['data'].to(device)
                mask_bio = sample['mask_bio'].to(device)
                mask_dx = sample['mask_dx'].to(device)
    
                pred_dx, pred_bio = model(data[:, 0].detach())
                ## Calculate loss
                loss_bio    = bioLoss(pred_bio, data[:, 1:, :5].detach(), mask_bio[:, 1:].detach()) ### get target data from 2nd timepoints
                loss_dx     = dxLoss(pred_dx, data[:, 1:, -3:].detach(), mask_dx[:, 1:].detach())
                loss_total  = loss_bio + loss_dx + 0.01 * strictly_increasing_loss(pred_bio) + 0.1 * constraint_values(data[:, 0, :5], pred_bio)

                loss_total_valid += loss_total.item() 
                loss_impute_valid += loss_bio.item()    
                loss_dx_valid += loss_dx.item()  
                ######################## 
                dx_metrics_valid.measure(pred_dx, data[:, 1:, -3:], mask_dx[:, 1:])


                ##########
            loss_dx_valid     = loss_dx_valid / len(dataloaders['valid'])
            loss_impute_valid = loss_impute_valid / len(dataloaders['valid'])
            loss_total_valid  = loss_total_valid / len(dataloaders['valid'])

            acc_valid, pre_valid, rec_valid, auc_valid = dx_metrics_valid.average()
        # Update learning rate
        # if "IMAGE" in model_config.DATA_TYPE:
        #     scheduler.step()
        ###################################################
                    
        # Save model
        # if min_loss_dx > loss_dx_valid:
        #     min_loss_dx = loss_dx_valid
        # if acc_max < acc_valid:
        #     acc_max = acc_valid
            # if os.path.exists(filename):
            #     os.remove(filename)    
        
        filename = "./checkpoint/irlstm/ep_{}_acc_{:.04f}_mauc_{:.04f}_mae_{:.04f}.pth".format(epoch, acc_valid, auc_valid, loss_impute_valid)

        torch.save(model.state_dict(), filename)
        print("Saving model: ", filename)

        print("Traing")
        print("Epoch: {} | Loss_train: {:.04f}  Loss_Impute: {:.04f} |Loss_DX {:.04f}".format(epoch, loss_total_train,loss_impute_train, loss_dx_train))
        print("Epoch: {} | Acc_train:  {:.04f} | Pre_train:  {:.04f} | Rec_train:  {:.04f} | mAUC_train:  {:.04f}".format(epoch, acc_train, pre_train, rec_train, auc_train))
        print("Valid")
        print("Epoch: {} | Loss_valid: {:.04f} | Loss_Impute: {:.04f} | Loss_DX {:.04f}".format(epoch, loss_total_valid, loss_impute_valid, loss_dx_valid))
        print("Epoch: {} | Acc_valid:  {:.04f} | Pre_valid:  {:.04f} | Rec_valid:  {:.04f} | mAUC_valid:  {:.04f}".format(epoch, acc_valid, pre_valid, rec_valid, auc_valid))




