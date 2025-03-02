import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch 
import os 
import random 
from monai.data.image_reader import NumpyReader
import config

##################################################################################
class ImageDataset(Dataset): ### For training Autoencoder
    
    def __init__(self, data_frame, path_root,transform=None, mode='train'):
        self.data_frame = data_frame
        self.path_root=path_root
        self.transform=transform
        if mode not in ['train', 'test', 'valid']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __len__(self):
        return len(self.data_frame)
    
    def preprocess_image(self, path):
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        assert 0 <= img.min() < img.max() <= 1, "Please normlaize the image : {}".format(path)
        return img

    def __getitem__(self, index):
        sample = dict()
        patientInfo = self.data_frame.iloc[index]
        patientid = self.data_frame.iloc[index]["PTID"]
        sample['id'] = patientid
        month = int(patientInfo["Month"])
        # sample["age"] = np.asarray(patientInfo["AGE"]).reshape(1)
        mri = self.preprocess_image(os.path.join(self.path_root, patientid, "visit_{:0>3}_MRI.nii".format(month))).astype(np.float32)
        mri = torch.from_numpy(mri).unsqueeze(0)

        sample["img"]  = mri

        return sample
    
#########################################

class LatentDataset(Dataset): ## For training latent diffusion
    
    def __init__(self, data_frame, path_root, transform=None, mode='train'):
        self.data_frame = data_frame
        self.path_root=path_root
        self.transform=transform
        if mode not in ['train', 'test', 'valid']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode
        self.reader = NumpyReader(npz_keys=['data'])

    def __len__(self):
        return len(self.data_frame)


    def __getitem__(self, index):
        sample = dict()
        patientInfo = self.data_frame.iloc[index]
        patientid = self.data_frame.iloc[index]["PTID"]
        sample['id'] = patientid
        month = int(patientInfo["Month"])
        # sample["age"] = np.asarray(patientInfo["AGE"]).reshape(1)
        latent = self.reader.read(os.path.join(self.path_root, patientid + "_visit_{:0>3}_latent.npz".format(month))).astype(np.float32)
        # latent = torch.from_numpy(latent).unsqueeze(0)
        sample["latent"] = latent
        
        # Apply transformations if provided
        if self.transform is not None:

            # Apply transform
            sample = self.transform(sample)
        
        context = torch.from_numpy(np.asarray(patientInfo[config.CONDITIONING_VARIABLES]).astype(np.float32))
        sample["context"] = context.unsqueeze(0)
        return sample
    
####################################################################  


class LongitudinalDataset(Dataset): ## For training controlnet
    
    def __init__(self, data_frame, path_root, transform=None, latent=True, mode='train'):
        self.data_frame = data_frame # .set_index("PTID")
        self.data_frame = pd.get_dummies(self.data_frame, columns=["DX"], dtype=int)

        min_month_idx = data_frame.groupby("PTID")["Month"].idxmin()
        # # Drop those rows
        self.data_frame_without_bl = data_frame.drop(min_month_idx)

        # self.list_ptid = list(data_frame["PTID"].unique())
        self.path_root=path_root
        self.transform=transform
        if mode not in ['train', 'test', 'valid']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode
        self.latent=latent
        self.reader = NumpyReader(npz_keys=['data'])

    def __len__(self):
        return len(self.data_frame_without_bl)

    def load_sequence_data(self, patientid, month_current, month_followup):

        if self.latent:
            source = self.reader.read(os.path.join(self.path_root, patientid + "_visit_{:0>3}_latent.npz".format(month_current))).astype(np.float32)
            source = torch.from_numpy(source)

            target = self.reader.read(os.path.join(self.path_root, patientid + "_visit_{:0>3}_latent.npz".format(month_followup))).astype(np.float32)
            target = torch.from_numpy(target)
        else:
            source = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path_root, patientid, "visit_{:0>3}_MRI.nii".format(month_current)))) #.astype(np.float32)
            source = torch.from_numpy(source).unsqueeze(0)

            target = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path_root, patientid, "visit_{:0>3}_MRI.nii".format(month_followup)))) #.astype(np.float32)
            target = torch.from_numpy(target).unsqueeze(0)


        return source, target
        
    def __getitem__(self, index):
        sample = dict()
        info_followup = self.data_frame_without_bl.iloc[index]
        patientid = info_followup["PTID"]
        month_followup = int(info_followup["Month"])
        sample['id'] = patientid

        patientInfo = self.data_frame[self.data_frame["PTID"] == patientid]
        month_current = int(sorted(list(patientInfo["Month"][patientInfo["MRI"]==1]))[0])
        assert month_current < month_followup, "month_current must be less than month_followup. Error at {}".format(patientid)

        source, target = self.load_sequence_data(patientid, month_current, month_followup)
        sample["month_curent"] = month_current
        sample["month_followup"] = month_followup
        if self.latent:
            sample["starting_latent"] = source
            sample["followup_latent"] = target
        else:
            sample["starting_image"] = source
            sample["followup_image"] = target

        sample["starting_age"] = torch.from_numpy(np.asarray(patientInfo[patientInfo["Month"] == month_current]["AGE"]).astype(np.float32))

        context = torch.from_numpy(np.asarray(patientInfo[patientInfo["Month"] == month_followup][config.CONDITIONING_VARIABLES]).astype(np.float32))
        sample["context"] = context

        if self.mode == "test":
            columns = config.CONDITIONING_BIO + config.CONDITIONING_DG + ["DX_CN", "DX_MCI", "DX_Dementia"]
            bl_context = torch.from_numpy(np.asarray(patientInfo[patientInfo["Month"] == month_current][columns]).astype(np.float32)).squeeze(0)
            sample["context_bl"] = bl_context # for testing  : this is input of IRLSTM model
        
        return sample

    
####################################################################  
import pandas as pd
import numpy as np

class MedDataset(Dataset): ### For training IRLSTM model
    
    def __init__(self, data_frame, length, mode='train'):
        self.data_frame = data_frame.set_index("PTID")
        self.data_frame = pd.get_dummies(self.data_frame, columns=["DX"], dtype=int)

        self.patients_ID = list(data_frame["PTID"].unique())
        self.length = length # length of the sequence
        if mode not in ['train', 'test', 'valid']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __len__(self):
        return len(self.patients_ID)

    def load_sequence_data(self, patientInfo, list_month):

        data_bio = np.full([self.length + 1, 5], np.nan)
        mask_bio = np.zeros([self.length + 1],  dtype=bool)

        data_dg = np.full([self.length + 1, 3], np.nan)

        data_dx = np.full([self.length + 1, 3], np.nan)
        mask_dx = np.zeros([self.length + 1],  dtype=bool)

        month_bl = list_month[0]
        for i in range(self.length + 1):
            month = month_bl + i * 6
            if month in list_month:
                data_bio[i] = np.array(patientInfo[config.CONDITIONING_BIO][patientInfo["Month"] == month])
                mask_bio[i] = True

                data_dg[i] = np.array(patientInfo[config.CONDITIONING_DX][patientInfo["Month"] == month])

                data_dx[i] = np.array(patientInfo[["DX_CN", "DX_MCI", "DX_Dementia"]][patientInfo["Month"] == month])
                mask_dx[i] = True

        
        ### Fill missing dx'
        old_dx = [1, 0, 0] # ~ CN
        list_idx_miss = []
 
        for j in range(len(data_dx)):
            if np.any(data_dx[j] != data_dx[j]): # Missing data
                if np.all(old_dx == [0, 0, 1]): # Dementia
                    data_dx[j] = old_dx
                else:
                    list_idx_miss.append(j)
            else:
                rec_dx = data_dx[j]
                if len(list_idx_miss) > 0 and np.all(rec_dx == old_dx):
                    for k in list_idx_miss:
                        data_dx[k] = old_dx
                list_idx_miss = [] #reset
                old_dx = rec_dx



        data = np.concatenate([data_bio, data_dg, data_dx], axis=1)
        return data, mask_bio, mask_dx
    
    def __getitem__(self, index):
        sample = dict()

        patientid = self.patients_ID[index]
        sample['id'] = patientid
        patientInfo = self.data_frame.loc[patientid].sort_values("Month", ascending=True)

        list_month_obs_mri = list(patientInfo["Month"][patientInfo["MRI"]==1])
        assert list_month_obs_mri[0] < list_month_obs_mri[-1], "list_month_obs_mri must be sorted. Error at {}".format(patientid)


        data, mask_bio, mask_dx = self.load_sequence_data(patientInfo, list_month_obs_mri)

        sample["data"]  = data.astype(np.float32)
        sample["mask_bio"]  = mask_bio
        sample["mask_dx"]  = mask_dx
   
        return sample

    @staticmethod
    def read_image(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))