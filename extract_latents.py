import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
# from monai import transforms
from models.wasserstein_autoencoder import init_wasserstein_autoencoder
from models import const
import SimpleITK as sitk

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_image', type=str, default="path to preprocessed images")
    parser.add_argument('--path2save', type=str, default="path to save latent feattures")
    parser.add_argument('--aekl_ckpt',   type=str, default="path to autoencoder checkpoint")
    args = parser.parse_args()

    frame = pd.read_csv("../dataset/data_cleaned_for_longitudinal.csv")
    frame = frame[frame["MRI"] == 1]

    ## Load model
    autoencoder = init_wasserstein_autoencoder(args.aekl_ckpt).to(DEVICE).eval()

    os.makedirs(args.path2save, exist_ok=True)

    with torch.no_grad():
        for idx in tqdm(range(len(frame))):
            info = frame.iloc[idx].to_frame().T
            ptid = info["PTID"].item()
            month = info["Month"].item()

            destpath = os.path.join(args.path2save, ptid + "_visit_{:0>3}_latent.npz".format(month) )          
            if os.path.exists(destpath): 
                continue

            mri = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.path_image, ptid, "visit_{:0>3}_MRI.nii".format(month)))).astype(np.float32)

            mri_tensor = torch.Tensor(mri).unsqueeze(0).unsqueeze(0).to(DEVICE)
            mri_latent = autoencoder.encode(mri_tensor)
            mri_latent = mri_latent.detach().cpu().numpy()[0]

            np.savez_compressed(destpath, data=mri_latent)


