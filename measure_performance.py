import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm

from models import wasserstein_autoencoder, climb
from models.sampling import sample_using_controlnet_and_z
from models import irlstm

from torch.utils.data import DataLoader

import dataloader
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import mean_squared_error as mse_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from lpips import LPIPS

from generative.losses import PerceptualLoss


device = torch.device("cuda:0")

model_irlstm = irlstm.IRLSTM(dim_hid=64, num_bio=5, num_classes=3, length_out=20, device=device)
model_irlstm.load_state_dict(torch.load("./checkpoint/irlstm.pth"))
model_irlstm.to(device).eval()

autoencoder = wasserstein_autoencoder.init_wasserstein_autoencoder("./checkpoint/autoencoder.pth").to(device).eval()
diffusion = climb.init_latent_diffusion("./checkpoint/diffusion.pth").to(device).eval()
controlnet  = climb.init_controlnet("./checkpoint/controlnet.pth").to(device).eval()

path_root = "/home/ssd1/Phuong/Thesis/src/dataset/TrainingData_nonskull"
path2save = "/home/ssd1/Phuong/Thesis/src/src_our/generated_imgs/swae_v1_47800_diffv4_unet31_cnet_82_ddim_25_las_10_irlstm79"

template = sitk.ReadImage("/home/ssd1/Phuong/Thesis/src/dataset/TrainingData_nonskull/002_S_0295/visit_000_MRI.nii")

print("Done")

perc_loss_fn = PerceptualLoss(spatial_dims=3, 
                              network_type="squeeze",
                              is_fake_3d=True, 
                              fake_3d_ratio=1).to(device)

data_plit = pd.read_csv("./dataset/data_ptid_train_val_test_split_longitudinal.csv")
frame = pd.read_csv("./dataset/data_cleaned_for_longitudinal.csv")
frame = frame[(frame["MRI"] == 1) * (frame["DX"].notnull())]
##### 
sex_mapping = {'Male': 0., 'Female': 1.}
frame["sex"] = frame["PTGENDER"].replace(sex_mapping)

dx_mapping = {'Dementia': 1., 'MCI': 0.5, 'CN': 0.}
frame["diagnosis"]= frame['DX'].replace(dx_mapping)
### normalize

frame["cerebral_cortex"]       = (frame["cerebral_cortex"] - 490000) / (750000- 490000)
frame["hippocampus"]           = (frame["hippocampus"] - 5000) / (14000- 5000)
frame["amygdala"]              = (frame["amygdala"] - 1700) / (5600- 1700)
frame["cerebral_white_matter"] = (frame["cerebral_white_matter"] - 449000) / (751000- 449000)
frame["lateral_ventricle"]     = (frame["lateral_ventricle"] - 13000) / (179000- 13000)

print()
frame["AGE"] = frame["AGE"]/100.

list_ptid_test = list(data_plit[data_plit["test"] == 1.]["PTID"])

frame_test = frame[frame["PTID"].isin(list_ptid_test)]

test_set = dataloader.LongitudinalDataset(data_frame=frame_test, path_root=path_root, latent=False, mode="test")
# Dataloaders:

test_loader = DataLoader(dataset=test_set, 
                          num_workers=2, 
                          batch_size=1, 
                          shuffle=False, 
                          persistent_workers=True, 
                          pin_memory=True)


def predict_bio_dx(model, input, month_followup, month_current):
    """
    input: bio, age, apoe4, sex, dx  BxDi
    output: age, sex, dx, bio   bxLxDo
    """
    pred_dx, pred_bio = model(input)
    # print(0, pred_dx.shape)
    pred_bio =  pred_bio.squeeze(0) # L x 5
    
    pred_dx = pred_dx.argmax(dim=-1) / 2. # [1xL]
    pred_dx = pred_dx.squeeze(0).unsqueeze(-1) ##-> [L x 1]

    dg = input[0, 5:8].unsqueeze(0).repeat(len(pred_bio), 1)

    pred_cond = torch.cat([dg, pred_dx, pred_bio], axis=1) # [L x 9]
    delta = int((month_followup-month_current) // 6 - 1)
    context = pred_cond[delta].unsqueeze(0)
    return context


scale_factor= 1.

mse_total = 0
ssim_total = 0
psnr_total = 0
lpip_total = 0
n= 0

list_out_of_length = []

with torch.no_grad():
    for sample in tqdm(test_loader):
        month_followup, month_current = sample["month_curent"][0].item(), sample["month_followup"][0].item()
        if (month_current-month_followup) // 6 >= 21:
            list_out_of_length.append(sample["id"])
            continue
        ## predicting bio and dx in following 10 years        
        context_bl = sample["context_bl"].float().to(device)
        context = predict_bio_dx(model_irlstm, context_bl, month_followup, month_current).to(device)
        ######### Loading baseline image and its accquisition age
        source = sample["starting_image"].float().to(device)
        starting_a = sample["starting_age"].float().to(device) #.unsqueeze(0)
        starting_z = autoencoder.encode(source)

        ## Loading target image
        target = sample["followup_image"].float().numpy()[0] #.to(device)
        ## Predicting brain images in followup visit

        predicted_image = sample_using_controlnet_and_z(
                        autoencoder=autoencoder, 
                        diffusion=diffusion, 
                        controlnet=controlnet, 
                        starting_z=starting_z, 
                        starting_a=starting_a, 
                        context=context, 
                        device=device,
                        scale_factor=scale_factor,
                        num_inference_steps=25, 
                        average_over_n=10
                    )
        ##### Measure the performance
        mse = ((target[0] - predicted_image)**2).mean() # [4:-4, 8:-8, 4:-4]
        score = ssim_metric(target[0], predicted_image, full=True, data_range=1)[0]
        psnr = psnr_metric(target[0], predicted_image, data_range=1)
        # fid_metric(predicted_image, target[idx])
        lpip = perc_loss_fn(torch.from_numpy(predicted_image).unsqueeze(0).unsqueeze(0).to(device), 
                            torch.from_numpy(target[0]).unsqueeze(0).unsqueeze(0).to(device)).item()
        
        mse_total += mse
        ssim_total += score
        psnr_total += psnr
        lpip_total += lpip
        n += 1
        #
        print(n, mse, score, psnr, lpip, mse_total/n, ssim_total/n, psnr_total/n, lpip_total/n)
        ###
        
        # Create a new image with the source data and target metadata
        new_target_image = sitk.GetImageFromArray(predicted_image)
        new_target_image.SetOrigin(template.GetOrigin())
        new_target_image.SetSpacing(template.GetSpacing())
        new_target_image.SetDirection(template.GetDirection())
        
        # Save the new target NIfTI file
        # os.makedirs( os.path.join(path2save, sample["id"][0]), exist_ok=True)
        # sitk.WriteImage(new_target_image, os.path.join(path2save, sample["id"][0], "visit_{:0>3}_MRI.nii".format(sample["month_followup"][0].item() )))
  
print(len(list_out_of_length), list_out_of_length)