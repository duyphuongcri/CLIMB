import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm

from models.wasserstein_autoencoder import init_wasserstein_autoencoder
from models import climb 
from models.sampling import sample_using_controlnet_and_z
from models import irlstm

import config

device = torch.device("cuda:0")

model_irlstm = irlstm.IRLSTM(dim_hid=64, num_bio=5, num_classes=3, length_out=20, device=device)
model_irlstm.load_state_dict(torch.load("./checkpoint/irlstm.pth"))
model_irlstm.to(device).eval()

autoencoder = init_wasserstein_autoencoder("./checkpoint/autoencoder.pth").to(device).eval()
diffusion = climb.init_latent_diffusion("./checkpoint/diffusion.pth").to(device).eval()
controlnet  = climb.init_controlnet("./checkpoint/controlnet.pth").to(device).eval()

path_root = "/home/ssd1/Phuong/Thesis/src/dataset/TrainingData_nonskull"
path2save = "/home/ssd1/Phuong/Thesis/src/src_our/generated_imgs/swae_v1_47800_diffv4_unet31_cnet_79_ddim_25_las_10_irlstm79_10years"

template = sitk.ReadImage("/home/ssd1/Phuong/Thesis/src/dataset/TrainingData_nonskull/002_S_0295/visit_000_MRI.nii")

print("Done")
#####################################

data_plit = pd.read_csv("../dataset/data_ptid_train_val_test_split_longitudinal.csv")
frame = pd.read_csv("../dataset/data_cleaned_for_longitudinal.csv")
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
frame = pd.get_dummies(frame, columns=["DX"], dtype=int)

list_ptid_test = list(data_plit[data_plit["test"] == 1.]["PTID"])

frame_test = frame[frame["PTID"].isin(list_ptid_test)]

min_month_idx = frame_test.groupby("PTID")["Month"].idxmin()
# #  those rows
frame_test = frame_test.loc[min_month_idx]


def predict_bio_dx(model, input):
    """
    input: bio, age, apoe4, sex, dx  BxDi
    output: age, sex, dx, bio   bxLxDo
    """
    pred_dx, pred_bio = model(input)
    # print(0, pred_dx.shape, pred_bio.shape)
    
    pred_dx = pred_dx.argmax(dim=-1) / 2. # [BxLx1]
    pred_dx = pred_dx.unsqueeze(-1)

    dg = input[:, 5:8].unsqueeze(1).repeat(1, pred_bio.shape[1], 1) # B x L x 3
    for i in range(pred_bio.shape[1]):
        dg[:, i, 0] += (i+1)*6/12/100
    context = torch.cat([dg, pred_dx, pred_bio], axis=-1) # [B x L x 9]

    return context


scale_factor= 1.

mse_total = 0
ssim_total = 0
psnr_total = 0
lpip_total = 0
n= 0

list_out_of_length = []

with torch.no_grad():
    for ptid in list_ptid_test:
        print(ptid)
        ## Predict bio and dx in following 10 years
        columns = config.CONDITIONING_BIO + config.CONDITIONING_DG + ["DX_CN", "DX_MCI", "DX_Dementia"]
        context_bl = torch.from_numpy(np.asarray(frame_test[frame_test["PTID"] == ptid][columns])).float().to(device) #.unsqueeze(0)
        context = predict_bio_dx(model_irlstm, context_bl)
        ######### Loading baseline image and its accquisition age
        min_month = frame_test[frame_test["PTID"] == ptid]["Month"].values[0]
        img = sitk.ReadImage(os.path.join(path_root, ptid, "visit_{:0>3}_MRI.nii".format(min_month)))
        img = sitk.GetArrayFromImage(img)
        img = torch.from_numpy(img).float().to(device).unsqueeze(0).unsqueeze(0)
        latent = autoencoder.encode(img)
        starting_a = torch.tensor([frame_test[frame_test["PTID"] == ptid]["AGE"].values[0]]).float().to(device).unsqueeze(0)
        ### Generating brain images for the next 10 years : 6 months per visit
        for i in range(context.shape[1]):
            # print(context[0, i])
            predicted_image = sample_using_controlnet_and_z(
                            autoencoder=autoencoder, 
                            diffusion=diffusion, 
                            controlnet=controlnet, 
                            starting_z=latent, 
                            starting_a=starting_a, 
                            context=context[:, i], 
                            device=device,
                            scale_factor=scale_factor,
                            num_inference_steps=25, 
                            average_over_n=10
                        )
            ##
            # Create a new image with the source data and target metadata
            new_target_image = sitk.GetImageFromArray(predicted_image)
            new_target_image.SetOrigin(template.GetOrigin())
            new_target_image.SetSpacing(template.GetSpacing())
            new_target_image.SetDirection(template.GetDirection())

            os.makedirs( os.path.join(path2save, ptid), exist_ok=True)
            sitk.WriteImage(new_target_image, os.path.join(path2save, ptid, "visit_{:0>3}_MRI.nii".format((i+1)*6 )))
    
