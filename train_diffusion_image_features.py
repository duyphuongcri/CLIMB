import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai import transforms
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm

from models import utils, wasserstein_autoencoder, climb

from models.sampling import sample_using_controlnet_and_z

import dataloader
import numpy as np
import SimpleITK as sitk

warnings.filterwarnings("ignore")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    controlnet, 
    dataset,
    scale_factor
):
    """
    Visualize the generation on tensorboard
    """
    # resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice( range(len(dataset)), 3 ) 

    for tag_i, i in enumerate(random_indices):

        starting_z = dataset[i]['starting_latent'] * scale_factor
        context    = dataset[i]['context']
        starting_a = dataset[i]['starting_age']

        patientid = dataset[i]['id']
        month_current, month_followup = dataset[i]['month_curent'], dataset[i]['month_followup']
        # print(month_current, month_followup)
        path = "/home/ssd1/Phuong/Thesis/src/dataset/TrainingData_nonskull"
        starting_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, patientid, "visit_{:0>3}_MRI.nii".format(month_current))))
        followup_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, patientid, "visit_{:0>3}_MRI.nii".format(month_followup))))


        # starting_image = torch.from_numpy(starting_image).unsqueeze(0).unsqueeze(0)
        # followup_image = torch.from_numpy(followup_image).unsqueeze(0).unsqueeze(0)

        predicted_image = sample_using_controlnet_and_z(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            controlnet=controlnet, 
            starting_z=starting_z, 
            starting_a=starting_a, 
            context=context, 
            device=DEVICE,
            scale_factor=scale_factor
        )

        utils.tb_display_cond_generation(
            writer=writer, 
            step=epoch, 
            tag=f'{mode}/comparison_{tag_i}',
            starting_image=starting_image, 
            followup_image=followup_image, 
            predicted_image=predicted_image
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_root', type=str, default="./latents/swae_47800")
    # parser.add_argument('--cache_dir',   required=True,   type=str)
    parser.add_argument('--output_dir',  type=str, default=None)
    parser.add_argument('--aekl_ckpt', type=str, default="path to pre-trained autoencoder")
    parser.add_argument('--diff_ckpt', type=str, default="path to pre-trained diffusion")
    parser.add_argument('--cnet_ckpt',   default=None,    type=str)
    parser.add_argument('--num_workers', default=8,       type=int)
    parser.add_argument('--n_epochs',    default=10000,       type=int)
    parser.add_argument('--batch_size',  default=32,      type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    
    args = parser.parse_args()

    

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
    ###

    list_ptid_train = list(data_plit[data_plit["train"] == 1.]["PTID"])
    list_ptid_valid = list(data_plit[data_plit["val"] == 1.]["PTID"])

    frame_train = frame[frame["PTID"].isin(list_ptid_train)]
    frame_valid = frame[frame["PTID"].isin(list_ptid_valid)]

    train_set = dataloader.LongitudinalDataset(data_frame=frame_train, path_root=args.path_root, latent=True, mode="train")
    valid_set = dataloader.LongitudinalDataset(data_frame=frame_valid, path_root=args.path_root, latent=True, mode="valid")
    # Dataloaders:
    train_loader = DataLoader(dataset=train_set, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)

    valid_loader = DataLoader(dataset=valid_set, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)

    autoencoder = wasserstein_autoencoder.init_wasserstein_autoencoder(args.aekl_ckpt)
    diffusion   = climb.init_latent_diffusion(args.diff_ckpt)
    controlnet  = climb.init_controlnet()

    if args.cnet_ckpt is not None:
        print('Resuming training...')
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        print('Copying weights from diffusion model')
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    # freeze the unet weights
    for p in diffusion.parameters():
        p.requires_grad = False

    # Move everything to DEVICE
    autoencoder.to(DEVICE)
    diffusion.to(DEVICE)
    controlnet.to(DEVICE)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    scale_factor = 1.
    print(f"Scaling factor set to {scale_factor}")

    scheduler = DDPMScheduler(num_train_timesteps=1000, 
                              schedule='scaled_linear_beta', 
                              beta_start=0.0015, 
                              beta_end=0.0205)
    ##################
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }
    datasets        = { 'train': train_set, 'valid': valid_set }

    min_loss = np.inf
    for epoch in range(args.n_epochs):
        
        for mode in loaders.keys():
            print('mode:', mode)
            loader = loaders[mode]
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_loss = 0.
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                
                if mode == 'train':
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(mode == 'train'):

                    starting_z = batch['starting_latent'].to(DEVICE)  * scale_factor
                    followup_z = batch['followup_latent'].to(DEVICE)  * scale_factor
                    context    = batch['context'].to(DEVICE)
                    starting_a = batch['starting_age'].to(DEVICE)

                    n, l = starting_z.shape[:2]

                    with autocast(enabled=True):

                        concatenating_age      = starting_a.view(n, l, 1, 1, 1).expand(n, l, *starting_z.shape[-3:])
                        controlnet_condition   = torch.cat([ starting_z, concatenating_age ], dim=1)

                        noise = torch.randn_like(followup_z).to(DEVICE)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
                        images_noised = scheduler.add_noise(followup_z, noise=noise, timesteps=timesteps)

                        down_h, mid_h = controlnet(
                            x=images_noised.float(), 
                            timesteps=timesteps, 
                            context=context.float(),
                            controlnet_cond=controlnet_condition.float()
                        )

                        noise_pred = diffusion(
                            x=images_noised.float(), 
                            timesteps=timesteps, 
                            context=context.float(), 
                            down_block_additional_residuals=down_h,
                            mid_block_additional_residual=mid_h
                        )

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                #-------------------------------
                # Iteration end
                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1

            # Epoch loss
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)
            
            # Logging visualization
            images_to_tensorboard(
                writer=writer,
                epoch=epoch,
                mode=mode, 
                autoencoder=autoencoder, 
                diffusion=diffusion, 
                controlnet=controlnet,
                dataset=datasets[mode], 
                scale_factor=scale_factor
            )

        if min_loss >  epoch_loss and "valid" == mode:
            min_loss = epoch_loss 
            print("Best at epoch: {}".format(epoch), min_loss)      
            savepath = os.path.join(args.output_dir, "cnet_best_{}.pth".format(epoch))
            torch.save(controlnet.state_dict(), savepath)

        # torch.save(controlnet.state_dict(), os.path.join(args.output_dir, 'cnet_last.pth'))
