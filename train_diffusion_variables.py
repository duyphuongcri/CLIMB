import os
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from tqdm import tqdm

from models import const, utils, wasserstein_autoencoder, climb
from models.sampling import sample_using_diffusion

import dataloader
import numpy as np

set_determinism(0)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    real_context,
    scale_factor
):
    """
    Visualize the generation on tensorboard
    """
    image = sample_using_diffusion(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            context=real_context,
            device=DEVICE, 
            scale_factor=scale_factor
        )
    utils.tb_display_generation(
            writer=writer, 
            step=epoch, 
            tag='real_context_{}'.format(mode),
            image=image
        )

    for tag_i, size in enumerate([ 'small', 'medium', 'large' ]):

        context = torch.tensor([[
            (torch.randint(60, 99, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age 
            0,                                                                # apoe4
            (torch.randint(1, 3,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
            (torch.randint(1, 4,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
            0.567, # (mean) cerebral cortex 
            0.539, # (mean) hippocampus
            0.578, # (mean) amygdala
            0.558, # (mean) cerebral white matter
            0.30 * (tag_i+1), # variable size lateral ventricles
        ]])

        image = sample_using_diffusion(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            context=context,
            device=DEVICE, 
            scale_factor=scale_factor
        )

        utils.tb_display_generation(
            writer=writer, 
            step=epoch, 
            tag=f'{mode}/{size}_ventricles',
            image=image
        )
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_root', type=str, default="path to the extracted latent space")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--aekl_ckpt', type=str, default="path to pre-trained autoencoder")
    parser.add_argument('--diff_ckpt',   default=None, type=str)
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=10000,     type=int)
    parser.add_argument('--batch_size',  default=32,    type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    args = parser.parse_args()
    
    transforms_fn = transforms.Compose([
        transforms.EnsureChannelFirstD(keys=['latent'], channel_dim=0), 
    ])
    ## Load data

    data_plit = pd.read_csv("../dataset/data_ptid_train_val_test_split_longitudinal.csv")
    frame = pd.read_csv("../dataset/data_cleaned_for_longitudinal_new.csv")
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

    train_set = dataloader.LatentDataset(data_frame=frame_train, path_root=args.path_root, transform=transforms_fn, mode="train")
    valid_set = dataloader.LatentDataset(data_frame=frame_valid, path_root=args.path_root, transform=transforms_fn, mode="valid")
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
                              shuffle=False, 
                              persistent_workers=True, 
                              pin_memory=True)
    

    autoencoder = wasserstein_autoencoder.init_wasserstein_autoencoder(args.aekl_ckpt).to(DEVICE)
    diffusion = climb.init_latent_diffusion(args.diff_ckpt).to(DEVICE)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule='scaled_linear_beta', 
        beta_start=0.0015, 
        beta_end=0.0205
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    scaler = GradScaler()

    scale_factor = 1. #1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")


    writer = SummaryWriter(args.output_dir)
    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }

    min_loss = np.inf
    for epoch in range(args.n_epochs):
        
        for mode in loaders.keys():
            
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in progress_bar:
                            
                with autocast(enabled=True):    
                        
                    if mode == 'train': 
                        optimizer.zero_grad(set_to_none=True)
                    latents = batch['latent'].to(DEVICE)
                    latents = latents * scale_factor
                    context = batch['context'].to(DEVICE)
                    n = latents.shape[0]
                                                            
                    with torch.set_grad_enabled(mode == 'train'):
                        
                        noise = torch.randn_like(latents).to(DEVICE)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()

                        noise_pred = inferer(
                            inputs=latents, 
                            diffusion_model=diffusion, 
                            noise=noise, 
                            timesteps=timesteps,
                            condition=context,
                            mode='crossattn'
                        )

                        loss = F.mse_loss( noise.float(), noise_pred.float() )

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1
        
            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            # visualize results
            images_to_tensorboard(
                writer=writer, 
                epoch=epoch, 
                mode=mode, 
                autoencoder=autoencoder, 
                diffusion=diffusion, 
                real_context=context[0],
                scale_factor=scale_factor
            )

        # save the model   
        if min_loss >  epoch_loss :
            min_loss = epoch_loss 
            print("Best at epoch: {}".format(epoch), min_loss)      
            savepath = os.path.join(args.output_dir, 'unet_best_{}.pth'.format(epoch))
            torch.save(diffusion.state_dict(), savepath)

        # torch.save(diffusion.state_dict(), os.path.join(args.output_dir, 'unet_last.pth'))