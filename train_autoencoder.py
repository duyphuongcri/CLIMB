import os
import argparse
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from monai.utils import set_determinism
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter

from models import utils
from models.wasserstein_autoencoder import init_wasserstein_autoencoder, init_patch_discriminator
from models.gradacc import GradientAccumulation
import dataloader
import numpy as np 

set_determinism(0)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


def sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1)) # B x num_projections
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1))) #B x num_projections
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections, dim=1)[0] -
                            torch.sort(distribution_projections, dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_root', type=str, default="path to preprocessed images")
    parser.add_argument('--output_dir', type=str, default="path to save model")
    parser.add_argument('--swae_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=100,     type=int)
    parser.add_argument('--batch_size',     default=3,    type=int)
    parser.add_argument('--max_batch_size',     default=3,    type=int) 
    parser.add_argument('--num_projections',     default=100,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    args = parser.parse_args()

    data_plit = pd.read_csv("./dataset/data_ptid_train_val_test_split_longitudinal.csv")
    frame = pd.read_csv("./dataset/data_cleaned_for_longitudinal.csv")
    frame = frame[frame["MRI"] == 1]

    list_ptid_train = list(data_plit[data_plit["train"] == 1.]["PTID"])
    list_ptid_valid = list(data_plit[data_plit["val"] == 1.]["PTID"])

    frame_train = frame[frame["PTID"].isin(list_ptid_train)]
    frame_valid = frame[frame["PTID"].isin(list_ptid_valid)]

    train_set = dataloader.ImageDataset(data_frame=frame_train, path_root=args.path_root, mode="train")
    valid_set = dataloader.ImageDataset(data_frame=frame_valid, path_root=args.path_root, mode="valid")
    # Dataloaders:
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    ## Load model
    autoencoder = init_wasserstein_autoencoder(args.swae_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    adv_weight          = 0.01
    perceptual_weight   = 0.01
    lambda_weight       = 0.01 
    latent_dim = 3 * 16 * 20 * 16
    prior_samples = torch.randn(1, latent_dim).to(DEVICE)

    l1_loss_fn = L1Loss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, 
                                      network_type="squeeze", 
                                      is_fake_3d=True, 
                                      fake_3d_ratio=0.2).to(DEVICE)
    
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_g, 
                                     grad_scaler=GradScaler())

    
    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                    expect_batch_size=args.batch_size,
                                    loader_len=len(train_loader),
                                    optimizer=optimizer_d, 
                                    grad_scaler=GradScaler())
    ###
    os.makedirs(args.output_dir, exist_ok=True)
    avgloss = utils.AverageLoss()
    writer = SummaryWriter(log_dir=args.output_dir)
    iteration = 0


    min_loss = np.inf
    
    for epoch in range(args.n_epochs):
        autoencoder.train()
        discriminator.train()

        ##### Train ##################
        for step, batch in tqdm(enumerate(train_loader)):
            iteration += 1

            ### Train autoencoder
            with autocast(enabled=True):
                images = batch["img"].to(DEVICE)
                B = len(images)
                
                z = autoencoder.encode(images)
                reconstruction = autoencoder.decode(z)

                # we use [-1] here because the discriminator also returns 
                # intermediate outputs and we want only the final one.
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # Computing the loss for the generator. In the Adverarial loss, 
                # if the discriminator works well then the logits are close to 0.
                # Since we use `target_is_real=True`, then the target tensor used
                # for the MSE is a tensor of 1, and minizing this loss will make 
                # the generator better at fooling the discriminator (the discriminator
                # weights are not optimized here).

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False) # for optimizer generator
                # Sliced Wasserstein loss
                
                swd_loss = lambda_weight * sliced_wasserstein_distance(z.view(B, -1), prior_samples.repeat(B, 1), args.num_projections, device=DEVICE)

                ####
                loss_g = rec_loss + per_loss + gen_loss + swd_loss
                ##

            gradacc_g.step(loss_g, step)
            
            ## Train discriminator
            with autocast(enabled=True):

                # Here we compute the loss for the discriminator. Keep in mind that
                # the loss used is an MSE between the output logits and the expected logits.
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # Logging.
            avgloss.put('Generator_train/reconstruction_loss',     rec_loss.item())
            avgloss.put('Generator_train/perceptual_loss',         per_loss.item())
            avgloss.put('Generator_train/adverarial_loss',         gen_loss.item())
            avgloss.put('Generator_train/swae_loss',               swd_loss.item())
            avgloss.put('Generator_train/total_loss',              loss_g.item())
            avgloss.put('Discriminator_train/adverarial_loss',     loss_d.item())

            if iteration % 200 == 0:

                #### Valid #####################
                autoencoder.eval()
                discriminator.eval()
                loss_g_total = 0

                with torch.no_grad():
                    for batch in tqdm(valid_loader):
                        with autocast(enabled=True):
                            images = batch["img"].to(DEVICE)
                            B = len(images)

                            z = autoencoder.encode(images)
                            reconstruction = autoencoder.decode(z)

                            # we use [-1] here because the discriminator also returns 
                            # intermediate outputs and we want only the final one.
                            logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                            # Computing the loss for the generator. In the Adverarial loss, 
                            # if the discriminator works well then the logits are close to 0.
                            # Since we use `target_is_real=True`, then the target tensor used
                            # for the MSE is a tensor of 1, and minizing this loss will make 
                            # the generator better at fooling the discriminator (the discriminator
                            # weights are not optimized here).

                            rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                            per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                            gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False) # for optimizer generator
                            ####    
                            swd_loss = lambda_weight * sliced_wasserstein_distance(z.view(B, -1), prior_samples.repeat(B, 1), args.num_projections, device=DEVICE)

                            ##################
                            # loss_g = rec_loss + per_loss + gen_loss + swd_loss
                            loss_g_total += rec_loss.item() + per_loss.item() #+ swd_loss.item()

                        # Logging.
                        avgloss.put('Generator_valid/reconstruction_loss',     rec_loss.item())
                        avgloss.put('Generator_valid/perceptual_loss',         per_loss.item())
                        avgloss.put('Generator_valid/adverarial_loss',         gen_loss.item())
                        avgloss.put('Generator_valid/swae_loss',               swd_loss.item())
                        avgloss.put('Generator_valid/total_loss',              rec_loss.item() + per_loss.item())

                avgloss.to_tensorboard(writer, iteration)
                utils.tb_display_reconstruction(writer, iteration, images[0].detach().cpu(), reconstruction[0].detach().cpu())


                # Save the model after each epoch.
                if min_loss > loss_g_total:
                    latent = z.detach().cpu().numpy()
                    print("Save model!!: ", iteration, loss_g_total / len(valid_loader), latent.min(), latent.max(), latent.mean(), latent.std())
                    min_loss = loss_g_total 

                    torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator_best.pth'))
                    torch.save(autoencoder.state_dict(),   os.path.join(args.output_dir, 'swae_best_{}.pth'.format(iteration)))

                # torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator_last.pth'))
                # torch.save(autoencoder.state_dict(),   os.path.join(args.output_dir, 'swae_last.pth'))