from typing import Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision
from torch.utils import data
import numpy as np

from time import time
import random
import logging
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.cm
import logging
from ....support.utilities.general_settings import Settings


# This is a dirty workaround for a stupid problem with pytorch and osx that mismanage openMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CVAE_2D(nn.Module):
    """
    This is the convolutionnal variationnal autoencoder for the 2D starmen dataset.
    """

    def __init__(self):
        super(CVAE_2D, self).__init__()
        nn.Module.__init__(self)
        self.beta = 5
        self.gamma = 50
        self.lr = 1e-4                                            # For epochs between MCMC steps
        self.epoch = 0                                            # For tensorboard to keep track of total number of epochs

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)     # 16 x 32 x 32 
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)    # 32 x 16 x 16 
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)    # 32 x 8 x 8 
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, Settings().dimension)
        self.fc11 = nn.Linear(2048, Settings().dimension)
        self.rotation1 = nn.Linear(Settings().dimension, Settings().dimension)
        self.rotation1.weight.data.copy_(torch.eye(Settings().dimension))

        self.rotation2 = nn.Linear(Settings().dimension, Settings().dimension)
        self.rotation2.weight.data.copy_(torch.eye(Settings().dimension))

        #self.fc2 = nn.Linear(8, 64)
        self.fc3 = nn.Linear(Settings().dimension,512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)    # 32 x 16 x 16 
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)   # 16 x 32 x 32 
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)    # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        #self.dropout = nn.Dropout(0.4)

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        mu = self.rotation1(torch.tanh(self.fc10(h3.flatten(start_dim=1))))
        logVar = self.rotation1(self.fc11(h3.flatten(start_dim=1)))
        return mu, logVar

    def decoder(self, encoded):
        #h5 = F.relu(self.fc2(encoded))
        h6 = F.relu(self.fc3(self.rotation2(encoded))).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed
    
    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2).to(device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std =1).to(device)
        if self.beta != 0:                   # beta VAE
            return mu + eps*std
        else:                           # regular AE
            return mu

    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed

    def plot_images_vae(self, data, n_images, writer=None):
        
        # Plot the reconstruction
        fig, axes = plt.subplots(2, n_images, figsize=(10,2))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(n_images):
            test_image = random.choice(data)
            test_image = Variable(test_image.unsqueeze(0)).to(device)
            mu, logVar, out = self.forward(test_image)
            axes[0][i].matshow(255*test_image[0][0].cpu().detach().numpy())
            axes[1][i].matshow(255*out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        
        if writer is not None:
            writer.add_images('reconstruction', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_reconstruction.png', bbox_inches='tight')
        plt.close()

        # Plot simulated data in all directions of the latent space
        fig, axes = plt.subplots(mu.shape[1], 7, figsize=(14,2*Settings().dimension))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(mu.shape[1]):
            for j in range(-3,4):
                simulated_latent = torch.zeros(mu.shape)
                simulated_latent[0][i] = j/4
                simulated_img = self.decoder(simulated_latent.unsqueeze(0).to(device))
                axes[i][(j+3)%7].matshow(255*simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        if writer is not None:
            writer.add_images('latent_directions', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_simulation_latent.png', bbox_inches='tight')
        plt.close()        
        self.training = True


    def plot_images_longitudinal(self, encoded_images, writer=None):
        self.training = False
        nrows, ncolumns = encoded_images.shape[0], encoded_images.shape[1]
        fig, axes = plt.subplots(nrows, ncolumns, figsize=(14,14))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(nrows):
            for j in range(ncolumns):
                simulated_img = self.decoder(encoded_images[i][j].unsqueeze(0).to(device))
                axes[i][j].matshow(simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        if writer is not None:
            writer.add_images('longitudinal_directions', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_simulation_longitudinal.png', bbox_inches='tight')
        plt.close('all')
        self.training = True

    def plot_images_gradient(self, encoded_gradient, writer=None):
        self.training = False
        
        ncolumns = encoded_gradient.shape[0] 
        fig, axes = plt.subplots(1, ncolumns-1, figsize=(6,2))
        decoded_p0 = self.decoder(encoded_gradient[0].unsqueeze(0).to(device))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(1,ncolumns):
            simulated_img = self.decoder(encoded_gradient[i].unsqueeze(0).to(device)) - decoded_p0
            axes[i-1].matshow(simulated_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm())
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        if writer is not None:
            writer.add_images('Gradient', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_gradient.png', bbox_inches='tight')
        plt.close('all')
        self.training = True



    def evaluate(self, data, longitudinal=None, individual_RER=None, writer=None, train_losses=None):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        criterion = self.loss
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
        nb_batches = 0
        encoded_data = torch.empty([0,Settings().dimension])

        with torch.no_grad():
            for data in dataloader:

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER)
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss
                else:
                    input_ = Variable(data).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, mu.to('cpu')), 0)

        if writer is not None:
            writer.add_scalars('Loss/recon', {'test' : trecon_loss/nb_batches, 'train' : train_losses[0]} , self.epoch)
            writer.add_scalars('Loss/kl', {'test' : tkl_loss/nb_batches, 'train' : train_losses[1]}, self.epoch)
            writer.add_scalars('Loss/alignment', {'test' : talignment_loss/nb_batches, 'train' : train_losses[2]}, self.epoch)

        loss = tloss/nb_batches
        self.training = True
        return loss, encoded_data

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        #recon_error = torch.nn.MSELoss(reduction='mean')(reconstructed, input_)
        recon_error = torch.sum((reconstructed - input_)**2) / input_.shape[0]
        return recon_error, kl_divergence

    def train(self, data_loader, test, optimizer, num_epochs=20, longitudinal=None, individual_RER=None, writer=None):

        self.to(device)
        criterion = self.loss
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch+1, num_epochs ))

            tloss = 0.0
            trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
            tmu, tlogvar = torch.zeros((1,Settings().dimension)), torch.zeros((1,Settings().dimension))
            nb_batches = 0

            for data in data_loader:
                optimizer.zero_grad()

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER) 
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss 
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss 
                    tmu = torch.cat((tmu, mu))
                    tlogvar = torch.cat((tlogvar, logVar))
                else:
                    input_ = Variable(data).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1
            epoch_loss = tloss/nb_batches

            if writer is not None:
                self.epoch += 1
                train_losses = (trecon_loss/nb_batches, tkl_loss/nb_batches, talignment_loss/nb_batches)
                test_loss, _ = self.evaluate(test, longitudinal=longitudinal, individual_RER=individual_RER, writer=writer, train_losses=train_losses)
                writer.add_histogram('Mu', tmu, self.epoch)
                writer.add_histogram('Logvar', tlogvar, self.epoch)
            else:
                test_loss, _ = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")

            # Save images to check quality as training goes
            if longitudinal is not None:
                self.plot_images_vae(test.data, 10, writer)
            else:
                self.plot_images_vae(test, 10)

        print('Complete training')
        return

class CVAE_3D(nn.Module):
    """
    This is the convolutionnal autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    This is the architecture suggested in Martinez-Mucia et al.
    """

    def __init__(self):
        super(CVAE_3D, self).__init__()
        nn.Module.__init__(self)
        self.beta = 5
        self.gamma = 10
        self.lr = 1e-4                                                      # For epochs between MCMC steps
        self.epoch = 0                                                      # For tensorboard to keep track of total number of epochs

        self.conv1 = nn.Conv3d(1, 32, 3, stride=2, padding=1)               # 32 x 40 x 48 x 40
        self.conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)              # 64 x 20 x 24 x 20
        self.conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)             # 128 x 10 x 12 x 10
        #self.conv4 = nn.Conv3d(64, 64, 3, stride=2, padding=1)             # 64 x 5 x 6 x 5
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        #self.bn4 = nn.BatchNorm3d(64)
        #self.fc1 = nn.Linear(9600, 512)
        self.fc20 = nn.Linear(153600, Settings().dimension)
        self.fc21 = nn.Linear(153600, Settings().dimension)

        self.fc3 = nn.Linear(Settings().dimension,38400)
        self.upconv1 = nn.ConvTranspose3d(256, 64, 3, stride=2, padding=1, output_padding=1)    # 64 x 10 x 12 x 10 
        self.upconv2 = nn.ConvTranspose3d(64, 64, 3, stride=2, padding=1, output_padding=1)   # 64 x 20 x 24 x 20 
        self.upconv3 = nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1)   # 32 x 40 x 48 x 40 
        self.upconv4 = nn.ConvTranspose3d(32, 1, 3, stride=2, padding=1, output_padding=1)    # 1 x 80 x 96 x 80
        self.bn5 = nn.BatchNorm3d(64)
        self.bn6 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(32)

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        #h4 = F.relu(self.bn4(self.conv4(h3)))
        #h5 = F.relu(self.fc1(h4.flatten(start_dim=1)))
        h5 = h3.flatten(start_dim=1)
        mu = torch.tanh(self.fc20(h5))
        logVar = self.fc21(h5)
        return mu, logVar

    def decoder(self, encoded):
        h5 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 256, 5, 6, 5])
        h6 = F.relu(self.bn5(self.upconv1(h5)))
        h7 = F.relu(self.bn6(self.upconv2(h6)))
        h8 = F.relu(self.bn7(self.upconv3(h7)))
        reconstructed = F.relu(self.upconv4(h8))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2).to(device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std =1).to(device)
        if self.beta != 0:                   # beta VAE
            return mu + eps*std
        else:                           # regular AE
            return mu

    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed
    
    def plot_images_vae(self, data, n_images, writer=None):
        # Plot the reconstruction
        fig, axes = plt.subplots(2, n_images, figsize=(8,2))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(n_images):
            test_image = random.choice(data)
            test_image = Variable(test_image.unsqueeze(0)).to(device)
            mu, logVar, out = self.forward(test_image)
            axes[0][i].matshow(255*test_image[0][0][30].cpu().detach().numpy())
            axes[1][i].matshow(255*out[0][0][30].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        
        if writer is not None:
            writer.add_images('reconstruction', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_reconstruction.png', bbox_inches='tight')
        plt.close()
        
        """
        # Plot simulated data in all directions of the latent space
        fig, axes = plt.subplots(mu.shape[1], 7, figsize=(12,2*Settings().dimension))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(mu.shape[1]):
            for j in range(-3,4):
                simulated_latent = torch.zeros(mu.shape)
                simulated_latent[0][i] = j/4
                simulated_img = self.decoder(simulated_latent.unsqueeze(0).to(device))
                axes[i][(j+3)%7].matshow(255*simulated_img[0][0][30].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        if writer is not None:
            writer.add_images('latent_directions', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_simulation_latent.png', bbox_inches='tight')
        plt.close()"""

    def plot_images_longitudinal(self, encoded_images, writer=None):
        nrows, ncolumns = encoded_images.shape[0], encoded_images.shape[1]
        fig, axes = plt.subplots(nrows, ncolumns, figsize=(14,14))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(nrows):
            for j in range(ncolumns):
                simulated_img = self.decoder(encoded_images[i][j].unsqueeze(0).to(device))
                axes[i][j].matshow(simulated_img[0][0][30].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        if writer is not None:
            writer.add_images('longitudinal_directions', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_simulation_longitudinal.png', bbox_inches='tight')
        plt.close('all')

    def plot_images_gradient(self, encoded_gradient, writer=None):
        ncolumns = encoded_gradient.shape[0] 
        fig, axes = plt.subplots(1, ncolumns-1, figsize=(6,2))
        decoded_p0 = self.decoder(encoded_gradient[0].unsqueeze(0).to(device))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(1,ncolumns):
            simulated_img = self.decoder(encoded_gradient[i].unsqueeze(0).to(device)) - decoded_p0
            axes[i-1].matshow(simulated_img[0][0][30].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm())
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        if writer is not None:
            writer.add_images('Gradient', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_gradient.png', bbox_inches='tight')
        plt.close('all')

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        #recon_error = torch.nn.MSELoss(reduction='mean')(reconstructed, input_)
        recon_error = torch.sum((reconstructed - input_)**2) / input_.shape[0]
        return recon_error, kl_divergence

    def evaluate(self, data, longitudinal=None, individual_RER=None, writer=None, train_losses=None):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        criterion = self.loss
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
        nb_batches = 0
        encoded_data = torch.empty([0,Settings().dimension])

        with torch.no_grad():
            for data in dataloader:

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER)
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss
                else:
                    input_ = Variable(data).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, mu.to('cpu')), 0)

        if writer is not None:
            writer.add_scalars('Loss/recon', {'test' : trecon_loss/nb_batches, 'train' : train_losses[0]} , self.epoch)
            writer.add_scalars('Loss/kl', {'test' : tkl_loss/nb_batches, 'train' : train_losses[1]}, self.epoch)
            writer.add_scalars('Loss/alignment', {'test' : talignment_loss/nb_batches, 'train' : train_losses[2]}, self.epoch)

        loss = tloss/nb_batches
        self.training = True
        return loss, encoded_data

    def train(self, data_loader, test, optimizer, num_epochs=20, longitudinal=None, individual_RER=None, writer=None):

        self.to(device)
        criterion = self.loss
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch+1, num_epochs ))

            tloss = 0.0
            trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
            tmu, tlogvar = torch.zeros((1,Settings().dimension)), torch.zeros((1,Settings().dimension))
            nb_batches = 0

            for data in data_loader:
                optimizer.zero_grad()

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER) 
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss 
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss 
                    tmu = torch.cat((tmu, mu))
                    tlogvar = torch.cat((tlogvar, logVar))
                else:
                    input_ = Variable(data).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1
            epoch_loss = tloss/nb_batches

            if writer is not None:
                self.epoch += 1
                train_losses = (trecon_loss/nb_batches, tkl_loss/nb_batches, talignment_loss/nb_batches)
                test_loss, _ = self.evaluate(test, longitudinal=longitudinal, individual_RER=individual_RER, writer=writer, train_losses=train_losses)
                writer.add_histogram('Mu', tmu, self.epoch)
                writer.add_histogram('Logvar', tlogvar, self.epoch)
            else:
                test_loss, _ = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")

            # Save images to check quality as training goes
            if longitudinal is not None:
                self.plot_images_vae(test.data, 10, writer)
            else:
                self.plot_images_vae(test, 10)

        print('Complete training')
        return
    
class ACVAE_3D(nn.Module):
    """
    This is the convolutionnal autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    This is the architecture suggested in Martinez-Mucia et al.
    """

    def __init__(self):
        super(ACVAE_3D, self).__init__()
        nn.Module.__init__(self)
        self.beta = 5
        self.gamma = 10
        self.lr = 1e-4                                                      # For epochs between MCMC steps
        self.epoch = 0                                                      # For tensorboard to keep track of total number of epochs

        self.conv10 = nn.Conv3d(1, 32, 3, stride=2, padding=1)               # 32 x 40 x 48 x 40
        self.conv11 = nn.Conv3d(32, 32, 3, stride=1, padding=1)              # 32 x 40 x 48 x 40
        self.conv20 = nn.Conv3d(32, 64, 3, stride=2, padding=1)              # 64 x 20 x 24 x 20
        self.conv21 = nn.Conv3d(64, 64, 3, stride=1, padding=1)              # 64 x 20 x 24 x 20
        self.conv30 = nn.Conv3d(64, 128, 3, stride=2, padding=1)             # 128 x 10 x 12 x 10
        self.conv31 = nn.Conv3d(128, 128, 3, stride=1, padding=1)            # 128 x 10 x 12 x 10
        self.conv40 = nn.Conv3d(128, 256, 3, stride=1, padding=1)            # 256 x 10 x 12 x 10
        self.conv41 = nn.Conv3d(256, 64, 3, stride=1, padding=1)             # 64 x 10 x 12 x 10
        self.bn10 = nn.BatchNorm3d(32)
        self.bn11 = nn.BatchNorm3d(32)
        self.bn20 = nn.BatchNorm3d(64)
        self.bn21 = nn.BatchNorm3d(64)
        self.bn30 = nn.BatchNorm3d(128)
        self.bn31 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        #self.fc1 = nn.Linear(9600, 512)
        self.fc20 = nn.Linear(76800, Settings().dimension)
        self.fc21 = nn.Linear(76800, Settings().dimension)

        self.fc3 = nn.Linear(Settings().dimension,38400)
        self.upconv1 = nn.ConvTranspose3d(256, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 10 x 12 x 10 
        self.upconv2 = nn.ConvTranspose3d(64, 64, 3, stride=2, padding=1, output_padding=1)   # 64 x 20 x 24 x 20 
        self.upconv3 = nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1)   # 32 x 40 x 48 x 40 
        self.upconv4 = nn.ConvTranspose3d(32, 1, 3, stride=2, padding=1, output_padding=1)    # 1 x 80 x 96 x 80
        self.bn5 = nn.BatchNorm3d(64)
        self.bn6 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(32)

    def encoder(self, image):
        h10 = F.relu(self.bn10(self.conv10(image)))
        h11 = F.relu(self.bn11(self.conv11(h10)))
        h20 = F.relu(self.bn20(self.conv20(h11)))
        h21 = F.relu(self.bn21(self.conv21(h20)))
        h30 = F.relu(self.bn30(self.conv30(h21)))
        h31 = F.relu(self.bn31(self.conv31(h30)))
        h40 = F.relu(self.bn4(self.conv40(h31)))
        h41 = F.relu(self.conv41(h40))
        h5 = h41.flatten(start_dim=1)
        mu = torch.tanh(self.fc20(h5))
        logVar = self.fc21(h5)
        return mu, logVar

    def decoder(self, encoded):
        h5 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 256, 5, 6, 5])
        h6 = F.relu(self.bn5(self.upconv1(h5)))
        h7 = F.relu(self.bn6(self.upconv2(h6)))
        h8 = F.relu(self.bn7(self.upconv3(h7)))
        reconstructed = F.relu(self.upconv4(h8))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2).to(device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std =1).to(device)
        if self.beta != 0:                   # beta VAE
            return mu + eps*std
        else:                           # regular AE
            return mu

    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed
    
    def plot_images_vae(self, data, n_images, writer=None):
        # Plot the reconstruction
        fig, axes = plt.subplots(2, n_images, figsize=(8,2))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(n_images):
            test_image = random.choice(data)
            test_image = Variable(test_image.unsqueeze(0)).to(device)
            mu, logVar, out = self.forward(test_image)
            axes[0][i].matshow(255*test_image[0][0][30].cpu().detach().numpy())
            axes[1][i].matshow(255*out[0][0][30].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        
        if writer is not None:
            writer.add_images('reconstruction', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_reconstruction.png', bbox_inches='tight')
        plt.close()
        
        """
        # Plot simulated data in all directions of the latent space
        fig, axes = plt.subplots(mu.shape[1], 7, figsize=(12,2*Settings().dimension))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(mu.shape[1]):
            for j in range(-3,4):
                simulated_latent = torch.zeros(mu.shape)
                simulated_latent[0][i] = j/4
                simulated_img = self.decoder(simulated_latent.unsqueeze(0).to(device))
                axes[i][(j+3)%7].matshow(255*simulated_img[0][0][30].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        if writer is not None:
            writer.add_images('latent_directions', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_simulation_latent.png', bbox_inches='tight')
        plt.close()"""

    def plot_images_longitudinal(self, encoded_images, writer=None):
        nrows, ncolumns = encoded_images.shape[0], encoded_images.shape[1]
        fig, axes = plt.subplots(nrows, ncolumns, figsize=(14,14))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(nrows):
            for j in range(ncolumns):
                simulated_img = self.decoder(encoded_images[i][j].unsqueeze(0).to(device))
                axes[i][j].matshow(simulated_img[0][0][30].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        if writer is not None:
            writer.add_images('longitudinal_directions', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_simulation_longitudinal.png', bbox_inches='tight')
        plt.close('all')

    def plot_images_gradient(self, encoded_gradient, writer=None):
        ncolumns = encoded_gradient.shape[0] 
        fig, axes = plt.subplots(1, ncolumns-1, figsize=(6,2))
        decoded_p0 = self.decoder(encoded_gradient[0].unsqueeze(0).to(device))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(1,ncolumns):
            simulated_img = self.decoder(encoded_gradient[i].unsqueeze(0).to(device)) - decoded_p0
            axes[i-1].matshow(simulated_img[0][0][30].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm())
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        if writer is not None:
            writer.add_images('Gradient', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_gradient.png', bbox_inches='tight')
        plt.close('all')

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        #recon_error = torch.nn.MSELoss(reduction='mean')(reconstructed, input_)
        recon_error = torch.sum((reconstructed - input_)**2) / input_.shape[0]
        return recon_error, kl_divergence

    def evaluate(self, data, longitudinal=None, individual_RER=None, writer=None, train_losses=None):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        criterion = self.loss
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
        nb_batches = 0
        encoded_data = torch.empty([0,Settings().dimension])

        with torch.no_grad():
            for data in dataloader:

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER)
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss
                else:
                    input_ = Variable(data).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, mu.to('cpu')), 0)

        if writer is not None:
            writer.add_scalars('Loss/recon', {'test' : trecon_loss/nb_batches, 'train' : train_losses[0]} , self.epoch)
            writer.add_scalars('Loss/kl', {'test' : tkl_loss/nb_batches, 'train' : train_losses[1]}, self.epoch)
            writer.add_scalars('Loss/alignment', {'test' : talignment_loss/nb_batches, 'train' : train_losses[2]}, self.epoch)

        loss = tloss/nb_batches
        self.training = True
        return loss, encoded_data

    def train(self, data_loader, test, optimizer, num_epochs=20, longitudinal=None, individual_RER=None, writer=None):

        self.to(device)
        criterion = self.loss
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch+1, num_epochs ))

            tloss = 0.0
            trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
            tmu, tlogvar = torch.zeros((1,Settings().dimension)), torch.zeros((1,Settings().dimension))
            nb_batches = 0

            for data in data_loader:
                optimizer.zero_grad()

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER) 
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss 
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss 
                    tmu = torch.cat((tmu, mu))
                    tlogvar = torch.cat((tlogvar, logVar))
                else:
                    input_ = Variable(data).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1
            epoch_loss = tloss/nb_batches

            if writer is not None:
                self.epoch += 1
                train_losses = (trecon_loss/nb_batches, tkl_loss/nb_batches, talignment_loss/nb_batches)
                test_loss, _ = self.evaluate(test, longitudinal=longitudinal, individual_RER=individual_RER, writer=writer, train_losses=train_losses)
                writer.add_histogram('Mu', tmu, self.epoch)
                writer.add_histogram('Logvar', tlogvar, self.epoch)
            else:
                test_loss, _ = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")

            # Save images to check quality as training goes
            if longitudinal is not None:
                self.plot_images_vae(test.data, 10, writer)
            else:
                self.plot_images_vae(test, 10)

        print('Complete training')
        return

class Dataset(data.Dataset):
    def __init__(self, images, labels, timepoints):
        self.data = images
        self.labels = labels
        self.timepoints = timepoints

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.data[index]
        y = self.labels[index]
        z = self.timepoints[index]
        return X, y, z

def fig2rgb_array(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return(data)

def main():
    """
    For debugging purposes only, once the architectures and training routines are efficient,
    this file will not be called as a script anymore.
    """
    logger.info(f"Device is {device}")

    epochs = 250
    batch_size = 4
    lr = 1e-5

    # Load data
    train_data = torch.load('../../../LAE_experiments/Starmen_data/Starmen_10')
    print(f"Loaded {len(train_data['data'])} encoded scans")
    train_data['data'].requires_grad = False
    torch_data = Dataset(train_data['data'].unsqueeze(1).float(), train_data['target'])
    train, test = torch.utils.data.random_split(torch_data.data, [len(torch_data)-2, 2])

    #autoencoder = CVAE()
    #criterion = autoencoder.loss
    
    autoencoder = CAE_2D()
    criterion = nn.MSELoss()
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                              shuffle=True, num_workers=1, drop_last=True)
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

    size = len(train)

    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    autoencoder.train(train_loader, test=test, criterion=criterion,
                      optimizer=optimizer, num_epochs=epochs)
    torch.save(autoencoder.state_dict(), path_LAE)

    return autoencoder



#if __name__ == '__main__':
 #   main()



