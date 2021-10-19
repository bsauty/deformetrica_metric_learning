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
        self.gamma = 0
        self.lr = 1e-4                                            # For epochs between MCMC steps
        self.epoch = 0                                            # For tensorboard to keep track of total number of epochs

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)     # 16 x 32 x 32 
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)    # 32 x 16 x 16 
        self.conv3 = nn.Conv2d(32,32, 3, stride=2, padding=1)     # 32 x 8 x 8 
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, Settings().dimension)
        self.fc11 = nn.Linear(2048, Settings().dimension)

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
        mu = torch.tanh(self.fc10(h3.flatten(start_dim=1)))
        logVar = self.fc11(h3.flatten(start_dim=1))
        return mu, logVar

    def decoder(self, encoded):
        #h5 = F.relu(self.fc2(encoded))
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
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

    def plot_images_longitudinal(self, encoded_images, writer=None):
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

    def plot_images_gradient(self, encoded_gradient, writer=None):
        ncolumns = encoded_gradient.shape[0] 
        fig, axes = plt.subplots(1, ncolumns-1, figsize=(6,2))
        decoded_p0 = self.decoder(encoded_gradient[0].unsqueeze(0).to(device))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(1,ncolumns):
            simulated_img = self.decoder(encoded_gradient[i].unsqueeze(0).to(device)) - decoded_p0
            axes[i-1].matshow(simulated_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'))
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        if writer is not None:
            writer.add_images('Gradient', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_gradient.png', bbox_inches='tight')
        plt.close('all')


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

class CAE_3D(nn.Module):
    """
    This is the convolutionnal autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    This is the architecture suggested in Martinez-Mucia et al.
    """

    def __init__(self):
        super(CAE, self).__init__()
        nn.Module.__init__(self)

        self.conv1 = nn.Conv3d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(64, 512, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        # self.bn4 = nn.BatchNorm3d(512)
        # self.maxpool = nn.MaxPool3d(2)

        self.fc = nn.Linear(512, 1200)
        self.up1 = nn.ConvTranspose3d(1, 16, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(16, 64, 3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.conv = nn.Conv3d(16, 1, 1)
        self.bn4 = nn.BatchNorm3d(16)
        self.bn5 = nn.BatchNorm3d(64)
        self.bn6 = nn.BatchNorm3d(32)
        self.dropout = nn.Dropout(0.25)

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        h4 = F.relu(self.conv4(h3))
        h5 = h4.mean(dim=(-3,-2,-1))  # Global average pooling layer after convolutions
        h5 = torch.tanh(h5).view(h5.size())
        return h5

    def decoder(self, encoded):
        h6 = F.relu(self.dropout(self.fc(encoded))).reshape([encoded.size()[0], 1, 10, 12, 10])
        h7 = F.relu(self.bn4(self.up1(h6)))
        h8 = F.relu(self.bn5(self.up2(h7)))
        h9 = F.relu(self.bn6(self.up3(h8)))
        h10 = F.relu(self.up4(h9))
        reconstructed = F.relu(self.conv(h10))
        return reconstructed

    def forward(self, image):
        encoded = self.encoder(image)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed

    def plot_images(self, data, n_images):
        im_list = []
        for i in range(n_images):
            test_image = random.choice(data)
            test_image = Variable(test_image.unsqueeze(0)).to(device)
            _, out = self.forward(test_image)

            im_list.append(Image.fromarray(255*test_image[0][0][30].cpu().detach().numpy()).convert('RGB'))
            im_list.append(Image.fromarray(255*out[0][0][30].cpu().detach().numpy()).convert('RGB'))

        im_list[0].save("Quality_control.pdf", "PDF", resolution=100.0, save_all=True, append_images=im_list[1:])

    def evaluate(self, data, criterion):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        nb_batches = 0
        encoded_data = torch.empty([0,512])
        with torch.no_grad():
            for data in dataloader:
                input_ = Variable(data).to(device)
                encoded, reconstructed = self.forward(input_)
                loss = criterion(reconstructed, input_)
                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, encoded.to('cpu')), 0)
        loss = tloss/nb_batches
        self.training = True
        return loss, encoded_data

    def train(self, data_loader, test, criterion, optimizer, num_epochs=20):

        self.to(device)

        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 10:
                break

            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            tloss = 0.0
            nb_batches = 0
            for data in data_loader:
                input_ = Variable(data).to(device)
                optimizer.zero_grad()
                encoded, reconstructed = self.forward(input_)
                loss = criterion(reconstructed, input_)
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1
            epoch_loss = tloss/nb_batches
            test_loss, _ = self.evaluate(test, criterion)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")

            self.plot_images(test, 10)
        print('Complete training')
        return

class CVAE_3D(nn.Module):
    """
    This is the convolutionnal autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    This is the architecture suggested in Martinez-Mucia et al.
    """

    def __init__(self):
        super(CVAE, self).__init__()
        nn.Module.__init__(self)

        # Encoder
        self.conv1 = nn.Conv3d(1, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv3d(32, 32, 5, stride=2, padding=2)
        self.conv3 = nn.Conv3d(32, 64, 5, stride=1, padding=2)
        self.conv4 = nn.Conv3d(64, 64, 5, stride=2, padding=2)
        self.conv5 = nn.Conv3d(64, 256, 3,stride=2, padding=1)
        self.conv6 = nn.Conv3d(256, 512, 3, stride=2, padding=1)
        self.bn10 = nn.BatchNorm3d(32)
        self.bn11 = nn.BatchNorm3d(32)
        self.bn20 = nn.BatchNorm3d(64)
        self.bn21 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)
        self.maxpool = nn.MaxPool3d(2)
        self.fc11 = nn.Linear(6144, 512)
        self.fc12 = nn.Linear(6144, 512)

        # Decoder
        self.fc2 = nn.Linear(512, 1200)
        #self.up1_3D =  nn.ConvTranspose3d(512, 256, 5, stride=2)
        self.up1 = nn.ConvTranspose3d(1, 256, 5, stride=2, padding=2, output_padding=1)
        self.up2 = nn.Conv3d(256, 256, 5, stride=1, padding=2)
        self.up3 = nn.ConvTranspose3d(256, 64, 5, stride=2, padding=2, output_padding=1)
        self.up4 = nn.Conv3d(64, 64, 5, stride=1, padding=2)
        self.up5 = nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv3d(32, 1, 3, stride=1, padding=1)
        self.bnDense = nn.BatchNorm1d(1200)
        self.bn50 = nn.BatchNorm3d(256)
        self.bn51 = nn.BatchNorm3d(256)
        self.bn60 = nn.BatchNorm3d(64)
        self.bn61 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(32)        

        self.dropout = nn.Dropout(0.25)
        
        self.beta = 0.2

    def encoder(self, image):
        h1 = F.relu(self.conv1(image))
        h2 = F.relu(self.bn11(self.conv2(h1)))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.bn21(self.conv4(h3)))
        h5 = F.relu(self.conv5(h4))
        h6 = F.relu(self.maxpool(self.conv6(h5))).flatten(start_dim=1)
        mu = self.fc11(h6)
        logVar = self.fc12(h6)
        return mu, logVar

    def decoder(self, encoded):
        h9 = F.relu(self.dropout(self.fc2(encoded))).reshape([encoded.size()[0], 1, 10, 12, 10])
        h10 = F.relu(self.up1(h9))
        h11 = F.relu(self.bn51(self.up2(h10)))
        h12 = F.relu(self.up3(h11))
        h13 = F.relu(self.bn61(self.up4(h12)))
        h14 = F.relu(self.bn7(self.up5(h13)))
        reconstructed = torch.sigmoid(self.conv(h14))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, image):
        mu, logVar = self.encoder(image)
        encoded = self.reparametrize(mu, logVar)
        reconstructed = self.decoder(encoded)
        # TODO: return mu or the encoded variable with reparametrization trick ?
        return mu, logVar, reconstructed

    def plot_images(self, data, n_images):
        im_list = []
        for i in range(n_images):
            test_image = random.choice(data)
            test_image = Variable(test_image.unsqueeze(0)).to(device)
            _, _, out = self.forward(test_image)

            im_list.append(Image.fromarray(255*test_image[0][0][30].cpu().detach().numpy()).convert('RGB'))
            im_list.append(Image.fromarray(255*out[0][0][30].cpu().detach().numpy()).convert('RGB'))

        im_list[0].save("Quality_control.pdf", "PDF", resolution=100.0, save_all=True, append_images=im_list[1:])

    def evaluate(self, data, criterion):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        nb_batches = 0
        encoded_data = torch.empty([0,512])
        with torch.no_grad():
            for data in dataloader:
                input_ = Variable(data).to(device)
                mu, logVar, reconstructed = self.forward(input_)
                loss = criterion(mu, logVar, reconstructed, input_)
                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, mu.to('cpu')), 0)
        loss = tloss/nb_batches
        self.training = True
        return loss, encoded_data

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        recon_error = nn.MSELoss()(reconstructed, input_) 
        return recon_error, kl_divergence

    def train(self, data_loader, test, criterion, optimizer, num_epochs=20):

        self.to(device)

        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 10:
                break

            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            tloss = 0.0
            nb_batches = 0
            for data in data_loader:
                input_ = Variable(data).to(device)
                optimizer.zero_grad()
                mu, logVar, reconstructed = self.forward(input_)
                loss = criterion(mu, logVar, reconstructed, input_)
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1
            epoch_loss = tloss/nb_batches
            test_loss, _ = self.evaluate(test, criterion)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")

            # Save images to check quality as training goes
            self.plot_images(test, 10)

        print('Complete training')
        return

class LAE_3D(nn.Module):
    """
    This is the longitudinal autoencoder that takes as input the latent variables from the CAE and tries to
    both align its latent representation according to the individual trajectories and reconstruct its input.
    For this to work the decoder from the CAE must be highly Lipschitzien but it that doesn't work we can
    change the loss to reconstruct the MRI directly.
    """

    def __init__(self):
        super(LAE, self).__init__()
        nn.Module.__init__(self)

        # encoder network
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # decoder network
        self.fc4 = nn.Linear(10, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 512)

        self.dropout = nn.Dropout(0.3)

    def encoder(self, x):
        h1 = self.dropout(F.relu(self.fc1(x)))
        h2 = self.dropout(F.relu(self.fc2(h1)))
        h3 = torch.tanh(self.fc3(h2))
        return h3

    def decoder(self, z):
        h4 = self.dropout(F.relu(self.fc4(z)))
        h5 = self.dropout(F.relu(self.fc5(h4)))
        h6 = F.relu(self.fc6(h5))
        return torch.tanh(h6)

    def forward(self, input):
        encoded = self.encoder(input)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed

    def evaluate(self, torch_data, criterion, longitudinal=None, individual_RER=None):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        dataloader = torch.utils.data.DataLoader(torch_data, batch_size=10, num_workers=10, shuffle=False)
        tloss = 0.0
        nb_batches = 0
        encoded_data = torch.empty([0,10])
        with torch.no_grad():
            for data in dataloader:
                input_ = Variable(data[0]).to(device)
                encoded, reconstructed = self.forward(input_)
                if longitudinal is not None:
                    loss = criterion(data, encoded, reconstructed, individual_RER)
                else:
                    loss = criterion(input_, reconstructed)
                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, encoded.to('cpu')), 0)
        self.training = True
        return tloss/nb_batches, encoded_data

    def train(self, data_loader, test, criterion, optimizer, num_epochs=20, longitudinal=None, individual_RER=None):
        """
        This training routine has to take as input an object from class Dataset in order to have access to the
        subject id and timepoint.
        """

        self.to(device)

        best_loss = 1e10
        early_stopping = 0

        for epoch in range(num_epochs):
            start_time = time()
            if early_stopping == 10:
                break

            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            tloss = 0.0
            nb_batches = 0

            for data in data_loader:
                input_ = Variable(data[0]).to(device)
                optimizer.zero_grad()
                encoded, reconstructed = self.forward(input_)
                if longitudinal is not None:
                    loss = criterion(data, encoded, reconstructed, individual_RER)
                else:
                    loss = criterion(input_, reconstructed)
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            epoch_loss = tloss / nb_batches
            if longitudinal is not None:
                test_loss, _ = self.evaluate(test, criterion, longitudinal=True, individual_RER=individual_RER)
            else:
                test_loss, _ = self.evaluate(test, criterion)

            if epoch_loss <= best_loss:
                early_stopping = 0
                best_loss = epoch_loss
            else:
                early_stopping += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")

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



