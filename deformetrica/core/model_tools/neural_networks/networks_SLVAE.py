import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import sys

sys.path.append('/home/benoit.sautydechalon/deformetrica')
import deformetrica as dfca

from time import time
import random
import logging
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.cm
from deformetrica.support.utilities.general_settings import Settings


# This is a dirty workaround for a  problem with pytorch and osx that mismanage openMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
class CVAE_3D(nn.Module):
    """
    This is the convolutionnal autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    """

    def __init__(self):
        super(CVAE_3D, self).__init__()
        nn.Module.__init__(self)
        self.beta = 1
        self.gamma = 500
        self.kappa = 100
        self.lr = 1e-4                                                      # For epochs between MCMC steps
        self.epoch = 0           
        self.name = 'CVAE_3D'   
        
        # Encoder
        self.conv1 = nn.Conv3d(1, 64, 3, stride=2, padding=1)              # 32 x 40 x 48 x 40
        self.conv2 = nn.Conv3d(64, 128, 3, stride=2, padding=1)            # 64 x 20 x 24 x 20
        self.conv3 = nn.Conv3d(128, 128, 3, stride=2, padding=1)           # 256 x 10 x 12 x 10
        #self.conv4 = nn.Conv3d(256, 256, 3, stride=2, padding=1)           # 256 x 10 x 12 x 10
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(128)
        #self.bn4 = nn.BatchNorm3d(256)
        self.fc10 = nn.Linear(153600, Settings().dimension)
        self.fc11 = nn.Linear(153600, Settings().dimension)

        self.fs1 = nn.Linear(Settings().dimension,2)
        #self.drop = nn.Dropout(1)
        #self.fsbn1 = nn.BatchNorm1d(10)
        #self.fs2 = nn.Linear(10,10)
        #self.fsbn2 = nn.BatchNorm1d(10)
        #self.fs3 = nn.Linear(10,2)
        
        # Decoder
        self.fc2 = nn.Linear(Settings().dimension, 76800)
        self.upconv1 = nn.ConvTranspose3d(512, 256, 3, stride=2, padding=1, output_padding=1)    # 64 x 10 x 12 x 10 
        self.upconv2 = nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1)    # 64 x 20 x 24 x 20 
        self.upconv3 = nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1)     # 32 x 40 x 48 x 40 
        self.upconv4 = nn.ConvTranspose3d(64, 1, 3, stride=2, padding=1, output_padding=1)       # 1 x 80 x 96 x 80
        self.bn5 = nn.BatchNorm3d(256)
        self.bn6 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(64)
        
    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        #h4 = F.relu(self.bn4(self.conv4(h3)))
        #h5 = F.relu(self.fc1(h4.flatten(start_dim=1)))
        h5 = h3.flatten(start_dim=1)
        mu = torch.tanh(self.fc10(h5))
        logVar = self.fc11(h5)
        #hfs1 = F.relu(self.fsbn1(self.fs1(mu)))
        #hfs2 = F.relu(self.fsbn2(self.fs2(hfs1)))
        fs = torch.tanh(self.fs1(F.dropout(mu,.2)))
        return mu, logVar, fs

    def decoder(self, encoded):
        h5 = F.relu(self.fc2(encoded).reshape([encoded.size()[0], 512, 5, 6, 5]))
        h6 = F.relu(self.bn5(self.upconv1(h5)))
        h7 = F.relu(self.bn6(self.upconv2(h6)))
        h8 = F.relu(self.bn7(self.upconv3(h7)))
        reconstructed = F.relu(torch.tanh(self.upconv4(h8)))
        return reconstructed
    
    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2).to(device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std =1).to(device)
        if self.beta != 0:                   # beta VAE
            return mu + eps*std
        else:                                # regular AE
            return mu

    def forward(self, image):
        mu, logVar, fs = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed, fs
    
    def plot_images_vae(self, data, n_images, writer=None, name=None):
        # Plot the reconstruction
        fig, axes = plt.subplots(6, n_images, figsize=(8,4.8), gridspec_kw={'height_ratios':[1,1,.8,.8,.7,.7]})
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(n_images):
            test_image = Variable(data[i].unsqueeze(0)).to(device)
            mu, logVar, out, _ = self.forward(test_image)
            axes[0][i].matshow(255*test_image[0][0][30].cpu().detach().numpy(), aspect="equal", cmap='Greys_r')
            axes[1][i].matshow(255*out[0][0][30].cpu().detach().numpy(), aspect="equal", cmap='Greys_r')
            axes[2][i].matshow(255*test_image[0][0][:,30].cpu().detach().numpy(), aspect="equal", cmap='Greys_r')
            axes[3][i].matshow(255*out[0][0][:,30].cpu().detach().numpy(), aspect="equal", cmap='Greys_r')
            axes[4][i].matshow(255*test_image[0][0][:,:,40].cpu().detach().numpy(), aspect="equal", cmap='Greys_r')
            axes[5][i].matshow(255*out[0][0][:,:,40].cpu().detach().numpy(), aspect="equal", cmap='Greys_r')

        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        
        if writer is not None:
            writer.add_images('reconstruction', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        if name is None:
            name = 'qc_reconstruction.png'
        plt.savefig(name, bbox_inches='tight')
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
        """
        nrows, ncolumns = encoded_images.shape[0], encoded_images.shape[1]
        fig, axes = plt.subplots(nrows, ncolumns, figsize=(12,14))
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
        plt.close('all')"""

        nrows, ncolumns = 3, encoded_images.shape[1]
        fig, axes = plt.subplots(3,7, figsize=(7,2.73), gridspec_kw={'height_ratios':[.8,.96,.8]})
        plt.subplots_adjust(wspace=0.03, hspace=0.02)
        for j in range(ncolumns):
            simulated_img = self.decoder(encoded_images[0][j].unsqueeze(0).to(device))
            axes[0][j].matshow(np.rot90(simulated_img[0][0][30].cpu().detach().numpy()), cmap='RdYlBu_r')
            axes[1][j].matshow(np.rot90(simulated_img[0][0][:,42].cpu().detach().numpy()), cmap='RdYlBu_r')
            axes[2][j].matshow(simulated_img[0][0][:,:,40].cpu().detach().numpy(), cmap='RdYlBu_r')
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('qc_reference_geodesic.png', bbox_inches='tight')
        plt.close('all')

    def plot_images_gradient(self, encoded_gradient, writer=None):
        ncolumns = encoded_gradient.shape[0] 
        fig, axes = plt.subplots(3, ncolumns, figsize=(2*ncolumns,6), gridspec_kw={'height_ratios':[.8,.96,.8]})
        decoded_p0 = self.decoder(torch.zeros(encoded_gradient[0].shape).unsqueeze(0).to(device))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(0,ncolumns):
            simulated_img = self.decoder(encoded_gradient[i].unsqueeze(0).to(device)) - decoded_p0
            axes[0][i].matshow(np.rot90(simulated_img[0][0][28].cpu().detach().numpy()), cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm())
            axes[1][i].matshow(simulated_img[0][0][:,30].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm())
            axes[2][i].matshow(simulated_img[0][0][:,:,40].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
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

    def evaluate(self, dataloader, longitudinal=None, individual_RER=None, writer=None, train_losses=None):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.eval()
        self.training = False
        criterion = self.loss
        fs_criterion = nn.MSELoss()
        tloss = 0.0
        tvae_loss, tfs_loss = 0.0, 0.0
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
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed, fs_output = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    target_fs = torch.tensor([(data[3][i], data[4][i]) for i in range(len(data[3]))]).to(device)    # target fs volumes
                    fs_loss = fs_criterion(fs_output, target_fs)                                                    # volume prediction error
                    loss = reconstruction_loss + self.beta * kl_loss + self.kappa * fs_loss

                    tfs_loss += fs_loss
                    tvae_loss += reconstruction_loss + self.beta * kl_loss

                    loss = reconstruction_loss + self.beta * kl_loss

                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, mu.to('cpu')), 0)

        if writer is not None:
            writer.add_scalars('Loss/recon', {'test' : trecon_loss/nb_batches, 'train' : train_losses[0]} , self.epoch)
            writer.add_scalars('Loss/kl', {'test' : tkl_loss/nb_batches, 'train' : train_losses[1]}, self.epoch)
            writer.add_scalars('Loss/alignment', {'test' : talignment_loss/nb_batches, 'train' : train_losses[2]}, self.epoch)

        test_loss, testfs_loss, testvae_loss = tloss/nb_batches, tfs_loss/nb_batches, tvae_loss/nb_batches
        self.training = True
        return test_loss, testfs_loss, testvae_loss, encoded_data

    def train_(self, data_loader, test_loader, optimizer, num_epochs=20, d_optimizer=None, longitudinal=None, individual_RER=None, writer=None):

        self.to(device)
        criterion = self.loss
        fs_criterion = nn.MSELoss()
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch+1, num_epochs ))

            tloss = 0.0
            trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
            tvae_loss, tfs_loss = 0.0, 0.0
            tmu, tlogvar = torch.zeros((1,Settings().dimension)).to(device), torch.zeros((1,Settings().dimension)).to(device)
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
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed, fs_output = self.forward(input_)                                     # forward pass  
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)                     # VAE loss
                    target_fs = torch.tensor([(data[3][i], data[4][i]) for i in range(len(data[3]))]).to(device)    # target fs volumes
                    fs_loss = fs_criterion(fs_output, target_fs)                                                    # volume prediction error
                    loss = reconstruction_loss + self.beta * kl_loss + self.kappa * fs_loss

                    tfs_loss += fs_loss
                    tvae_loss += reconstruction_loss + self.beta * kl_loss

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1
            epoch_loss, epochfs_loss, epochvae_loss = tloss/nb_batches, tfs_loss/nb_batches, tvae_loss/nb_batches

            if writer is not None:
                self.epoch += 1
                train_losses = (trecon_loss/nb_batches, tkl_loss/nb_batches, talignment_loss/nb_batches)
                test_loss, _ = self.evaluate(test_loader, longitudinal=longitudinal, individual_RER=individual_RER, writer=writer, train_losses=train_losses)
                writer.add_histogram('Mu', tmu, self.epoch)
                writer.add_histogram('Logvar', tlogvar, self.epoch)
            else:
                test_loss, testfs_loss, testvae_loss, _ = self.evaluate(test_loader)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")
            logger.info(f"VAE loss (train/test): {epochvae_loss:.3e}/{testvae_loss:.3e} and fs loss :  {torch.sqrt(epochfs_loss):.3e}/{torch.sqrt(testfs_loss):.3e} ")

            if not(epoch%10):
                # Save images to check quality as training goes
                if longitudinal is not None:
                    self.plot_images_vae(test.data, 10, name='qc_reconstruction_test.png')
                    self.plot_images_vae(data[0], 10, name='qc_reconstruction_train.png')
                else:
                    self.plot_images_vae(next(iter(test_loader))[0], 10, name='qc_reconstruction_test.png')
                    self.plot_images_vae(data[0], 10, name='qc_reconstruction_train.png')

        print('Complete training')
        return

class Dataset(data.Dataset):
    def __init__(self, images, labels, timepoints, ventricles, hippocampus):
        self.data = images
        self.labels = labels
        self.timepoints = timepoints
        self.ventricles = ventricles
        self.hippocampus = hippocampus

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.data[index]
        y = self.labels[index]
        z = self.timepoints[index]
        v = self.ventricles[index]
        h = self.hippocampus[index]
        return X, y, z, v, h


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
    logger.info("DEBUGGING THE networks_SLVAE.py FILE")
    logger.info(f"Device is {device}")

    Settings().dimension = 16
    epochs = 250
    batch_size = 10
    lr = 6e-5

    # Load data
    train_data = torch.load('../../../LAE_experiments/ADNI_data/ADNI_t1', map_location='cpu')
    print(f"Loaded {len(train_data['data'])} scans")
    train_data['data'].requires_grad = False
    torch_data = Dataset(train_data['data'].unsqueeze(1).float(), train_data['labels'], train_data['timepoints'],\
                train_data['ventricles'], train_data['hippocampus'])
    train, test = torch.utils.data.random_split(torch_data, [len(torch_data)-200, 200])
    print(f"Loaded {len(train)} train scans, {len(test)} test scans")
     
    autoencoder = CVAE_3D()

    autoencoder.beta = 2
    autoencoder.kappa = 1e5

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                              shuffle=True, num_workers=1, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=True, num_workers=1, drop_last=True)

    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")
    logger.info(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)} parameters")

    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    autoencoder.train_(train_loader, test_loader, optimizer=optimizer, num_epochs=epochs)
    torch.save(autoencoder.state_dict(), "SLVAE_debugging")

    return autoencoder



if __name__ == '__main__':
    main()