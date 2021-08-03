import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision
from torch.utils import data
from time import time
from PIL import Image

# This is a dirty workaround for a stupid problem with pytorch and osx that mismanage openMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random

class CAE(nn.Module):
    """
    This is the convolutionnal autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    This is the architecture suggested in Martinez-Mucia et al.
    """

    def __init__(self):
        super(CAE, self).__init__()
        nn.Module.__init__(self)

        # Encoder
        self.pad = torch.nn.ConstantPad3d((0,6,0,8,0,3), 0)
        self.conv1 = nn.Conv3d(1, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv3d(32, 32, 5, stride=2)
        self.conv3 = nn.Conv3d(32, 64, 5, stride=1, padding=2)
        self.conv4 = nn.Conv3d(64, 64, 5, stride=2)
        self.conv5 = nn.Conv3d(64, 256, 3,stride=2)
        self.conv6 = nn.Conv3d(256, 512, 3, stride=2, padding=1)
        #self.fc1 = nn.Linear(76800, 512)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)

        # Decoder
        self.fc2 = nn.Linear(512, 1080)
        self.up1 = nn.ConvTranspose3d(1, 256, 5, stride=2)
        self.up2 = nn.Conv3d(256, 256, 5, stride=1, padding=2)
        self.up3 = nn.ConvTranspose3d(256, 64, 5, stride=2)
        self.up4 = nn.Conv3d(64, 64, 5, stride=1, padding=2)
        self.up5 = nn.ConvTranspose3d(64, 32, 3, stride=2)
        self.conv = nn.Conv3d(32, 1, 3, stride=1, padding=2)
        self.bn3 = nn.BatchNorm3d(1)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(64)

    def encoder(self, image):
        h1 = F.relu(self.conv1(self.pad(image)))
        h2 = self.bn1(F.relu(self.conv2(h1)))
        h3 = F.relu(self.conv3(h2))
        h4 = self.bn2(F.relu(self.conv4(h3)))
        h5 = F.relu(self.conv5(h4))
        h6 = F.relu(self.conv6(h5))
        h7 = self.fc1(h6.flatten(start_dim=1))  # Dense layer after convolutions
        h7 = torch.sigmoid(h7).view(h7.size())
        return h7

    def encoder_gap(self, image):
        h1 = F.relu(self.conv1(self.pad(image)))
        h2 = self.bn1(F.relu(self.conv2(h1)))
        h3 = F.relu(self.conv3(h2))
        h4 = self.bn2(F.relu(self.conv4(h3)))
        h5 = F.relu(self.conv5(h4))
        h6 = F.relu(self.conv6(h5))
        h7 = h6.mean(dim=(-3,-2,-1))  # Global average pooling layer after convolutions
        h7 = torch.sigmoid(h7).view(h7.size())
        return h7

    def decoder(self, encoded):
        h9 = self.bn3(F.relu(self.fc2(encoded)).reshape([encoded.size()[0], 1, 9, 12, 10]))
        h10 = F.relu(self.up1(h9))
        h11 = self.bn4(F.relu(self.up2(h10)))
        h12 = F.relu(self.up3(h11))
        h13 = self.bn5(self.up4(h12))
        h14 = self.up5(h13)
        reconstructed = torch.sigmoid(self.conv(h14))
        return reconstructed[:,:,:85,:104,:90]

    def forward(self, image):
        encoded = self.encoder_gap(image)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed
    
    def plot_images(self, data, n_images):
        im_list = []
        for i in range(n_images):
            test_image = random.choice(data)[0]
            test_image = Variable(test_image.unsqueeze(0))
            _, out = self.forward(test_image)

            im_list.append(Image.fromarray(255*test_image[0][0][30].cpu().detach().numpy()).convert('RGB'))
            im_list.append(Image.fromarray(255*out[0][0][30].cpu().detach().numpy()).convert('RGB'))

        im_list[0].save("Quality_control.pdf", "PDF", resolution=100.0, save_all=True, append_images=im_list[1:])

    def evaluate(self, data, criterion):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        encoded_data = torch.empty([0,512])
        with torch.no_grad():
            for data in dataloader:
                input_ = Variable(data)
                encoded, reconstructed = self.forward(input_)
                loss = criterion(reconstructed, input_)
                tloss += float(loss)
                encoded_data = torch.cat((encoded_data, encoded), 0)
            loss = tloss / len(data)
        return loss, encoded_data

    def train(self, data_loader, test, size, criterion, optimizer, num_epochs=20):
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):
            start_time = time()
            if es == early_stopping:
                break

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            tloss = 0.0
            for data in data_loader:
                images, _ = data
                optimizer.zero_grad()
                input_ = Variable(images)
                encoded, reconstructed = self.forward(input_)
                loss = criterion(reconstructed, input_)
                loss.backward()
                optimizer.step()
                tloss += float(loss)
            epoch_loss = tloss / size

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            print(f"Epoch loss: {epoch_loss} took {end_time-start_time} seconds")

            # Save images to check quality as training goes
            self.plot_images(test, 10)

        print('Complete training')
        return


class LAE(nn.Module):
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

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3

    def decoder(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return torch.sigmoid(h6)

    def forward(self, input):
        encoded = self.encoder(input)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed

    def evaluate(self, data, criterion):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        encoded_data = torch.empty([0,10])
        with torch.no_grad():
            for data in dataloader:
                input_ = Variable(data)
                encoded, reconstructed = self.forward(input_)
                loss = criterion(reconstructed, input_)
                tloss += float(loss)
                encoded_data = torch.cat((encoded_data, encoded), 0)
            loss = tloss / len(data)
        return loss, encoded_data


    def train(self, data_loader, test, size, criterion, optimizer, num_epochs=20):

        best_loss = 1e10
        early_stopping = 0

        for epoch in range(num_epochs):
            start_time = time()
            if early_stopping == 10:
                break

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            tloss = 0.0
            for data in data_loader:
                input_ = Variable(data)
                optimizer.zero_grad()
                encoded, reconstructed = self.forward(input_)
                loss = criterion(reconstructed, input_)
                loss.backward()
                optimizer.step()
                tloss += float(loss)
            epoch_loss = tloss / size
            test_loss, _ = self.evaluate(test, criterion)

            if epoch_loss <= best_loss:
                early_stopping = 0
                best_loss = epoch_loss
            else:
                early_stopping += 1
            end_time = time()
            print(f"Epoch loss (train/test): {epoch_loss}/{test_loss} took {end_time-start_time} seconds")

        print('Complete training')
        return


class Dataset(data.Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.data[index]
        y = self.labels[index]
        return X, y


def main():
    """
    For debugging purposes only, once the architectures and training routines are efficient,
    this file will not be called as a script anymore.
    """
    epochs = 250
    batch_size = 4
    early_stopping = 20
    lr = 0.000001

    # Load data
    train_data = torch.load('../../../LAE_experiments/encoded_dataset')
    print(f"Loaded {len(train_data['data'])} encoded scans")
    torch_data = Dataset(train_data['data'].unsqueeze(1), train_data['target'])
    train, test = torch.utils.data.random_split(torch_data, [len(torch_data)-5, 5])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                              shuffle=True, num_workers=4, drop_last=True)
    autoencoder = LAE()
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")
    criterion = nn.MSELoss()
    size = len(train)
    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    autoencoder.train(train_loader, test, size, criterion, optimizer, num_epochs=epochs)
    torch.save(autoencoder.state_dict(), 'CAE')


if __name__ == '__main__':
    main()



