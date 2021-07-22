import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision
from torch.utils import data

# This is a dirty workaround for a stupid problem with pytorch and osx that mismanage openMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random


class CAE(nn.Module):
    """
    This is the convolutionnal autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    """

    def __init__(self):
        super(CAE, self).__init__()
        nn.Module.__init__(self)

        # Encoder
        self.pad = torch.nn.ConstantPad3d((0,6,0,8,0,3), 0)
        self.conv1 = nn.Conv3d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(8, 8, 3, stride=1, padding=0)
        self.pool1 = nn.MaxPool3d(2, padding=0)
        self.pool2 = nn.MaxPool3d(2, padding=1)
        self.fc1 = nn.Linear(14784, 600)
        self.fc2 = nn.Linear(600, 600)

        # Decoder
        self.fc3 = nn.Linear(600, 14784)
        self.up1 = nn.ConvTranspose3d(8, 8, 3, stride=2, padding=1)
        self.up2 = nn.ConvTranspose3d(8, 8, 3, stride=2, padding=0)
        self.up3 = nn.ConvTranspose3d(8, 16, 3, stride=2, padding=1)
        self.conv = nn.Conv3d(16, 1, 4, stride=1, padding=2)

    def encoder(self, image):
        h1 = F.relu(self.conv1(self.pad(image)))
        h2 = self.pool1(h1)
        h3 = F.relu(self.conv2(h2))
        h4 = self.pool1(h3)
        h5 = F.relu(self.conv3(h4))
        h6 = self.pool2(h5)
        h7 = F.relu(self.fc1(h6.flatten(start_dim=1)))
        h8 = F.relu(self.fc2(h7))
        h8 = h8.view(h8.size())
        return h8

    def decoder(self, encoded):
        h9 = F.relu(self.fc3(encoded))
        h10 = F.relu(self.up1(h9.reshape([encoded.size()[0], 8, 11, 14, 12])))
        h11 = F.relu(self.up2(h10))
        h12 = F.relu(self.up3(h11))
        reconstructed = self.conv(h12)
        reconstructed = torch.sigmoid(reconstructed)
        reconstructed = reconstructed.view(reconstructed.size())
        return reconstructed[:,:,:85,:104,:90]

    def forward(self, image):
        encoded = self.encoder(image)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed

    def train(self, data_loader, size, criterion, optimizer, num_epochs=20, early_stopping=1e-7):
        print('Start training')
        best_loss = 1e10
        early_stopping = 0
        
        for epoch in range(num_epochs):
            
            if early_stopping == 100:
                break

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            
            tloss = 0.0
            for data in data_loader:
                (idx, timepoints), images = data
                optimizer.zero_grad()
                input_ = Variable(images).cuda()
                encoded, reconstructed = self.forward(input_)
                loss = criterion(reconstructed, input_)
                loss.backward()
                optimizer.step()
                tloss += float(loss)
            epoch_loss = tloss / size

            if epoch_loss <= best_loss:
                early_stopping = 0
                best_loss = epoch_loss
            else:
                early_stopping += 1
            print('Epoch loss: {:4f}'.format(epoch_loss))
        print('Complete training')
        return


class LAE(nn.Module):
    """
    This is the longitudinal autoencoder that takes as input the latent variables from the CAE and tries to
    both align its latent representation according to the individual trajectories and reconstruct its input.
    For this to work the decoder from the CAE must be highly Lipschitzien but it that doesn't work we can
    change the loss to reconstruct the MRI directly.
    """

    def __init__(self, args):
        super(Autoencoder, self).__init__()
        nn.Module.__init__(self)

        # encoder network
        self.fc1 = nn.Linear(args.input_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(400, args.latent_dim)

        # decoder network
        self.fc4 = nn.Linear(args.latent_dim, 400)
        self.fc5 = nn.Linear(400, 500)
        self.fc6 = nn.Linear(500, args.input_dim)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3

    def decoder(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return F.sigmoid(h6)

    def forward(self, input):
        encoded = self.encoder(input)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed

    def train(self, data_loader, size, criterion, optimizer, num_epochs=20):
        print('Start training')
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            tloss = 0.0
            for data in data_loader:
                inputs = data['data']
                optimizer.zero_grad()
                encoded, reconstructed = self(Variable(inputs.cuda()))
                loss = criterion(data, encoded, reconstructed, Variable(inputs.cuda()))
                loss.backward()
                optimizer.step()
                tloss += loss.data[0]
            epoch_loss = tloss / size
            print('Epoch loss: {:4f}'.format(epoch_loss))
        print('Complete training')
        return


class Dataset(data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.data = images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.data[index]
        y = self.labels[index]
        return X, y


def main():
    epochs = 5000
    es = 1e-6
    batch_size = 10
    in_dim = 28
    lr = 0.01

    # Load data
    train_data = torch.load('../../../LAE_experiments/small_dataset')
    print(f"Loaded {len(train_data['data'])} MRI scans")
    torch_data = Dataset(train_data['target'], train_data['data'].unsqueeze(1))
    data_loader = torch.utils.data.DataLoader(torch_data, batch_size=batch_size,
                                              shuffle=True, num_workers=4, drop_last=True)
    autoencoder = CAE().cuda()
    criterion = nn.BCELoss()
    size = len(train_data)
    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    autoencoder.train(data_loader, size, criterion, optimizer, num_epochs=epochs)

    test_image = random.choice(train_data['data']).cuda()
    test_image = Variable(test_image.unsqueeze(0).unsqueeze(0))
    _, out = autoencoder(test_image)

    torchvision.utils.save_image(test_image[0][0][30], 'in.png')
    torchvision.utils.save_image(out[0][0][30], 'out.png')


if __name__ == '__main__':
    main()



