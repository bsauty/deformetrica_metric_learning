from doctest import testsource
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import numpy as np
import time
import logging
import sys

# This is a dirty workaround for a  problem with pytorch and osx that mismanage openMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ffs(nn.Module):
    def __init__(self):
        super(ffs, self).__init__()
        nn.Module.__init__(self)
            
        self.d_conv1 = nn.Conv3d(1, 64, 3, stride=2, padding=1)               # 32 x 40 x 48 x 40
        self.d_conv2 = nn.Conv3d(64, 128, 3, stride=2, padding=1)             # 64 x 20 x 24 x 20
        self.d_conv3 = nn.Conv3d(128, 128, 3, stride=2, padding=1)            # 128 x 10 x 12 x 10
        #self.d_conv4 = nn.Conv3d(128, 256, 3, stride=2, padding=1)           # 256 x 5 x 6 x 5
        self.d_conv5 = nn.Conv3d(128, 1, 1, stride=1)              # 1 x 10 x 12 x 10
        self.d_bn1 = nn.BatchNorm3d(32)
        self.d_bn2 = nn.BatchNorm3d(64)
        self.d_bn3 = nn.BatchNorm3d(128)
        self.d_bn4 = nn.BatchNorm3d(256)
        self.d_bn5 = nn.BatchNorm3d(1)
        self.relu1 = nn.LeakyReLU(0.02, inplace=True)
        self.relu2 = nn.LeakyReLU(0.02, inplace=True)
        self.relu3 = nn.LeakyReLU(0.02, inplace=True)
        self.relu4 = nn.LeakyReLU(0.02, inplace=True)
        self.relu5 = nn.LeakyReLU(0.02, inplace=True)
        #self.d_fc1 = nn.Linear(38400, 500)
        self.dropout = nn.Dropout(.9)
        self.d_fc = nn.Linear(1200, 2)
        
    def forward(self, image):
        d1 = self.relu1(self.d_conv1(image))
        d2 = self.relu2(self.d_conv2(d1))
        d3 = self.relu3(self.d_conv3(d2))
        #d4 = self.relu4(self.d_conv4(d3))
        d5 = self.relu5(self.d_conv5(d3))
        d6 = torch.sigmoid(self.d_fc(F.dropout(d5.flatten(start_dim=1), .3)))
        return d6
    
    def evaluate(self, test_loader, criterion=nn.MSELoss()):
        self.to(device)
        self.training = False
        self.eval()

        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in test_loader:

                input_ = Variable(data[0]).to(device)
                output = self.forward(input_)
                target = torch.tensor([(data[3][i], data[4][i]) for i in range(len(data[3]))]).to(device)
                loss = criterion(output, target)
                tloss += float(loss)
                nb_batches += 1

        return tloss/nb_batches

        
    def train_(self, data_loader, test_loader, optimizer, num_epochs=20, criterion=nn.MSELoss()):

        self.to(device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time.time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch+1, num_epochs ))

            tloss = 0.0
            nb_batches = 0

            for data in data_loader:
                optimizer.zero_grad()

                input_ = Variable(data[0]).to(device)
                output = self.forward(input_)
                target = torch.tensor([(data[3][i], data[4][i]) for i in range(len(data[3]))]).to(device)
                loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1
            epoch_loss = tloss/nb_batches

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1

            test_loss = self.evaluate(test_loader, criterion)
            
            end_time = time.time()
            logger.info(f"Epoch loss (train/test): {np.sqrt(epoch_loss):.3e}/{np.sqrt(test_loss):.3e} took {end_time-start_time} seconds")

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

def main():
    """
    For debugging purposes only, once the architectures and training routines are efficient,
    this file will not be called as a script anymore.
    """
    print
    logger.info("Testing a light freesurfer alternative that outputs ventricles and hippocampus volumes from t1 scans")
    logger.info(f"Device is {device}")

    epochs = 100
    batch_size = 10
    lr = 1e-4

    # Load data
    train_data = torch.load('../../../LAE_experiments/ADNI_data/ADNI_t1')
    train_data['data'].requires_grad = False
    torch_data = Dataset(train_data['data'].unsqueeze(1).float(), train_data['labels'], train_data['timepoints'],\
                train_data['ventricles'], train_data['hippocampus'])
    train, test = torch.utils.data.random_split(torch_data, [len(torch_data)-200, 200])
    print(f"Loaded {len(train)} train scans, {len(test)} test scans")
    
    predictor = ffs()
    criterion = nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                              shuffle=True, num_workers=1, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=True, num_workers=1, drop_last=True)
    print(f"Model has a total of {sum(p.numel() for p in predictor.parameters())} parameters")

    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(predictor.parameters(), lr=lr)
    predictor.train_(train_loader, test_loader, criterion=criterion, optimizer=optimizer, num_epochs=epochs)
    torch.save(predictor.state_dict(), 'predictor')

    return predictor


if __name__ == '__main__':
    main()