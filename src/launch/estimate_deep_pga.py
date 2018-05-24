import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
import torchvision
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import warnings
import time

# Estimators

from pydeformetrica.src.core.estimators.mcmc_saem import McmcSaem
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from pydeformetrica.src.core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from pydeformetrica.src.core.observations.manifold_observations.image import Image
from pydeformetrica.src.core.model_tools.manifolds.metric_learning_nets import MnistNet
from pydeformetrica.src.core.models.deep_pga import DeepPga
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA


# Runs the deep pga model on 1000 randomly extracted digits for mnist, for varying latent space dimensions.
# Experimental for now.


#for latent_space_dimension in range(2, 20):
for latent_space_dimension in [2]:


    Settings().dimension = 2
    Settings().output_dir = 'output_' + str(latent_space_dimension)

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)


    trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True)

    train_dataset = LongitudinalDataset()
    test_dataset = LongitudinalDataset()

    train_labels = []
    test_labels = []

    for elt in trainset:
        image, label = elt
        #if label == 2:
        if len(train_dataset.deformable_objects) < 800:
            img = Image()
            img.set_points(np.array(image, dtype=float)/np.max(image))
            img.update()
            train_dataset.deformable_objects.append([img])
            train_dataset.subject_ids.append(len(train_dataset.deformable_objects))
            train_labels.append(label)
        elif len(test_dataset.deformable_objects) < 200:
            img = Image()
            img.set_points(np.array(image, dtype=float)/np.max(image))
            img.update()
            test_dataset.deformable_objects.append([img])
            test_dataset.subject_ids.append(len(test_dataset.deformable_objects))
            test_labels.append(label)

    np.savetxt(os.path.join(Settings().output_dir, 'labels_train.txt'), np.array(train_labels))
    np.savetxt(os.path.join(Settings().output_dir, 'labels_test.txt'), np.array(test_labels))

    a, b = np.array(image).shape

    train_dataset.update()
    test_dataset.update()

    images_data = np.array([elt[0].get_points() for elt in train_dataset.deformable_objects])
    images_data_torch = torch.from_numpy(images_data).type(Settings().tensor_scalar_type)

    pca = PCA(n_components=latent_space_dimension)
    latent_space_positions = pca.fit_transform([elt[0].get_points().flatten() for elt in train_dataset.deformable_objects])
    reconstructed = pca.inverse_transform(latent_space_positions)

    latent_test = pca.transform([elt[0].get_points().flatten() for elt in test_dataset.deformable_objects])
    reconstructed_test = pca.inverse_transform(latent_test)

    reconstruction_error_train = mean_squared_error(reconstructed, [elt[0].get_points().flatten() for elt in train_dataset.deformable_objects])
    reconstruction_error_test = mean_squared_error(reconstructed_test, [elt[0].get_points().flatten() for elt in test_dataset.deformable_objects])
    print('PCA mse on train:', reconstruction_error_train)
    print('PCA mse on test:', reconstruction_error_train)
    to_write = np.array([reconstruction_error_train, reconstruction_error_test])
    np.savetxt(os.path.join(Settings().output_dir, 'pca_reconstruction_error.txt'), to_write)

    # We now normalize every latent_space_positions
    for i in range(latent_space_dimension):
        latent_space_positions[:, i] = (latent_space_positions[:, i] - np.mean(latent_space_positions[:, i]))/np.std(latent_space_positions[:, i])

    latent_space_positions_torch = torch.from_numpy(latent_space_positions).type(Settings().tensor_scalar_type)


    # We now instantiate the neural network
    net = MnistNet(in_dimension=latent_space_dimension)
    #net.double()
    criterion = nn.MSELoss()
    nb_epochs = 50
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    train_dataset_fit = TensorDataset(latent_space_positions_torch, images_data_torch)
    train_dataloader = DataLoader(train_dataset_fit, batch_size=10, shuffle=True)


    for epoch in range(nb_epochs):
        train_loss = 0
        nb_train_batches = 0
        for (z, y) in train_dataloader:
            nb_train_batches += 1
            var_z = Variable(z)
            var_y = Variable(y)
            predicted = net(var_z)
            loss = criterion(predicted, var_y)
            net.zero_grad()
            loss.backward()
            train_loss += loss.cpu().data.numpy()[0]
            optimizer.step()

        train_loss /= nb_train_batches

        print("Epoch {}/{}".format(epoch, nb_epochs),
              "Train loss:", train_loss)


    metric_parameters = net.get_parameters()

    # We instantiate the model
    model = DeepPga()
    model.net = net
    model.set_metric_parameters(metric_parameters)

    model.latent_space_dimension = latent_space_dimension

    model.set_noise_variance(train_loss / (a*b))
    model.update()

    model.individual_random_effects['latent_position'].mean = np.zeros((latent_space_dimension,))
    model.individual_random_effects['latent_position'].set_variance(1.0)

    individual_RER = {}
    individual_RER['latent_position'] = latent_space_positions

    sampler = SrwMhwgSampler()
    estimator = McmcSaem()
    estimator.sampler = sampler

    # latent positions proposal:
    latent_position_proposal_distribution = MultiScalarNormalDistribution()
    latent_position_proposal_distribution.set_variance_sqrt(0.1)
    sampler.individual_proposal_distributions['latent_position'] = latent_position_proposal_distribution

    estimator.sample_every_n_mcmc_iters = 10

    estimator.max_iterations = 200
    estimator.number_of_burn_in_iterations = 200
    estimator.max_line_search_iterations = 10
    estimator.convergence_tolerance = 1e-3

    estimator.print_every_n_iters = 1
    estimator.save_every_n_iters = 30

    estimator.dataset = train_dataset
    estimator.statistical_model = model

    # Initial random effects realizations
    estimator.individual_RER = individual_RER

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'DeepPgaModel'
    print('')
    print('[ update method of the ' + estimator.name + ' optimizer ]')

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    print('>> Estimation took: ' + str(time.strftime("%d days, %H hours, %M minutes and %S seconds.",
                                                         time.gmtime(end_time - start_time))))

    # We now need to estimate the residual on the test set... we create a new estimator.
    model.is_frozen['metric_parameters'] = True
    model.is_frozen['noise_variance'] = True


    #estimator = GradientAscent()

    estimator = ScipyOptimize()
    estimator.memory_length = 10

    Settings().output_dir = 'test_output_' + str(latent_space_dimension)
    if not os.path.isdir(Settings().output_dir):
        os.mkdir(Settings().output_dir)

    estimator.sample_every_n_mcmc_iters = 25
    estimator.convergence_tolerance = 1e-3
    estimator.max_line_search_iterations = 10

    estimator.max_iterations = 300

    estimator.print_every_n_iters = 1
    estimator.save_every_n_iters = 10

    estimator.dataset = test_dataset
    estimator.statistical_model = model

    individual_RER = {}
    individual_RER['latent_position'] = np.zeros((200, latent_space_dimension))
    estimator.individual_RER = individual_RER

    estimator.update()
    estimator.write()


