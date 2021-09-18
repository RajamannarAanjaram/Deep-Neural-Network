
import torch
from src.dataLoader import Loader
from src.plots import Plots
from src.model import Model_loader
from src.optimise import learner


train_transform,test_transform = Loader.transform()
train_data,test_data = Loader.Loader(train_transform,test_transform)

# Plots.sampleVisual(train_data)


use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'

modelParams = {'LN': 0, 'GN': 0, 'BN': 1}


for key,val in modelParams.items():
    model = Model_loader.models(key,device)
    learner(model, train_data, test_data, val, 3, device)

