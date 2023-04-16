import numpy as np
import torch
from collections import OrderedDict
from opt_einsum import contract
from scipy.special import logit
from tqdm import tqdm
from scipy.special import xlogy
from scipy.special import softmax
from spectra import spectra_util
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import scipy
import pandas as pd
from pyvis.network import Network
import random

from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.dirichlet import Dirichlet

class SPECTRA(nn.Module):
   def __init__(self):
       return 0
           
class SPECTRA_DataModule(pl.LightningDataModule):
    def __init__(self):
        return 0

class SPECTRA_LitModel(pl.LightningModule):
    def __init__(self, internal_model):
        super().__init__()
        self.internal_model = internal_model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class est_spectra():
    def __init__(self):
        return 0
