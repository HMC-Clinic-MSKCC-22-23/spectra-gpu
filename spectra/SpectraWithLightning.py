import numpy as np
import torch
from collections import OrderedDict
from opt_einsum import contract
from scipy.special import logit
from tqdm import tqdm
from scipy.special import xlogy
from scipy.special import softmax
from spectra import spectra_util
import lightning.pytorch as pl
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
        X_b, alpha_b = batch
        loss = self.compute_loss(X_b, alpha_b)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.internal_model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
        return [optimizer], [lr_scheduler]
    
    def compute_loss(self, X, alpha):
        assert(self.use_cell_types) #if this is False, fail because model has not been initialized to use cell types
        
        batch_size = 1000 # can be changed here, like num_epochs

        # create the weird softmax theta 
        theta = torch.cat([temp.softmax(dim = 1) for temp in torch.split(self.internal_model.theta, split_size_or_sections = self.internal_model.L_list, dim = 1)], dim = 1)
        #initialize loss and fetch global parameters
        eta = self.internal_model.eta.exp()/(1.0 + (self.internal_model.eta).exp())
        eta = 0.5*(eta + eta.T)
        gene_scaling = self.internal_model.gene_scalings.exp()/(1.0 + self.internal_model.gene_scalings.exp()) #ctp1 x p
        kappa = self.internal_model.kappa.exp()/(1 + self.internal_model.kappa.exp()) #ctp1
        rho = self.internal_model.rho.exp()/(1 + self.internal_model.rho.exp()) #ctp1
        #handle theta x gene_scalings 
        
        gene_scaling_ = contract('ij,ki->jk',gene_scaling, self.internal_model.factor_to_celltype) #p x L_tot
        theta_ = theta * (gene_scaling_ + self.internal_model.delta)  #p x L_tot
        alpha_ = torch.exp(alpha) # should get the same thing as original gpu implementation, which is batch x L_tot
        recon = contract('ik,jk->ij' , alpha_, theta_)
        term1 = -1.0*(torch.xlogy(X,recon) - recon).sum()
        
        eta_ = eta[None,:,:]*self.internal_model.B_mask #ctp1 x L_tot x L_tot 
        
        mat = contract('il,clj,kj->cik',theta_,eta_,theta_) #ctp1 x p x p
        term2 = -1.0*((torch.xlogy(self.internal_model.adj_matrix*self.internal_model.weights, (1.0 - rho.reshape(-1,1,1))*(1.0 -kappa.reshape(-1,1,1))*mat + (1.0 - rho.reshape(-1,1,1))*kappa.reshape(-1,1,1)))*self.internal_model.ct_vec.reshape(-1,1,1)).sum()
        term3 = -1.0*((torch.xlogy(self.internal_model.adj_matrix_1m,(1.0 -kappa.reshape(-1,1,1))*(1.0 - rho.reshape(-1,1,1))*(1.0 - mat) + rho.reshape(-1,1,1)))*self.internal_model.ct_vec.reshape(-1,1,1)).sum()
        loss = (self.n/batch_size)*self.internal_model.lam*term1 + term2 + term3 #upweight local param terms to be correct on expectation 
        return loss


class est_spectra():
    def __init__(self):
        return 0
