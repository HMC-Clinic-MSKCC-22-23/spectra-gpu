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
from pytorch_lightning.callbacks import Callback

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

        self.cell_scores = None
        self.factors = None
        self.B_diag = None
        self.eta_matrices = None 
        self.gene_scalings = None 
        self.rho = None 
        self.kappa = None 

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
    
    def save(self, fp):
        torch.save(self.internal_model.state_dict(),fp)
  
    def load(self,fp,labels = None):
        self.internal_model.load_state_dict(torch.load(fp))
    
    def return_eta_diag(self):
        return self.B_diag
    def return_cell_scores(self):
        return self.cell_scores
    def return_factors(self):
        return self.factors 
    def return_eta(self):
        return self.eta_matrices
    def return_rho(self):
        return self.rho 
    def return_kappa(self):
        return self.kappa
    def return_gene_scalings(self): 
        return self.gene_scalings
    def return_graph(self, ct = "global"):
        model = self.internal_model
        if self.use_cell_types:
            eta = (model.eta[ct]).exp()/(1.0 + (model.eta[ct]).exp())
            eta = 0.5*(eta + eta.T)
            theta = torch.softmax(model.theta[ct], dim = 1)
            mat = contract('il,lj,kj->ik',theta,eta,theta).detach().numpy()
        else: 
            eta = model.eta.exp()/(1.0 + model.eta.exp())
            eta = 0.5*(eta + eta.T)
            theta = torch.softmax(model.theta, dim = 1)
            mat = contract('il,lj,kj->ik',theta,eta,theta).detach().numpy()
        return mat
        
    def matching(self, markers, gene_names_dict, threshold = 0.4):
        """
        best match based on overlap coefficient
        """
        markers = pd.DataFrame(markers)
        if self.use_cell_types:
            matches = []
            jaccards = []
            for i in range(markers.shape[0]):
                max_jacc = 0.0 
                best = ""
                for key in gene_names_dict.keys():
                    for gs in gene_names_dict[key].keys():
                        t = gene_names_dict[key][gs]

                        jacc = spectra_util.overlap_coefficient(list(markers.iloc[i,:]),t)
                        if jacc > max_jacc:
                            max_jacc = jacc
                            best = gs 
                matches.append(best)
                jaccards.append(max_jacc)
            
        else:
            matches = []
            jaccards = []
            for i in range(markers.shape[0]):
                max_jacc = 0.0 
                best = ""
                for key in gene_names_dict.keys():
                    t = gene_names_dict[key]

                    jacc = spectra_util.overlap_coefficient(list(markers.iloc[i,:]),t)
                    if jacc > max_jacc:
                        max_jacc = jacc
                        best = key 
                matches.append(best)
                jaccards.append(max_jacc)
        output = []
        for j in range(markers.shape[0]):
            if jaccards[j] > threshold:
                output.append(matches[j])
            else:
                output.append("0")
        return np.array(output)

class SPECTRA_Callback(Callback):
    def on_train_end(self, trainer, pl_module):

        model = pl_module.internal_model

        out = (torch.exp(model.alpha)*model.alpha_mask).detach().cpu().numpy()
        theta = torch.cat([temp.softmax(dim = 1) for temp in torch.split(model.theta, split_size_or_sections = model.L_list, dim = 1)], dim = 1)

        gene_scaling = model.gene_scalings.exp()/(1.0 + model.gene_scalings.exp())
        gene_scaling_ = contract('ij,ki->jk',gene_scaling, model.factor_to_celltype) #p x L_tot
        scaled = (theta * (gene_scaling_ + model.delta)).T.detach().cpu().numpy()

        new_factors = scaled/(scaled.sum(axis = 0,keepdims =True) + 1.0)
        cell_scores = out*scaled.mean(axis = 1).reshape(1,-1) 

        # calculate B_diag
        Bg = model.eta.exp()/(1.0 + model.eta.exp())
        Bg = 0.5*(Bg + Bg.T)
        self.B_diag = torch.diag(Bg).detach().cpu().numpy()


        # calculate eta matrix
        eta = OrderedDict()
        Bg = model.eta.exp()/(1.0 + model.eta.exp())
        Bg = 0.5*(Bg + Bg.T)

        for ct in model.ct_order:
            eta[ct] = Bg[model.start_pos[ct]: model.start_pos[ct] +model.L[ct], model.start_pos[ct]: model.start_pos[ct] +model.L[ct]].detach().cpu().numpy()
        self.eta_matrices = eta

        
        #new store params stuff 
        self.cell_scores = cell_scores
        self.factors = new_factors
        self.gene_scalings = {ct : gene_scaling[i].detach().cpu().numpy() for i, ct in enumerate(model.ct_order)}
        self.rho = {ct: model.rho[i].exp().detach().cpu().numpy()/(1.0 + model.rho[i].exp().detach().cpu().numpy()) for i, ct in enumerate(model.ct_order)}
        self.kappa = {ct: model.kappa[i].exp().detach().cpu().numpy()/(1.0 + model.kappa[i].exp().detach().cpu().numpy()) for i, ct in enumerate(model.ct_order)}


class est_spectra():
    def __init__(self):
        return 0
