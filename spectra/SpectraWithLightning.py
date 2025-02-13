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
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split

import torch.nn.functional as F
import torch.nn as nn
import scipy
import pandas as pd
from pyvis.network import Network
import random

from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.dirichlet import Dirichlet

### Class for SPECTRA model 
from spectra.initialization import *

class SPECTRA(nn.Module): 
    """

    Parameters
        ----------
        X : np.ndarray or torch.Tensor
            the ``(n, p)`` -shaped matrix containing logged expression count data. Used for initialization of self.n and self.p but not stored as an attribute
        labels : np.ndarray or NoneType
            the ``(n, )`` -shaped array containing cell type labels. If use_cell_types == False, then should be set to None

        L : dict or OrderedDict [if use_cell_types == False, then int]
            ``number of cell types + 1``-shaped dictionary. Must have "global" as a key, indicating the number of global factors
            {
                "global": 15,
                "CD8": 5
                ...
            }
            > Format matches output of K_est.py to estimate the number of
            > Must match cell type labels provided during training
            > Recommended practice is to assign at minimum 2 factors per cell type
            > Note that L contains the number of factors that describe the graph.
        device : the device on which to instantiate the tensors, given from the wrapper SPECTRA Lightning Module object
        adj_matrix :  dict or OrderedDict
            ``a dictionary of adjacency matrices, one for every cell type + a "global"
            {
                "global": ``(p, p)``-shaped binary np.ndarray
                "CD8": ...
            }
        weights : dict or OrderedDict or NoneType [if use_cell_types == False, then ``(p, p)``-shaped array]
            the ``(p, p)``-shaped set of edge weights per . If weight[i,j] is non-zero when adj_matrix[i,j] = 0
            this weight is ignored.

            if weights == None, no weights are used
        lam : float
            lambda parameter of the model, which controls the relative influence of the graph vs expression loss functions. This term multiplies the expression loss, so smaller values of lambda upweight the prior information
        delta : float
            delta parameter of the model, which controls a lower bound for gene scaling factors. If delta is small then the maximum ratio between gene scaling factors is larger and lowly expressed genes can be put on the same scale as highly expressed genes.
        kappa : float or NoneType
            kappa controls the background rate of edges in the graph. if kappa is a float, kappa is fixed to the given float value. If kappa == None, then kappa is a parameter that is estimated from the data.
        rho : float or NoneType
            rho controls the bakcground rate of non-edges in the graph. if rho is a float, rho is fixed to the given float value. If rho == None, then rho is a parameter that is estimated from the data.
        use_cell_types: bool
            use_cell_types is a Boolean variable that determines whether cell type labels are used to fit the model. If False, then parameters are initialized as nn.Parameter rather than as nn.ParameterDict with cell type keys that index nn.Parameter values

    Attributes
        ----------
        model.delta : delta parameter of the model
        model.lam : lambda parameter of the model


        model.L : L parameter, either int, dict or OrderedDict()
        model.p : number of genes
        model.n : number of cells
        model.use_cell_types : if True then cell types are considered, else cell types ignored. Affects the dimensions of the initialized parameters.
        model.kappa : if not kappa, nn.ParameterDict() if use_cell_types, else nn.Parameter(). If kappa is a float, it is fixed throughout training
        model.rho : if not rho, nn.ParamterDict() if use_cell_types, else nn.Parameter. If rho is a float it is fixed throughout training
        model.adj_matrix : adjacency matrix with diagonal removed. dict containing torch.Tensors
        model.adj_matrix_1m : 1 - adjacency matrix with diagonal removed. dict containing torch.Tensors

        model.weights : contains edge weights. format matches adj_matrix

        model.cell_types : np.ndarray containing array of unique cell types

        model.cell_type_counts : dict {key = cell type, values = number of cells}

        model.theta : nn.ParameterDict() or nn.Parameter() containing the factor weights
        model.alpha : nn.ParameterDict() or nn.Parameter() containing the cell loadings
        model.eta : nn.ParameterDict() or nn.Parameter() containing the interaction matrix between factors
        model.gene_scaling : nn.ParameterDict() or nn.Parameter() containing the gene scale factors
        model.selection :  nn.ParameterDict() or nn.Parameter() containing the attention weights. Only initialized when L[cell_type] > K[cell_type] for some cell type or when L > K and use_cell_types == False

        model.kgeql_flag : dict or bool. dictionary of boolean values indicating whether K >= L.  When use_cell_types == False, it is a boolean value
    Methods
        ----------
        model.initialize(self,annotations, word2id, W, init_scores, val = 25) : calculates the dictionary given to set_factor_weights

        model.set_factor_weights(self, gene_sets,val) : initialize the model based on given dictionary of gene sets. val is a float that determines the strength of the initialization.

        model.set_factor_weights_no_celltypes(self, gs_list, val) : initialize the model based on given list of gene sets. val is a float that determines the strength of the initialization.

    """

    # possible that the non-parameter tensors that get initialized here need be turned into buffers
    def __init__(self,X, labels, device, L, vocab = None, gs_dict = None, use_weights = False, adj_matrix = None, weights = None, lam = 0.1, delta=0.1,kappa = None, rho = None, use_cell_types = True):
        super(SPECTRA, self).__init__()


        # hyperparameters
        self.delta = delta
        self.lam = lam
        self.L = L
        self.use_cell_types = use_cell_types
        self.device = device

        # if gs_dict is provided instead of adj_matrix, convert to adj_matrix, overrides adj_matrix and weights
        if gs_dict is not None:
            gene2id = dict((v, idx) for idx, v in enumerate(vocab))

            if use_cell_types:
                adj_matrix, weights = spectra_util.process_gene_sets(gs_dict=gs_dict, gene2id=gene2id,
                                                                     weighted=use_weights)
            else:
                adj_matrix, weights = spectra_util.process_gene_sets_no_celltypes(gs_dict=gs_dict, gene2id=gene2id,
                                                                                  weighted=use_weights)

        # for memory efficiency we don't store X in the object attributes, but require X dimensions to be known at initialization
        self.p = X.shape[1]
        self.n = X.shape[0]

        print("Building parameter set...")
        # we only have to do this once so for loop is ok

        # here we are creating lists in place of the original input formats -- to be converted into 3-way tensors instead of dictionaries with a cell type axis
        lst_adj_matrix = []
        lst_adj_matrix1m = []
        lst_weights = []
        ct_order = []
        L_tot = 0
        L_list = []
        self.start_pos = OrderedDict()

        for cell_type in adj_matrix.keys():
            ct_order.append(cell_type)
            L_tot += L[cell_type]
            L_list.append(L[cell_type])
            if len(adj_matrix[cell_type]) > 0:
                temp = torch.Tensor(adj_matrix[cell_type]) - torch.Tensor(np.diag(np.diag(adj_matrix[cell_type])))
                lst_adj_matrix.append(temp)
                lst_adj_matrix1m.append(1.0 - temp - torch.diag(torch.diag(1.0 - temp)))
            else:
                lst_adj_matrix.append(torch.zeros((self.p, self.p)))
                lst_adj_matrix1m.append(torch.zeros((self.p, self.p)))
        #self.adj_matrix = torch.stack(lst_adj_matrix).to(self.device) #(cell_types + 1, p, p )
        self.register_buffer("adj_matrix", torch.stack(lst_adj_matrix))
        #self.adj_matrix_1m = torch.stack(lst_adj_matrix1m).to(self.device) #(cell_types + 1, p, p)
        self.register_buffer("adj_matrix_1m", torch.stack(lst_adj_matrix1m))

        if weights:
            for cell_type in ct_order:
                if len(weights[cell_type]) > 0:
                    lst_weights.append(
                        torch.Tensor(weights[cell_type]) - torch.Tensor(np.diag(np.diag(weights[cell_type]))))
                else:
                    lst_weights.append(torch.zeros((self.p, self.p)))
        else:
            self.weights = self.adj_matrix
        # self.weights = torch.stack(lst_weights) #.to(device) 
        self.register_buffer("weights", torch.stack(lst_weights))
        self.ct_order = ct_order
        self.L_tot = L_tot
        self.L_list = L_list
        self.n_cell_typesp1 = len(ct_order)

        # just need to construct these masks once
        #self.alpha_mask = torch.zeros((self.n,self.L_tot)).to(self.device)
        self.register_buffer("alpha_mask", torch.zeros((self.n, self.L_tot))) 
        #self.B_mask = torch.zeros((self.n_cell_typesp1, self.L_tot, self.L_tot)).to(self.device) #need to double check that this works
        self.register_buffer("B_mask", torch.zeros((self.n_cell_typesp1, self.L_tot, self.L_tot)))
        #self.factor_to_celltype = torch.zeros((self.L_tot, self.n_cell_typesp1)).to(self.device) 
        self.register_buffer("factor_to_celltype", torch.zeros((self.L_tot, self.n_cell_typesp1)))

        self.theta = nn.Parameter(Normal(0., 1.).sample([self.p, self.L_tot]))
        self.eta = nn.Parameter(Normal(0., 1.).sample([self.L_tot, self.L_tot]))
        self.alpha = nn.Parameter(Normal(0., 1.).sample([self.n, self.L_tot]))
        self.gene_scalings = nn.Parameter(Normal(0., 1.).sample([self.n_cell_typesp1, self.p]))

        # need to add kappa and rho initilizations and figure out the masking for B and loadings and how to multiply the gene scalings by the factors -- create a one hot vector for cell type --> factor assignments to get
        if kappa == None:
            self.kappa = nn.Parameter(Normal(0., 1.).sample([self.n_cell_typesp1]))
        if rho == None:
            self.rho = nn.Parameter(Normal(0., 1.).sample([self.n_cell_typesp1]))
        if kappa != None:
            #self.kappa = (torch.ones((self.n_cell_typesp1))*torch.tensor(np.log(kappa /(1-kappa)))).to(self.device) 
            self.register_buffer("kappa",
                                 (torch.ones((self.n_cell_typesp1)) * torch.tensor(np.log(kappa / (1 - kappa)))))
        if rho != None:
            #self.rho =  (torch.ones((self.n_cell_typesp1))*torch.tensor(np.log(rho /(1-rho)))).to(self.device) 
            self.register_buffer("rho", (torch.ones((self.n_cell_typesp1)) * torch.tensor(np.log(rho / (1 - rho)))))

        # make sure
        if ct_order[0] != "global":
            raise Exception("First key in ordered dict must be global")

        counter = 0
        counter_B = 0
        for ct in ct_order:
            self.start_pos[ct] = counter
            if ct == "global":
                self.alpha_mask[:, counter:counter + L[ct]] = 1.0
            else:
                cells_mask = labels == ct
                self.alpha_mask[cells_mask, counter:counter + L[ct]] = 1.0

            self.B_mask[counter_B, counter:counter + L[ct], counter:counter + L[ct]] = 1.0
            self.factor_to_celltype[counter:counter + L[ct], counter_B] = 1.0
            counter = counter + L[ct]
            counter_B = counter_B + 1

        # check that these masks are correct
        self.cell_type_counts = []
        for cell_type in ct_order:
            if cell_type == "global":
                self.cell_type_counts.append(self.n)
            else:
                n_c = sum(labels == cell_type)
                # mimic behavior of old version: Tues Feb 21, 2023 <-- if there are no annotations we'll just set the whole loss to 0 by setting cell type counts to 0
                if len(adj_matrix[cell_type]) > 0:
                    self.cell_type_counts.append(n_c)
                else:
                    self.cell_type_counts.append(0)
        self.cell_type_counts = np.array(self.cell_type_counts)
        # self.ct_vec =  torch.Tensor(self.cell_type_counts/float(self.n)).to(self.device)
        self.register_buffer("ct_vec", torch.Tensor(self.cell_type_counts / float(self.n)))

    def initialize(self,annotations, word2id, W, init_scores, val = 25):
        """
        self.use_cell_types must be True
        create form of gene_sets:
        
        cell_type (inc. global) : set of sets of idxs
        
        filter based on L_ct
        """
        if self.use_cell_types:
            if init_scores == None:
                init_scores = compute_init_scores(annotations, word2id, torch.Tensor(W)) 
            gs_dict = OrderedDict()
            for ct in annotations.keys():
                mval = max(self.L[ct] - 1, 0) 
                sorted_init_scores = sorted(init_scores[ct].items(), key=lambda x:x[1])
                sorted_init_scores = sorted_init_scores[-1*mval:]
                names = set([k[0] for k in sorted_init_scores])  
                lst_ct = []
                for key in annotations[ct].keys():
                    if key in names:
                        words = annotations[ct][key]
                        idxs = []
                        for word in words:
                            if word in word2id:
                                idxs.append(word2id[word])
                        lst_ct.append(idxs)
                gs_dict[ct] = lst_ct
            self.set_factor_weights(gene_sets = gs_dict, val = val)
        else:
            if init_scores == None:
                init_scores = compute_init_scores_noct(annotations,word2id,torch.Tensor(W))
            lst = []
            mval = max(self.L - 1, 0)
            sorted_init_scores = sorted(init_scores.items(), key = lambda x:x[1])
            sorted_init_scores = sorted_init_scores[-1*mval:]
            names = set([k[0] for k in sorted_init_scores])   
            for key in annotations.keys():
                if key in names:
                    words = annotations[key]
                    idxs = []
                    for word in words:
                        if word in word2id:
                            idxs.append(word2id[word])
                    lst.append(idxs)
            self.set_factor_weights_no_celltypes(gs_list = lst, val = val)

    def set_factor_weights(self, gene_sets, val):
        """
        form of gene_sets:

        cell_type (inc. global) : set of sets of idxs
        """

        for i, ct in enumerate(self.ct_order):
            assert (self.L[ct] >= len(gene_sets[ct]))
            count = self.start_pos[ct]
            if self.L[ct] > 0:
                if self.cell_type_counts[i] > 0:
                    for gene_set in gene_sets[ct]:
                        self.theta.data[:, count][gene_set] = val
                        count = count + 1
                    for j in range(self.L[ct]):
                        self.eta.data[self.start_pos[ct] + j, self.start_pos[ct] + self.L[ct] - 1] = -val
                        self.eta.data[self.start_pos[ct] + self.L[ct] - 1, self.start_pos[ct] + j] = -val
                    self.theta.data[:, self.start_pos[ct] + self.L[ct] - 1][self.adj_matrix[i].sum(axis=1) == 0] = val
                    self.theta.data[:, self.start_pos[ct] + self.L[ct] - 1][self.adj_matrix[i].sum(axis=1) != 0] = -val


class SPECTRA_DataModule(pl.LightningDataModule):
    """
        defines the DataModule to be used with the lightning trainer

        Parameters:
            X : np.ndarray or torch.Tensor
            the ``(n, p)`` -shaped matrix containing logged expression count data. Used for initialization of self.n and self.p but not stored as an attribute

            alpha_mask : alpha as defined in SPECTRA but the mask also computed and used on the object

            device : the device currently in use by the Lightning Module

            Ideally would also include batch size
    """
    def __init__(self, X, alpha_mask, device):    
        super().__init__()
        self.X = X
        self.alpha_mask = alpha_mask
        self.batch_size = 1000
        self.device = device

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = TensorDataset(torch.Tensor(self.alpha_mask).to(self.device), torch.Tensor(self.X).to(self.device))
            self.train = dataset 
            self.validate = dataset 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = TensorDataset(torch.Tensor(self.alpha_mask).to(self.device), torch.Tensor(self.X).to(self.device))  

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validate, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size)



class SPECTRA_LitModel(pl.LightningModule):
    """
    Parameters
        ----------
        X : np.ndarray or torch.Tensor
            the ``(n, p)`` -shaped matrix containing logged expression count data. Used for initialization of self.n and self.p but not stored as an attribute
        labels : np.ndarray or NoneType
            the ``(n, )`` -shaped array containing cell type labels. If use_cell_types == False, then should be set to None

        L : dict or OrderedDict [if use_cell_types == False, then int]
            ``number of cell types + 1``-shaped dictionary. Must have "global" as a key, indicating the number of global factors
            {
                "global": 15,
                "CD8": 5
                ...
            }
            > Format matches output of K_est.py to estimate the number of
            > Must match cell type labels provided during training
            > Recommended practice is to assign at minimum 2 factors per cell type
            > Note that L contains the number of factors that describe the graph.
        adj_matrix :  dict or OrderedDict
            ``a dictionary of adjacency matrices, one for every cell type + a "global"
            {
                "global": ``(p, p)``-shaped binary np.ndarray
                "CD8": ...
            }
        weights : dict or OrderedDict or NoneType [if use_cell_types == False, then ``(p, p)``-shaped array]
            the ``(p, p)``-shaped set of edge weights per . If weight[i,j] is non-zero when adj_matrix[i,j] = 0
            this weight is ignored.

            if weights == None, no weights are used
        lam : float
            lambda parameter of the model, which controls the relative influence of the graph vs expression loss functions. This term multiplies the expression loss, so smaller values of lambda upweight the prior information
        delta : float
            delta parameter of the model, which controls a lower bound for gene scaling factors. If delta is small then the maximum ratio between gene scaling factors is larger and lowly expressed genes can be put on the same scale as highly expressed genes.
        kappa : float or NoneType
            kappa controls the background rate of edges in the graph. if kappa is a float, kappa is fixed to the given float value. If kappa == None, then kappa is a parameter that is estimated from the data.
        rho : float or NoneType
            rho controls the bakcground rate of non-edges in the graph. if rho is a float, rho is fixed to the given float value. If rho == None, then rho is a parameter that is estimated from the data.
        use_cell_types: bool
            use_cell_types is a Boolean variable that determines whether cell type labels are used to fit the model. If False, then parameters are initialized as nn.Parameter rather than as nn.ParameterDict with cell type keys that index nn.Parameter values


    Attributes:

    internal_model : the model the trainer trains on

    cell_scores
    factors
    B_diag
    eta_matrices
    gene_scalings
    rho
    kappa

    Methods:

    training_step : the lighting hook to define what happens for each epoch in each batch of the data

    backward : hook for the backward step of the model, so that it retains the graph as it trains

    configure_optimizers : lightning hook to configure the optimizer and the lr_scheduler

    compute_loss : custom hook for computing the loss at each training step

    model.return_selection()
    model.return_eta_diag()
    model.return_cell_scores()
    model.return_factors() 
    model.return_eta()
    model.return_rho() 
    model.return_kappa()
    model.return_gene_scalings()
    model.return_graph(ct = "global") : 
    model.matching(markers, gene_names_dict, threshold = 0.4):


    """
    def __init__(self,X, labels, L, vocab = None, gs_dict = None, use_weights = False, adj_matrix = None, weights = None, lam = 0.1, delta=0.1,kappa = None, rho = None, use_cell_types = True):
        super().__init__()
        
        self.internal_model = SPECTRA(X = X, labels = labels,  device = self.device, L = L, vocab = vocab, gs_dict = gs_dict, use_weights = use_weights, lam = lam, delta=delta,kappa = kappa, rho = rho, use_cell_types = use_cell_types)
        
        self.register_buffer("cell_scores", None)
        self.register_buffer("factors", None)
        self.register_buffer("B_diag", None)
        self.register_buffer("eta_matrices", None)
        self.register_buffer("gene_scalings", None)
        self.register_buffer("rho", None)
        self.register_buffer("kappa", None)

    def on_fit_start(self):
        self.internal_model.to(self.device)


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        X_b, alpha_b = batch
        loss = self.compute_loss(X_b, alpha_b).to(self.device)
        return loss


    def backward(self, loss):
        loss.backward(retain_graph=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.internal_model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
        return [optimizer], [lr_scheduler]
    
    def compute_loss(self, X, alpha):
        assert(self.internal_model.use_cell_types) #if this is False, fail because model has not been initialized to use cell types
        batch_size = 1000 # can be changed here, like num_epochs
        
        # create the weird softmax theta 
        theta = torch.cat([temp.softmax(dim = 1) for temp in torch.split(self.internal_model.theta, split_size_or_sections = self.internal_model.L_list, dim = 1)], dim = 1)
        #initialize loss and fetch global parameters
        eta = self.internal_model.eta.exp()/(1.0 + (self.internal_model.eta).exp())
        eta = 0.5*(eta + eta.T)
        gene_scaling = self.internal_model.gene_scalings.exp()/(1.0 + self.internal_model.gene_scalings.exp()) #ctp1 x p)
        kappa = self.internal_model.kappa.to(self.device).exp()/(1 + self.internal_model.kappa.to(self.device).exp()) #ctp1
        rho = self.internal_model.rho.to(self.device).exp()/(1 + self.internal_model.rho.to(self.device).exp()) #ctp1
        #handle theta x gene_scalings 
        
        gene_scaling_ = contract('ij,ki->jk',gene_scaling, self.internal_model.factor_to_celltype.to(self.device)) #p x L_tot
        theta_ = theta * (gene_scaling_ + self.internal_model.delta)  #p x L_tot
        alpha_ = torch.exp(alpha) # should get the same thing as original gpu implementation, which is batch x L_tot
        recon = contract('ik,jk->ij' , alpha_, theta_)
        term1 = -1.0*(torch.xlogy(X,recon) - recon).sum()
        
        eta_ = eta[None,:,:]*self.internal_model.B_mask.to(self.device) #ctp1 x L_tot x L_tot 
        
        mat = contract('il,clj,kj->cik',theta_,eta_,theta_) #ctp1 x p x p
        
        term2 = -1.0*((torch.xlogy(self.internal_model.adj_matrix.to(self.device)*self.internal_model.weights.to(self.device), (1.0 - rho.reshape(-1,1,1))*(1.0 -kappa.reshape(-1,1,1))*mat + (1.0 - rho.reshape(-1,1,1))*kappa.reshape(-1,1,1)))*self.internal_model.ct_vec.to(self.device).reshape(-1,1,1)).sum()
        term3 = -1.0*((torch.xlogy(self.internal_model.adj_matrix_1m.to(self.device),(1.0 -kappa.reshape(-1,1,1))*(1.0 - rho.reshape(-1,1,1))*(1.0 - mat) + rho.reshape(-1,1,1)))*self.internal_model.ct_vec.to(self.device).reshape(-1,1,1)).sum()
        loss = (self.internal_model.n/batch_size)*self.internal_model.lam*term1 + term2 + term3 #upweight local param terms to be correct on expectation 
        return loss
    
    
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
        if model.use_cell_types:
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
    """
    defines a custom hook to store parameters cell_scores, factors, gene_scalings, rho, and kappa
    once the training step is complete

    takes in the trainer object, and the Lightning module object (SPECTRA_LitModel)
    """
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
        pl_module.B_diag = torch.diag(Bg).detach().cpu().numpy()


        # calculate eta matrix
        eta = OrderedDict()
        Bg = model.eta.exp()/(1.0 + model.eta.exp())
        Bg = 0.5*(Bg + Bg.T)

        for ct in model.ct_order:
            eta[ct] = Bg[model.start_pos[ct]: model.start_pos[ct] +model.L[ct], model.start_pos[ct]: model.start_pos[ct] +model.L[ct]].detach().cpu().numpy()
        pl_module.eta_matrices = eta

        
        #new store params stuff 
        pl_module.cell_scores = cell_scores
        pl_module.factors = new_factors
        pl_module.gene_scalings = {ct : gene_scaling[i].detach().cpu().numpy() for i, ct in enumerate(model.ct_order)}
        pl_module.rho = {ct: model.rho[i].exp().detach().cpu().numpy()/(1.0 + model.rho[i].exp().detach().cpu().numpy()) for i, ct in enumerate(model.ct_order)}
        pl_module.kappa = {ct: model.kappa[i].exp().detach().cpu().numpy()/(1.0 + model.kappa[i].exp().detach().cpu().numpy()) for i, ct in enumerate(model.ct_order)}






def est_spectra(adata, gene_set_dictionary, L=None, use_highly_variable=True, cell_type_key=None, use_weights=True,
                lam=0.008, delta=0.001, kappa=None, rho=0.05, use_cell_types=True, n_top_vals=50, filter_sets=True,
                **kwargs):
    """

    Parameters
        ----------
        adata : AnnData object
            containing cell_type_key with log count data stored in .X
        gene_set_dictionary : dict or OrderedDict()
            maps cell types to gene set names to gene sets ; if use_cell_types == False then maps gene set names to gene sets ;
            must contain "global" key in addition to every unique cell type under .obs.<cell_type_key>
        L : dict, OrderedDict(), int , NoneType
            number of factors per cell type ; if use_cell_types == False then int. Else dictionary. If None then match factors
            to number of gene sets (recommended)
        use_highly_variable : bool
            if True, then uses highly_variable_genes
        cell_type_key : str
            cell type key, must be under adata.obs.<cell_type_key> . If use_cell_types == False, this is ignored
        use_weights : bool
            if True, edge weights are estimated based on graph structure and used throughout training
        lam : float
            lambda parameter of the model. weighs relative contribution of graph and expression loss functions
        delta : float
            delta parameter of the model. lower bounds possible gene scaling factors so that maximum ratio of gene scalings
            cannot be too large
        kappa : float or None
            if None, estimate background rate of 1s in the graph from data
        rho : float or None
            if None, estimate background rate of 0s in the graph from data
        use_cell_types : bool
            if True then cell type label is used to fit cell type specific factors. If false then cell types are ignored
        n_top_vals : int
            number of top markers to return in markers dataframe
        determinant_penalty : float
            determinant penalty of the attention mechanism. If set higher than 0 then sparse solutions of the attention weights
            and diverse attention weights are encouraged. However, tuning is crucial as setting too high reduces the selection
            accuracy because convergence to a hard selection occurs early during training [todo: annealing strategy]
        filter_sets : bool
            whether to filter the gene sets based on coherence
        **kwargs : (num_epochs = 10000, lr_schedule = [...], verbose = False)
            arguments to .train(), maximum number of training epochs, learning rate schedule and whether to print changes in
            learning rate

     Returns: SPECTRA_LitModel object [after training]

     In place: adds 1. factors, 2. cell scores, 3. vocabulary, and 4. markers as attributes in .obsm, .var, .uns

    """

    if use_cell_types == False:
        raise NotImplementedError("use_cell_types == False is not supported yet")


    if L == None:
        init_flag = True
        if use_cell_types:
            L = {}
            for key in gene_set_dictionary.keys():
                length = len(list(gene_set_dictionary[key].values()))
                L[key] = length + 1
        else:
            length = len(list(gene_set_dictionary.values()))
            L = length + 1
    # create vocab list from gene_set_dictionary
    lst = []
    if use_cell_types:
        for key in gene_set_dictionary:
            for key2 in gene_set_dictionary[key]:
                gene_list = gene_set_dictionary[key][key2]
                lst += gene_list
    else:
        for key in gene_set_dictionary:
            gene_list = gene_set_dictionary[key]
            lst += gene_list

    # lst contains all of the genes that are in the gene sets --> convert to boolean array
    bools = []
    for gene in adata.var_names:
        if gene in lst:
            bools.append(True)
        else:
            bools.append(False)
    bools = np.array(bools)

    if use_highly_variable:
        idx_to_use = bools | adata.var.highly_variable  # take intersection of highly variable and gene set genes (todo: add option to change this at some point)
        X = adata.X[:, idx_to_use]
        vocab = adata.var_names[idx_to_use]
        adata.var["spectra_vocab"] = idx_to_use
    else:
        X = adata.X
        vocab = adata.var_names

    if use_cell_types:
        labels = adata.obs[cell_type_key].values
        for label in np.unique(labels):
            if label not in gene_set_dictionary:
                gene_set_dictionary[label] = {}
            if label not in L:
                L[label] = 1
    else:
        labels = None
    if type(X) == scipy.sparse.csr.csr_matrix:
        X = np.array(X.todense())
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    if filter_sets:
        if use_cell_types:
            new_gs_dict = {}
            init_scores = compute_init_scores(gene_set_dictionary, word2id, torch.Tensor(X))
            for ct in gene_set_dictionary.keys():
                new_gs_dict[ct] = {}
                mval = max(L[ct] - 1, 0)
                sorted_init_scores = sorted(init_scores[ct].items(), key=lambda x: x[1])
                sorted_init_scores = sorted_init_scores[-1 * mval:]
                names = set([k[0] for k in sorted_init_scores])
                for key in gene_set_dictionary[ct].keys():
                    if key in names:
                        new_gs_dict[ct][key] = gene_set_dictionary[ct][key]
        else:
            init_scores = compute_init_scores_noct(gene_set_dictionary, word2id, torch.Tensor(X))
            new_gs_dict = {}
            mval = max(L - 1, 0)
            sorted_init_scores = sorted(init_scores.items(), key=lambda x: x[1])
            sorted_init_scores = sorted_init_scores[-1 * mval:]
            names = set([k[0] for k in sorted_init_scores])
            for key in gene_set_dictionary.keys():
                if key in names:
                    new_gs_dict[key] = gene_set_dictionary[key]
        gene_set_dictionary = new_gs_dict
    else:
        init_scores = None
    print("Initializing model...")
    spectra_lit = SPECTRA_LitModel(X = X, labels = labels,  L = L, vocab = vocab, gs_dict = gene_set_dictionary, use_weights = use_weights, lam = lam, delta=delta,kappa = kappa, rho = rho, use_cell_types = use_cell_types)
    
    spectra_lit.internal_model.initialize(gene_set_dictionary, word2id, X, init_scores)
    spectra_lit.internal_model.to(spectra_lit.device)
    print("initialized internal model")
    
    spectra_dm = SPECTRA_DataModule(spectra_lit.internal_model.alpha * spectra_lit.internal_model.alpha_mask, X, spectra_lit.device)
    print("created dataModule")
    print("Beginning training...")
    trainer = pl.Trainer(max_epochs = 1000, callbacks = [SPECTRA_Callback()])
    trainer.fit(model = spectra_lit, train_dataloaders = spectra_dm) 

    adata.uns["SPECTRA_factors"] = spectra_lit.factors
    adata.obsm["SPECTRA_cell_scores"] = spectra_lit.cell_scores
    adata.uns["SPECTRA_markers"] = return_markers(factor_matrix=spectra_lit.factors, id2word=id2word, n_top_vals=n_top_vals)
    adata.uns["SPECTRA_L"] = L
    return spectra_lit


def return_markers(factor_matrix, id2word, n_top_vals=100):
    idx_matrix = np.argsort(factor_matrix, axis=1)[:, ::-1][:, :n_top_vals]
    df = pd.DataFrame(np.zeros(idx_matrix.shape))
    for i in range(idx_matrix.shape[0]):
        for j in range(idx_matrix.shape[1]):
            df.iloc[i, j] = id2word[idx_matrix[i, j]]
    return df.values


def graph_network(adata, mat, gene_set, thres=0.20, N=50):
    vocab = adata.var_names[adata.var["spectra_vocab"]]
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    net = Network(height='750px', width='100%', bgcolor='#FFFFFF', font_color='black', notebook=True)
    net.barnes_hut()

    idxs = []
    for term in gene_set:
        idxs.append(word2id[term])
    ests = list(set(list(mat[idxs, :].sum(axis=0).argsort()[::-1][:N]) + idxs))
    ests_names = []
    count = 0
    for est in ests:
        ests_names.append(id2word[est])
        if est not in idxs:
            net.add_node(count, label=id2word[est], color='#00ff1e')
        else:
            net.add_node(count, label=id2word[est], color='#162347')
        count += 1

    inferred_mat = mat[ests, :][:, ests]
    for i in range(len(inferred_mat)):
        for j in range(i + 1, len(inferred_mat)):
            if inferred_mat[i, j] > thres:
                net.add_edge(i, j)
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node['value'] = len(neighbor_map[node['id']])

    return net


def graph_network_multiple(adata, mat, gene_sets, thres=0.20, N=50):
    gene_set = []
    for gs in gene_sets:
        gene_set += gs

    vocab = adata.var_names[adata.var["spectra_vocab"]]
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    net = Network(height='750px', width='100%', bgcolor='#FFFFFF', font_color='black', notebook=True)
    net.barnes_hut()
    idxs = []
    for term in gene_set:
        idxs.append(word2id[term])
    ests = list(set(list(mat[idxs, :].sum(axis=0).argsort()[::-1][:N]) + idxs))
    ests_names = []
    count = 0

    color_map = []
    for gene_set in gene_sets:
        random_color = ["#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        color_map.append(random_color[0])

    for est in ests:
        ests_names.append(id2word[est])
        if est not in idxs:
            net.add_node(count, label=id2word[est], color='#00ff1e')
        else:
            for i in range(len(gene_sets)):
                if id2word[est] in gene_sets[i]:
                    color = color_map[i]
                    break
            net.add_node(count, label=id2word[est], color=color)
        count += 1

    inferred_mat = mat[ests, :][:, ests]
    for i in range(len(inferred_mat)):
        for j in range(i + 1, len(inferred_mat)):
            if inferred_mat[i, j] > thres:
                net.add_edge(i, j)
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node['value'] = len(neighbor_map[node['id']])

    return net


def gene_set_graph(gene_sets):
    """
    input
    [
    ["a","b", ... ],
    ["b", "d"],

    ...
    ]
    """

    net = Network(height='750px', width='100%', bgcolor='#FFFFFF', font_color='black', notebook=True)
    net.barnes_hut()
    count = 0
    # create nodes
    genes = []
    for gene_set in gene_sets:
        genes += gene_set

    color_map = []
    for gene_set in gene_sets:
        random_color = ["#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        color_map.append(random_color[0])

    for gene in genes:
        for i in range(len(gene_sets)):
            if gene in gene_sets[i]:
                color = color_map[i]
                break
        net.add_node(gene, label=gene, color=color)

    for gene_set in gene_sets:
        for i in range(len(gene_set)):
            for j in range(i + 1, len(gene_set)):
                net.add_edge(gene_set[i], gene_set[j])

    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node['value'] = len(neighbor_map[node['id']])

    return net
