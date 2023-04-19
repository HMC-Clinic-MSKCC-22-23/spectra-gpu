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

class SPECTRA(nn.Module):
   def __init__(self):
       return 0




class SPECTRA_DataModule(pl.LightningDataModule):
    def __init__(self, X, alpha_mask):    #assuming paramters are computed and accessible
        super().__init__()
        self.X = X
        self.alpha_mask = alpha_mask
        self.batch_size = 1000      #for now 

    def prepare_data(self):
        dataset= TensorDataset(torch.Tensor(self.alpha_mask), torch.Tensor(self.X))  # alpha_mask as feature, X as target ? 
        # loader_tr = DataLoader(dataset,batch_size=self.batch_size, shuffle=False, num_workers=4) #added this in case we need iter things, incomplete 
        self.dataset= dataset


        # batching?? 

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = Dataset(self.dataset, train=True, transform=True) # not sure about transform, train file name
            self.validate = Dataset(self.dataset) # validation file name

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = Dataset(self.dataset)    # test file name

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validate, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size)






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

    # print("CUDA Available: ", torch.cuda.is_available())
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## remove this

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
    spectra_internal = SPECTRA()
    spectra_dm = SPECTRA_DataModule()
    spectra_lit = SPECTRA_LitModel(spectra_internal)
    #SPECTRA  # .to(device) ## remove "to(device)"
    # print("CUDA memory: ", 1e-9*torch.cuda.memory_allocated(device=device))
    print("initialized internal model")
    spectra_lit.initialize(gene_set_dictionary, word2id, X, init_scores)
    print("Beginning training...")
    spectra_lit.train(spectra_dm, callbacks = [SPECTRA_Callback()],**kwargs)  # change to call to lightning trainer? unless we decide to stay with the custom loop

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


def load_from_pickle(fp, adata, gs_dict, cell_type_key):
    spectra_internal = SPECTRA()
    model = SPECTRA_LitModel(spectra_internal)
    model.load(fp, labels=np.array(adata.obs[cell_type_key]))
    return (model)

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
