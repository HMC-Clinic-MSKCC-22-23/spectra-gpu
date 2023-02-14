#import packages
import scanpy as sc
import pandas as pd

#spectra imports 
from spectra import spectra as spc

adata = sc.read_h5ad("../../clean_myeloid.h5ad")

annotations = {
    'global': {},
    'Alveolar Macrophage': {
        'gene-set':['MARCO', 'KRT79', 'KRT19', 'CAR4', 'CHIL3']
    },
    'ARG1 Macrophage': {
        'mouse-1':['CCL17', 'CCR5', 'MAFB', 'CD86', 'ARG1'],
        'mouse-2':['GPNMB', 'SPP1', 'CD68']
    },
    'C1QA Macrophage': {
        'canonical':['CSF1R', 'MRC1', 'APOE', 'C1QA', 'C1QB', 'C1QC', 'CTSB', 'CTSD']
    },
    'CCR7 cDC2': {
        'gene-set':['FSCN1', 'CCR7', 'LY75', 'CCL22', 'CD40', 'BIRC3', 'NFKB2', 'IL12B',
            'MARCKSL1', 'CD274', 'TNFRSF9', 'SEMA7A', 'STAT4']
    },
    'cDC1': {
        'gene-set':['XCR1', 'CLEC9A', 'CADM1', 'BATF3', 'IRF8']
    },
    'cDC2': {
        'gene-set':['ZBTB46', 'CD207', 'IRF4', 'CEBPB', 'LILRB4A', 'ITGAX']
    },
    'CSF3R Monocyte': {
        'neutrophil-like':['S100A8', 'S100A9', 'CSF3R', 'IL1B']
    },
    'Monocyte': {
        'classical':['FN1', 'F13A1', 'VCAN', 'LY6C1', 'LY6C2', 'CCR2'],
        'non-classical':['ITGAL', 'CX3CR1', 'FCGR4', 'CD300E', 'ACE']
    },
    'pDC': {
        'gene-set':['TCF4', 'PLD4', 'BCL11A', 'RUNX2', 'SIGLECH', 'CCR9', 'BST2']
    }
}

model = spc.est_spectra(adata = adata, gene_set_dictionary = annotations, 
                        use_highly_variable = False, cell_type_key = 'Celltype_myeloid',
                        use_weights = True, lam = 0.1, 
                        delta = 0.001,kappa = 0.00001, rho = 0.00001, 
                        use_cell_types = True, #set to False to not use the cell type annotations
                        n_top_vals = 25, 
                        num_epochs = 5000 #for demonstration purposes we will only run 2 epochs, we recommend 10,000 epochs
                       )

pd.DataFrame(adata.obsm["SPECTRA_cell_scores"]).to_csv("cell_scores.csv", header=False, index=False)
pd.DataFrame(adata.uns["SPECTRA_factors"]).to_csv("factors.csv", header=False, index=False)