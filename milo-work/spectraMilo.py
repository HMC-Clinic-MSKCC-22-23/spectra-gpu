#import packages
import numpy as np
import scanpy as sc
import pandas as pd
import plotnine as pn
import statistics as st
import math
pn.options.figure_size = (6,4)

def sort_factors(X_norm_log: np.ndarray, cell_score_matrix: np.ndarray, gene_loading_matrix: np.ndarray, plot_path : str, method : str = "explained_variance"):

    n_cells, _ = cell_score_matrix.shape

    n_factors, n_genes = gene_loading_matrix.shape

    # defined as variance of each selected gene divided by total variance
    if method == "explained_variance":

        # first, select top genes for each factor
        top_genes = select_top_genes(gene_loading_matrix)

        # calculate variance of each gene in cell x gene matrix
        gene_variance = [X_norm_log[:,x].var() for x in range(n_genes)]
        total_variance = sum(gene_variance)

        # divide sum of genes' variance by total to get explained_variance for each factor
        explained_variance = []

        for curr_factor in range(n_factors):
            explained_variance.append( sum([gene_variance[curr_gene] for curr_gene in top_genes[curr_factor]]) / total_variance)

        # if a plot path is provided, plot the factors' explained variance
        if plot_path:
            plot = (pn.ggplot()
                    + pn.aes(x = list(range(n_factors)), y = explained_variance)
                    + pn.xlab("Factor")
                    + pn.ylab("Explained Variance")
                    + pn.geom_point()
                    + pn.theme_bw())
            
            plot.save(f"{plot_path}/factors_explained_variance.pdf", dpi = 250, verbose = False)

        # sort by explained variance, return indices of sorted factors by explained variance
        sort_index = [factor for factor, _ in sorted(enumerate(explained_variance), key = lambda x: x[1], reverse = True)]


    # defined as sum of highly_variable_genes(cell_ranger) dispersions of selected genes
    elif method == "gene_dispersions":

        # first, select top genes for each factor
        top_genes = select_top_genes(gene_loading_matrix)

        # perform highly_variable_genes
        # the higher the 'dispersion' value, the more variation (according to "highly_variable" tag, backed up by variance)
        dummy_adata = sc.AnnData(X_norm_log)
        sc.pp.highly_variable_genes(dummy_adata, flavor = 'cell_ranger')

        # sum all selected genes for each factor
        sum_disp = []

        for curr_factor in range(n_factors):
            sum_disp.append(sum([dummy_adata.var["dispersions_norm"][curr_gene] for curr_gene in top_genes[curr_factor]]))

        # if a plot path is provided, plot the factors' explained variance
        if plot_path:
            plot = (pn.ggplot()
                    + pn.aes(x = list(range(n_factors)), y = sum_disp)
                    + pn.xlab("Factor")
                    + pn.ylab("Sum of Normalized Gene Dispersions")
                    + pn.geom_point()
                    + pn.theme_bw())
            
            plot.save(f"{plot_path}/factors_gene_dispersions.pdf", dpi = 250, verbose = False)

        # sort factors by their sum_disp
        sort_index = [factor for factor, _ in sorted(enumerate(sum_disp), key = lambda x: x[1], reverse = True)]


    # defined as highly_variable_genes(cell_ranger) dispersions of factors
    elif method == "factor_dispersions":

        # perform highly_variable_genes
        # the higher the 'dispersion' value, the more variation
        dummy_adata = sc.AnnData(cell_score_matrix)
        sc.pp.highly_variable_genes(dummy_adata, flavor = 'cell_ranger')

        factor_dispersions = dummy_adata.var["dispersions_norm"].values 

        # if a plot path is provided, plot the factors' explained variance
        if plot_path:
            plot = (pn.ggplot()
                    + pn.aes(x = list(range(n_factors)), y = factor_dispersions)
                    + pn.xlab("Factor")
                    + pn.ylab("Normalized Dispersion")
                    + pn.geom_point()
                    + pn.theme_bw())
            
            plot.save(f"{plot_path}/factors_factor_dispersions.pdf", dpi = 250, verbose = False)

        # sort factors by their sum_disp
        sort_index = [factor for factor, _ in sorted(enumerate(factor_dispersions), key = lambda x: x[1], reverse = True)]

    else:
        raise ValueError("Invalid method - select 'explained_variance', 'gene_dispersions', or 'factor_dispersions'")

    return sort_index


def select_top_genes(gene_loading_matrix: np.ndarray):

    n_factors = len(gene_loading_matrix)
    top_genes = []

    for curr_factor in range(n_factors):

        curr_loadings = gene_loading_matrix[curr_factor]

        curr_cutoff = st.mean(curr_loadings) + 2 * st.stdev(curr_loadings)

        curr_top_genes = [idx for idx, value in enumerate(curr_loadings) if value > curr_cutoff]

        top_genes.append(curr_top_genes)
    
    return top_genes


def run_milo(adata, sorted_factors : list[int], n_factors : int, plot_path : str):
    return

if __name__ == "__main__":

    # adata = sc.read_h5ad("../../new_annData.h5ad")

    adata = sc.read_h5ad("../../clean_myeloid.h5ad")

    cell_scores = pd.read_csv("../../milo_cell_scores.csv", index_col = False, header = None).to_numpy()

    gene_loadings = pd.read_csv("../../milo_gene_loadings.csv", index_col = False, header = None).to_numpy()

    # factor_list = sort_factors(adata.X.toarray(), adata.obsm["SPECTRA_cell_scores"], adata.uns["SPECTRA_factors"], ".")
    
    factor_list = sort_factors(adata.layers["norm_log"], cell_scores, gene_loadings, ".", "factor_dispersions")
    