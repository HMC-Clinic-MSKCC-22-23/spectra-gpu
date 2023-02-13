import time
import tracemalloc
import numpy as np
import pandas as pd
import json 
import scanpy as sc
from spectra import spectra as spc

# user-defined parameters
num_iters = 3
root_dir = "C:/Users/Brian/OneDrive/Documents/2023Spring/Clinic/"
save_dir = root_dir + "benchmarks/"
cell_type_key = "annotation_SPADE_1"



# load gene set
with open(root_dir + "spectra/annotations_2.json", "rb") as file:
    annotations = json.load(file)

# load data
adata = sc.read_h5ad(root_dir + "bassez_data.h5ad")

for cell_type in adata.obs[cell_type_key]:
    if cell_type not in annotations:
        annotations[cell_type] = {}

# delete adata
del adata

# set up benchmarks
benchmarks = pd.DataFrame(columns=['Run', 'Wall Time', 'CPU Time', 'Max Memory'])
benchmarks['iter'] = list(range(num_iters))


for iter in range(num_iters):

    print(f"Starting iteration {iter}")

    # load data
    adata = sc.read_h5ad(root_dir + "bassez_data.h5ad")

    start_wall_time = time.time()
    start_cpu_time = time.process_time()

    tracemalloc.start()

    model = spc.est_spectra(adata = adata, gene_set_dictionary = annotations, use_highly_variable = True,
                            cell_type_key = cell_type_key, use_weights = True, lam = 0.1, delta=0.001,
                            kappa = 0.00001, rho = 0.00001, use_cell_types = True, n_top_vals = 25)

    (_, max_mem) = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_wall_time = time.time() - start_wall_time
    elapsed_cpu_time = time.process_time() - start_cpu_time

    benchmarks.at[iter, 'Wall Time'] = elapsed_wall_time
    benchmarks.at[iter, 'CPU Time'] = elapsed_cpu_time
    benchmarks.at[iter, 'Max Memory'] = max_mem

    pd.DataFrame(adata.obsm["SPECTRA_cell_scores"]).to_csv(save_dir + f"cell_scores_{iter}.csv", header=False, index=False)
    pd.DataFrame(adata.uns["SPECTRA_factors"]).to_csv(save_dir + f"factors_{iter}.csv", header=False, index=False)


benchmarks.to_csv(save_dir + "benchmarks.csv", header=True, index = False)