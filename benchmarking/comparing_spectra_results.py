import scanpy as sc
import csv
from scipy.optimize import linear_sum_assignment

# going to start with just comparing two runs
factors_1 = []
factors_2 = []

# reading in factors from csv

with open('factors_0.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # get list of factors from csv
    factors_1 = list(csv_reader)

    csv_file.close()

with open('factors_1.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    factors_2 = list(csv_reader)

    csv_file.close()

    
row_ind_1, col_ind_1 = linear_sum_assignment(factors_1)
row_ind_2, col_ind_2 = linear_sum_assignment(factors_2)


print(factors_1[0][col_ind_1[0]])
print(factors_2[0][col_ind_2[0]])

# method that roshan gave
# my guess is that we want the sums to be similar - going to test later
# with the runs that we've already completed
# sp.optimize.linear_sum_assignment()