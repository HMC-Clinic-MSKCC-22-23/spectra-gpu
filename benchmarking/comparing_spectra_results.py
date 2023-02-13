import scanpy as sc
import csv
from scipy.optimize import linear_sum_assignment

factors = []

with open('factors_0.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    factors = list(csv_reader)

    print(factors[0][0])



# method that roshan gave
# my guess is that we want the sums to be similar - going to test later
# with the runs that we've already completed
# sp.optimize.linear_sum_assignment()