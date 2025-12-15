import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gudhi as gd
import gudhi.representations
from scipy.spatial import Delaunay
import math
import matplotlib.cm as cm
import itertools
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

##################################################################################################

def matrix_filtration(dataset:np.array, f:str = "A"):
    
    
    #Calculate a filter matrix from a dataset and a Delaunay triangulation.
    #---------------------------------------------------------------------
    #INPUT:
    #---------------------------------------------------------------------
    #dataset : np.ndarray
    #f : {'A', 'B', 'C'}, optional
    #    Types of filtrations:
    #    - 'A': |exp(x) - exp(y)|
    #    - 'B': max(exp(x), exp(y))
    #    - 'C': max_expr - min(exp(x), exp(y))
    #--------------------------------------------------------------------
    #OUTPUT:
    #--------------------------------------------------------------------
    #dist_Matrix : np.ndarray

    
    # Separate the dataset into points and expression
    points = dataset[:,:-1]
    expr = dataset[:,-1]
    maxim_expr = np.max(expr)
    n = len(points)

    # Delaunay triangulation
    tri = Delaunay(points, qhull_options = "QJ")   
    
    # Initialize the distance matrix
    dist_Matrix = np.full((n,n), np.inf)    
        
    # Assign values to the diagonal 
    if f == "A":
        np.fill_diagonal(dist_Matrix, 0)
    elif f == "B":
        np.fill_diagonal(dist_Matrix, expr)
    elif f == "C":
        np.fill_diagonal(dist_Matrix, maxim_expr - expr)
    else:
        raise ValueError("The parameter f must be'A', 'B' o 'C'.")

    # Calculate the filtration values for the edges 
    edges = set()  

    for simplex in tri.simplices:
        for x, y in itertools.combinations(simplex, 2):
            if (x, y) in edges or (y, x) in edges:
                continue                  
            edges.add((x, y))
            if f == "A":
                r = abs(expr[x] - expr[y])
            elif f == "B":
                r = max(expr[x], expr[y])
            elif f == "C":
                r = maxim_expr - min(expr[x], expr[y])
            dist_Matrix[x, y] = dist_Matrix[y, x] = r
            
    return dist_Matrix

##################################################################################################

def filtration(M:np.array):

    max_edge = np.max(M[M < np.inf])
    rips_complex = gd.RipsComplex(distance_matrix = M, max_edge_length = max_edge*10) #Por qué *10?
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)
    
    for i in range(M.shape[0]):
        simplex_tree.assign_filtration([i], M[i,i])

    return simplex_tree

##################################################################################################


def persistence_intervals(st: gd.SimplexTree):

    st.persistence()
    H0 = st.persistence_intervals_in_dimension(0)
    H1 = st.persistence_intervals_in_dimension(1)

    return H0, H1