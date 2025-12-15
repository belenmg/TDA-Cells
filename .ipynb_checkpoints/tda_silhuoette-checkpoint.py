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

def silhuoette_point(H,x):

    sh = 0
    w = np.sum((H[:,1] - H[:,0])**2)

    for interval in H:
        b, d = interval
        h = (d-b)/2
        c = (b+d)/2    
        w_bd = (d-b)**2    
        tent_bd = max([0,h-abs(x-c)])
        sh += math.sqrt(2)*w_bd/w*tent_bd

    return sh

##################################################################################################

def silhuoette(res, Hs): 

    B = min(np.min(H[:, 0]) for H in Hs if len(H)>0)
    D = max(np.max(H[:, 1]) for H in Hs if len(H)>0)
    xs = np.linspace(B,D,res) 

    shs = []

    for H in Hs: 
    
        if len(H) > 0:
            sh = np.array([silhuoette_point(H,x) for x in xs])
            shs.append(sh)
        else:
            shs.append(0)

    return xs, shs

##################################################################################################

def silhuoette_graphics(xs, shs, colores):

    for t in range(len(shs)):
        if isinstance(shs[t], np.ndarray):    
            plt.plot(xs, shs[t], label = f"t = {t+1}", color = colores[t])
            plt.grid(True)
            plt.legend()
            
    #plt.show()
    
##################################################################################################

def silhuoette_diff(xs, shs, norm):

    D = []
    d_txt = []

    for t in range(len(shs)-1):
        if isinstance(shs[t], np.ndarray):
            if norm == '2':
                d = np.linalg.norm(shs[t] - shs[t+1])
                D.append(d)
            elif norm == '1':
                d = np.linalg.norm(shs[t] - shs[t+1],1)
                D.append(d)
            elif norm == 'L1':
                L = np.trapz(np.abs(shs[t] - shs[t+1]), xs)
                D.append(L)        
            s = f"{t+1},{t+2}"
            d_txt.append(s)

    return D, d_txt

##################################################################################################
# FORMA 2: 
##################################################################################################

def silhuoette2(H:np.array, resolution = 1000):
    
    SH = gd.representations.Silhouette(resolution = resolution, weight = lambda x: np.power(x[1]-x[0],2))    
    sh = SH.fit_transform([H])

    return sh

def silhuoette_graphics2(sh, color = None):

    plt.plot(sh[0], color = color)
    plt.grid(True)

def silhuoette_diff2(shs, norm):

    D = []
    d_txt = []

    for t in range(len(shs)-1):
        if isinstance(shs[t], np.ndarray):
            if norm == '2':
                d = np.linalg.norm(shs[t] - shs[t+1])
                D.append(d)
            elif norm == '1':
                d = np.linalg.norm(shs[t] - shs[t+1],1)
                D.append(d)     
            s = f"{t+1},{t+2}"
            d_txt.append(s)

    return D, d_txt

