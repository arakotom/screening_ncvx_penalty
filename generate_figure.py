#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 22:39:50 2017

@author: alain
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from utils import get_results
import seaborn as sns

color_pal = sns.color_palette("colorblind", 10).as_hex()    
plt.close("all")
pathfig = './'

dofig = 2
if dofig ==1 :
    sigma_bruit = 2
    n_samplesvec =  [500] 
    n_featuresvec = [5000] 
    n_informative = 5
    n_iter = 20
    tol_vec =  [1e-3,1e-4,1e-5]
    path_results = './'
elif dofig==2:
    sigma_bruit = 2
    n_samplesvec =  [50] 
    n_featuresvec = [100] 
    n_informative = 5
    n_iter = 1
    tol_vec =  [1e-3,1e-4,1e-5]

    path_results = './'



    
type_of_fig = 'reg_path'
N = 50 
dual_gap_inner = 1e-4
screen_every = 10


algo_list = ['bcd','gist','mm_screening_genuine','mm_screening_2']
method_name = ['ncxCD', 'GIST','MM genuine', 'MM screening']

for n_samples,n_features in zip(n_samplesvec,n_featuresvec):
    C_all = np.zeros((len(algo_list),len(tol_vec)))
    M_all = np.zeros((len(algo_list),len(tol_vec)))
    S_all = np.zeros((len(algo_list),len(tol_vec)))
    V_all = np.zeros((len(algo_list),len(tol_vec), n_iter))

    F_meas_vec=[]
    for i, tol in enumerate(tol_vec):
       print('------------------',n_samples,n_features,tol, sigma_bruit,'---------------')
       nb_fmeas = 500
       for j, method in enumerate(algo_list):     
           try :

               best_perf, best_mod, all_timing, percent_opt,std_perf,std_timing, best_cost,std_cost,f_meas = get_results(path_results,method,n_samples,n_features,n_informative,sigma_bruit,tol,N,dual_gap_inner,screen_every)
               F_meas_vec.append(f_meas)
               if type_of_fig == 'reg_path':
                   M_all[j,i] = all_timing
                   S_all[j,i] = std_timing
                   comments = 'Regularization Path'
                   ylabel = 'Percentage of time of ncxCD'


               print(type_of_fig, method,best_perf, best_mod, all_timing,percent_opt)
               print('\n')
           except:
               pass
        
    N_method = len(algo_list)
    if type_of_fig == 'reg_path':
        ref = M_all[0].copy() 
        M_all/=M_all[0]
        M_all *=100
        S_all/= ref
        S_all *=100
        ylim = [0, (np.amax(M_all)*1.1)]
           

    ind = np.arange(len(tol_vec))  # the x locations for the groups
    width = 0.15       # the width of the bars
    list_color = ['#0173b2','#ece133','#029e73','#d55e00','#cc78bc','#ca9161']

    fig, ax = plt.subplots()
    rects = []
    titlefile = '{} - n={:2d} d={:3d} p={:d} $\sigma$={:2.2f}'.format(comments,n_samples,n_features, n_informative,sigma_bruit)
    for i in range(len(algo_list)):
        aux = ax.bar(ind+0.225 + i*width, M_all[i],width, yerr=S_all[i], color=list_color[i])
        rects.append(aux)
        
    ax.grid(True)    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Tolerance')
    ax.set_title(titlefile)
    ax.set_xticks(ind + len(tol_vec)*width)
    ax.set_xticklabels(['{:2.2e}'.format(t) for t in tol_vec])
    ax.set_ylim(ylim)
    ax.legend(rects, method_name,loc='upper left')
    plt.savefig(pathfig + titlefile.replace(" ","").replace("=","-").replace("$","").replace(".","-").replace("\\","") + '.pdf',dpi=300)


