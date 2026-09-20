import numpy as np
import math
import random
import pandas as pd
from numpy import linalg as LA
def wbo(fun, dim, lb, ub, Max_iter, pop=30):
    lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
    ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
    X = np.random.uniform(lb, ub, (pop, dim))                  
    fitness = np.array([fun(p) for p in X])
    sorted_idx = np.argsort(fitness)       
    Curve = np.zeros([Max_iter, 1])
    gbest  = X[sorted_idx[0],:].copy()
    fit_best = fitness[sorted_idx[0]].copy()
    c1 = 1
    # Main iteration starts here
    for t in range(Max_iter):
        if t < Max_iter/2:
            c2 = 1
        else:
            c2 = 1
        for i in range(pop):
            current_X = X[i,:].copy()
            Z1 = gbest.copy()
            Z2 = X[i,:].copy()
            E = c1 * (2 *np.random.random()-1) * (1 - t/Max_iter) 
            rl = c2 * levy1(dim)
            similarity = cosine_similarity(Z1,Z2)
            if similarity >0:
                Z = (Z1 + Z2)/2
            else:
                if np.random.random()<0.5:
                    Z = Z1.copy()
                else:
                    Z = Z2.copy()
            if np.random.random()<0.5:
                X[i,:] = Z - E * np.abs(rl * Z - current_X)
            else:
                X[i,:] = Z - E * np.abs(Z - rl * current_X)
                  
            X[i,:] = np.clip(X[i,:], lb, ub)
            fitness[i] = fun(X[i,:])

        sorted_idx = np.argsort(fitness)
        if fitness[sorted_idx[0]] < fit_best:
            gbest = X[sorted_idx[0],:].copy()
            fit_best = fitness[sorted_idx[0]].copy()
        Curve[t] = fit_best.copy()
    
    return fit_best, gbest, Curve
