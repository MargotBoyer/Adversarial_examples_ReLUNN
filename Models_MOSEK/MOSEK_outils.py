import numpy as np
import mosek


def return_k_j_from_i(i,n):
    """Retourne le numéro et la couche d'un neurone (j,k) à partir de son index
    dans la matrice des z"""
    somme = np.cumsum(n)
    real_i = i - 1
    couche = np.searchsorted(somme, i)
    if i == 0:
      return (-1,-1)
    elif couche == 0:
        return (0,real_i)
    else :
        return (couche, real_i - somme[couche-1])
    

def return_i_from_k_j__variable_z(k,j,n):
    """ Retourne l'index correspondant à la variable z du neurone j de la couche k
    dans la matrice des z (peut contenir des sigmas plus loin) """
    i = sum(n[:k]) + j + 1
    return i

def return_i_from_k_j__variable_sigma(k,j,n):
    """ Retourne l'index correspondant à la variable sigma du neurone j de la couche k
    dans la matrice des z et sigma.
    """
    i = 1 + sum(n) + sum(n[1:k]) + j 
    return i

def return_good_j_beta(j,ytrue):
    """Retourne l'index correspondant d'un beta dans la matrice des betas 
    sachant que la matrice ne contient pas Beta_ytrue.
    """
    if j<ytrue : 
        return j + 1
    else : 
        return j

def reconstitue_matrice(taille, tab_triang):
    """ Reconstitue la matrice symétrique à
      partir des valeurs de la partie inférieures notées dans un tableau
      à une dimension

    Args:
        taille (int): dimension de la matrice carrée
        tab_triang (list): tableau des valeurs de la partie triangulaire inférieure de la matrice
    """
    #print("taille : ", taille)
    mat = np.zeros((taille, taille))
    tri_indices = np.triu_indices(taille)
    #print("tri indices : ", tri_indices)
    mat[tri_indices] = tab_triang
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


def adapte_parametres_mosek(task : mosek.Task):
    # Réduire la précision
    task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, 1e-3)
    task.putdouparam(mosek.dparam.intpnt_tol_pfeas, 1e-3)
    task.putdouparam(mosek.dparam.intpnt_tol_dfeas, 1e-3)
    # Limiter le temps et les itérations
    task.putdouparam(mosek.dparam.optimizer_max_time, 60) 
    task.putintparam(mosek.iparam.intpnt_max_iterations, 100)  
    # Désactiver le présolve
    ##task.putintparam(mosek.iparam.presolve_use, mosek.MSK_PRESOLVE_MODE_OFF)
    # Utiliser le simplexe dual
    ##task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.dual_simplex)
    # Limiter les threads
    task.putintparam(mosek.iparam.num_threads, 2)