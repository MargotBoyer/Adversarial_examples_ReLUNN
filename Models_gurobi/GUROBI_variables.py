import gurobipy as gp
from gurobipy import GRB
from typing import List


def add_variable_z(
        m : gp.Model, 
        K : int, 
        n : List[int], 
        L : List[List[float]], 
        couche : int = None, 
        neurone : int =None,
        impose_positive : bool = True):
    """Genere la variable z

    Args:
        m (gp.Model): Modele gurobi
        K (int): Nombre total de couches du reseau
        n (List[int]): Taille des couches de l'entree a la sortie
        L (List[List[float]]): Borne inf des vecteurs des couches
        couche (int, optional): Numero de la couche jusqu'a laquelle initialiser la variable z
        neurone (int, optional): Numero de l'unique neurone de la couche couche a initialiser
        couche et neurone doivent être simultanément None ou simultanément not None !
    """
    delta = 0.01
    z = gp.tupledict()
    max_couche = K
    # print("Initialisation de la variable z...")
    if couche is not None : 
        max_couche = couche
    # print(f"Neurone : {neurone}, maxcouche = {max_couche}")
    for k in range(max_couche + 1):
        lb_ = L[k]
        if impose_positive and (k>0) and (k<K):
            lb_ = [0] * n[k]
        if (k==K) and neurone is not None :
            print(f"Creation de la variable z pour le neurone {neurone} de la couche {couche}")
            z[k, neurone] = m.addVar(lb = lb_[neurone]-delta, vtype=GRB.CONTINUOUS)
            continue
        for j in range(n[k]):
            z[k, j] = m.addVar(lb = lb_[j]-delta, vtype=GRB.CONTINUOUS)  # PAS DE LOWER BOUNDS ICI !
    return z


def add_variable_beta(m : gp.Model, K : int, n : List[int], relax : bool):
    beta = gp.tupledict()
    for j in range(n[K]):
        if relax :
            beta[j] = m.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS)
        else:
            beta[j] = m.addVar(vtype=GRB.BINARY)
    return beta


def add_variable_sigma(m : gp.Model, 
                       couche_finale : int, 
                       n : List[int], 
                       relax : bool,
                       neurones_stables : List = []):
    sigma = gp.tupledict()
    for k in range(1, couche_finale+1):
        for j in range(n[k]):
            if (k,j) in neurones_stables:
                # print(f"La variable sigma pour le neurone {j} de la couche {k} n'a pas ete cree : ce neurone est stable.")
                pass
            elif relax == 1:
                sigma[k, j] = m.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS)
            else:
                sigma[k, j] = m.addVar(vtype=GRB.BINARY)
    return sigma

def add_variable_s(m : gp.Model, K : int, n : List[int]):
    s = gp.tupledict()
    for k in range(1, K + 1):
        for j in range(n[k]):
            s[k, j] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
    return s

def add_variable_d(m : gp.Model, K : int, n : List[int]):
    d = gp.tupledict()
    for j in range(n[0]):
        d[j] = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
    return d