import gurobipy as gp
from gurobipy import GRB
from typing import List

def add_objective_diff(
        m : gp.Model, 
        z : gp.tupledict, 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        K : int, 
        n : List[int], 
        ytrue : int):
    """ Objectif Adv1 """
    m.setObjective( (gp.quicksum(W[K-1][ytrue][i] * z[K-1,i] for i in range(n[K-1])) + 
                     b[K-1][ytrue] - gp.quicksum(gp.quicksum(W[K-1][j][i] * z[K-1,i] for i in range(n[K-1])) + 
                                                 b[K-1][j] for j in range(n[K]) if j!= ytrue) ), GRB.MINIMIZE)
    

def add_objective_diff_with_beta(
        m : gp.Model, 
        z : gp.tupledict, 
        beta : gp.tupledict, 
        K : int, 
        n : List[int], 
        W : List[List[float]], 
        b : List[List[float]], 
        ytrue : int):
    """ Objectif Adv2 ou Adv3 """
    m.setObjective( (gp.quicksum(W[K-1][ytrue][i] * z[K-1,i] for i in range(n[K-1])) + b[K-1][ytrue]
                      - gp.quicksum((gp.quicksum(W[K-1][j][i] * z[K-1,i] for i in range(n[K-1])) + 
                                     b[K-1][j]) * beta[j] for j in range(n[K]) if j!=ytrue)), 
                                     GRB.MINIMIZE)

def add_objective_diff_ycible(
        m : gp.Model, 
        z : gp.tupledict, 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        K : int, 
        n : List[int], 
        ytrue : int,
        ycible : int):
    """ Objectif Advycible """
    m.setObjective( 
        gp.quicksum(W[K-1][ytrue][i] * z[K-1,i] for i in range(n[K-1])) + b[K-1][ytrue] 
                     - ((gp.quicksum(W[K-1][ycible][i] * z[K-1,i] for i in range(n[K-1])) + b[K-1][ycible] ) ), GRB.MINIMIZE
                     )
    

def add_objective_dist(m : gp.Model, d : gp.tupledict, n : List[int]):
    """ Objectif avec les distances du modèle Fischetti """
    m.setObjective(gp.quicksum(d[j] for j in range(n[0])), GRB.MINIMIZE)


def add_objective_U(m : gp.Model, 
                    z : gp.tupledict, 
                    couche : int, 
                    neurone : int):
    """ Objectif calculant la borne sup U de la valeur du neurone neurone de la couche couche"""
    m.setObjective(z[couche,neurone], GRB.MAXIMIZE)


def add_objective_L(m : gp.Model, 
                    z : gp.tupledict, 
                    couche : int, 
                    neurone : int):
    """ Objectif calculant la borne inf L de la valeur du neurone neurone de la couche couche"""
    m.setObjective(z[couche,neurone], GRB.MINIMIZE)


def add_objective_U_linear(m : gp.Model, 
                    z : gp.tupledict, 
                    W : List[List[List[float]]],
                    b : List[List[float]],
                    n : List[int],
                    couche : int, 
                    neurone : int,
                    neurones_inactifs_stables : list = None):
    """ Objectif calculant la borne sup U de la valeur du neurone neurone de la couche couche"""
    m.setObjective(gp.quicksum(W[couche-1][neurone][i] * z[couche-1,i] for i in range(n[couche-1]) if (couche,i) not in neurones_inactifs_stables) 
                   + b[couche-1][neurone], GRB.MAXIMIZE)

def add_objective_L_linear(m : gp.Model, 
                    z : gp.tupledict, 
                    W : List[List[List[float]]],
                    b : List[List[float]],
                    n : List[int],
                    couche : int, 
                    neurone : int,
                    neurones_inactifs_stables : list = []):
    """ Objectif calculant la borne sup L de la valeur du neurone neurone de la couche couche"""
    m.setObjective(gp.quicksum(W[couche-1][neurone][i] * z[couche-1,i] for i in range(n[couche-1]) if (couche,i) not in neurones_inactifs_stables)
                   + b[couche-1][neurone], GRB.MINIMIZE)
