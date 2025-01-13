import gurobipy as gp
from gurobipy import GRB
from typing import List

# ***************************************************************************
# *********** Contraintes sur la boule intiale *******************************
# ***************************************************************************
def add_initial_ball_constraints(
        m : gp.Model, 
        z : gp.tupledict, 
        x0 : List[float], 
        epsilon : float, 
        n : List[int],
        L_init : List[float],
        U_init : List[float],
        neurone : int = None):
    """ Ajout des contraintes sur la boule initiale :   x - epsilon <= z0 <= x + epsilon """
    if neurone is not None :
        m.addConstr(z[0, neurone] <= min(x0[neurone] + epsilon, U_init[neurone]) )
        # m.addConstr(z[0, j] <= x0[j] + epsilon)
        m.addConstr(z[0, neurone] >= max(x0[neurone] - epsilon, L_init[neurone]) )
    else : 
        for j in range(n[0]):
            m.addConstr(z[0, j] <= min(x0[j] + epsilon, U_init[j]) )
            # m.addConstr(z[0, j] <= x0[j] + epsilon)
            m.addConstr(z[0, j] >= max(x0[j] - epsilon, L_init[j]) )
            # m.addConstr(z[0, j] >= x0[j] - epsilon )

def add_distance_constraints(
        m : gp.Model, 
        z : gp.tupledict, 
        d : gp.tupledict, 
        x0 : List[float], 
        epsilon : float, 
        n : List[int]):
    """ Ajoute les contraintes definissant les distances dj = | xj - z0j | """
    for j in range(n[0]):
        m.addConstr(z[0, j] - x0[j] <= d[j])
        m.addConstr(z[0, j] - x0[j] >= -1 * d[j])
        m.addConstr(d[j] <= epsilon)
        m.addConstr(d[j] >= 0)





# ***************************************************************************
# *** Contraintes sur le passage par les couches internes (avec ReLU) ******
# ***************************************************************************
def add_hidden_layer_constraints_with_s(
        m : gp.Model, 
        z : gp.tupledict, 
        s : gp.tupledict, 
        sigma : gp.tupledict, 
        K : int, 
        n : List[int], 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        U : List[List[float]], 
        L : List[List[float]],
        neurones_actifs_stables : List = [],
        neurones_inactifs_stables : List = []):
    """ Contrainte ReLU1 """
    for k in range(1, K):
        for j in range(n[k]):
            if (k,j) in neurones_inactifs_stables :
                    m.addConstr(z[k, j] == 0)
                    continue
            elif (k,j) in neurones_actifs_stables :
                    m.addConstr(z[k, j] == gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1])) 
                                + b[k - 1][j])
                    continue
            m.addConstr(
                gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1]))
                + b[k - 1][j]
                == z[k, j] - s[k, j]
            )

            m.addConstr(z[k, j] <= U[k][j] * sigma[k, j])
            m.addConstr(s[k, j] <= -L[k][j] * (1 - sigma[k, j]))
            m.addConstr(s[k, j] >= 0)
            m.addConstr(s[k, j] <= -L[k][j])

            


def add_hidden_layer_constraints_quad(
        m : gp.Model, 
        z : gp.tupledict, 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        K : int, 
        n : List[int],
        neurones_actifs_stables : List = [],
        neurones_inactifs_stables : List = []):
    """ Contrainte ReLU4 """
    for k in range(1, K):
        for j in range(n[k]):
            if (k,j) in neurones_inactifs_stables :
                    m.addConstr(z[k, j] == 0)
                    continue
            elif (k,j) in neurones_actifs_stables :
                    m.addConstr(z[k, j] == gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1])) 
                                + b[k - 1][j])
                    continue
            m.addConstr(gp.quicksum(W[k-1][j][i] * z[k-1, i] for i in range(n[k-1])) + b[k-1][j] <= z[k, j])
            m.addConstr((z[k, j] - gp.quicksum(W[k-1][j][i] * z[k-1, i] for i in range(n[k-1])) - b[k-1][j]) * z[k, j] == 0)


def add_hidden_layer_constraints_with_sigma_quad(
        m : gp.Model, 
        z : gp.tupledict, 
        sigma : gp.tupledict, 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        K : int, 
        n : List[int], 
        U : List[List[float]], 
        L : List[List[float]],
        neurones_actifs_stables : List = [],
        neurones_inactifs_stables : List = []):
    """ Contrainte ReLU3 """
    for k in range(1, K):
        for j in range(n[k]):
            if (k,j) in neurones_inactifs_stables :
                    m.addConstr(z[k, j] == 0)
                    continue
            elif (k,j) in neurones_actifs_stables :
                    m.addConstr(z[k, j] == gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1])) 
                                + b[k - 1][j])
                    continue
            m.addConstr(
                gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1]))
                + b[k - 1][j]
                <= U[k][j] * sigma[k, j]
            )
            m.addConstr(
                gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1]))
                + b[k - 1][j]
                >= L[k][j] * (1 - sigma[k, j])
            )

            m.addConstr(
                z[k, j]
                == sigma[k, j]
                * (
                    gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1]))
                    + b[k - 1][j]
                )
            )


def add_hidden_layer_constraints_with_sigma_linear_Glover(
        m : gp.Model, 
        z : gp.tupledict, 
        sigma : gp.tupledict, 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        K : int, 
        n : List[int], 
        U : List[List[float]], 
        L : List[List[float]],
        neurones_actifs_stables : List = [], 
        neurones_inactifs_stables : List = []):
    """ Contrainte ReLU3 """
    for k in range(1, K):
        for j in range(n[k]):
            if (k,j) in neurones_inactifs_stables :
                 m.addConstr(z[k, j] == 0)
                 continue
            elif (k,j) in neurones_actifs_stables :
                 m.addConstr(z[k, j] == gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1])) 
                             + b[k - 1][j])
                 continue

            m.addConstr(
                gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1]))
                + b[k - 1][j]
                <= U[k][j] * sigma[k, j]
            )
            m.addConstr(
                gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1]))
                + b[k - 1][j]
                >= L[k][j] * (1 - sigma[k, j])
            )

            m.addConstr(
                z[k, j]
                >= gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1]))
                + b[k - 1][j]
                - U[k][j] * (1 - sigma[k, j])
            )
            m.addConstr(z[k, j] >= L[k][j] * sigma[k, j])
            m.addConstr(
                z[k, j]
                <= gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1]))
                + b[k - 1][j]
                - L[k][j] * (1 - sigma[k, j])
            )
            m.addConstr(z[k, j] <= U[k][j] * sigma[k, j])


def add_bounds_on_hidden_layers_constraints(
        m : gp.Model, 
        z : gp.tupledict, 
        K : int, 
        U : List[List[float]], 
        n : List[int]):
    """ Contraintes de bornes sur z """
    for k in range(1,K):
        for j in range(n[k]):
            m.addConstr(z[k,j]<=U[k][j])
            m.addConstr(z[k,j]>=0)


def add_quadratic_bounds_on_all_layers_constraints(
        m : gp.Model, 
        z : gp.tupledict, 
        x0 : List[float],
        K : int, 
        U : List[List[float]], 
        L : List[List[float]], 
        n : List[int],
        epsilon : float):
    """ Contraintes de bornes sur z pour la couche initiale et les couches internes """
    for j in range(n[0]):
        lb = max(L[0][j], x0[j] - epsilon)
        ub = min(U[0][j], x0[j] + epsilon)
        m.addConstr(z[0,j] * z[0,j] - (lb + ub) *  z[0,j] + lb * ub <= 0)
    for k in range(1,K):
        for j in range(n[k]):
            m.addConstr(z[k,j] * z[k,j] - (L[k][j] + U[k][j]) *  z[k,j] + L[k][j] * U[k][j] <= 0)



def add_hidden_layers_ReLU_convex_relaxation(
        m : gp.Model, 
        z : gp.tupledict, 
        K : int, 
        n : List[int], 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        U : List[List[float]], 
        L : List[List[float]],
        neurones_actifs_stables : List = [],
        neurones_inactifs_stables : List = []):
    """ Contrainte ReLU non exacte mais linéaire : relaxation convexe """
    for k in range(1, K):
        for j in range(n[k]):
            if (k,j) in neurones_inactifs_stables :
                 m.addConstr(z[k, j] == 0)
                 continue
            elif (k,j) in neurones_actifs_stables :
                 m.addConstr(z[k, j] == gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1])) 
                             + b[k - 1][j])
                 continue
            
            m.addConstr(z[k,j] >= gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1])) + b[k - 1][j])
            # Calcul des coordonnées de la droite y = ax + b reliant les points (U,U) et (L,0)
            a = U[k][j] / (U[k][j] - L[k][j])
            d = - L[k][j] * a

            m.addConstr(z[k,j] <= 
                        a * (gp.quicksum(W[k - 1][j][i] * z[k - 1, i] for i in range(n[k - 1])) + b[k - 1][j]) + d)




# ***************************************************************************
# ************ Contrainte dernière couche SANS ReLU *************************
# ***************************************************************************
def add_last_layer(
        m : gp.Model, 
        z : gp.tupledict, 
        K : int, 
        n : List[int], 
        W : List[List[float]], 
        b : List[List[float]],
        neurone : int = None):  
    """ Contrainte sur la dernière couche (uniquement lineaire) du modele """
    # Cette contrainte pose problème : l'éviter et remplacer directement la valeur de zK+1 
    # dans les contraintes adversariales et l'objectif
    if neurone is not None : 
        m.addConstr(z[K,neurone] - gp.quicksum(W[K-1][neurone][i] * z[K-1,i] for i in range(n[K-1])) == b[K-1][neurone] )
        return
    for j in range(n[K]):
        m.addConstr(z[K,j] - gp.quicksum(W[K-1][j][i] * z[K-1,i] for i in range(n[K-1])) == b[K-1][j] )





# ***************************************************************************
# **************** Contrainte adversariale *********************************
# ***************************************************************************
def add_adversarial_constraints(
        m : gp.Model, 
        z : gp.tupledict, 
        beta : gp.tupledict, 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        U : List[List[float]], 
        K : int, 
        n : List[int], 
        ytrue : int, 
        epsilon_adv : float):
    """ Contrainte adversariale Adv1 (premiere partie de la contrainte)"""
    for j in range(n[K]):
        if j != ytrue:
            m.addConstr(U[K][ytrue] * (1 - beta[j]) + 
                        gp.quicksum(W[K-1][j][i] * z[K-1, i] for i in range(n[K-1])) + b[K-1][j] >= 
                        (1 + epsilon_adv) * 
                        (gp.quicksum(W[K-1][ytrue][i] * z[K-1, i] for i in range(n[K-1])) + b[K-1][ytrue]))



def add_adversarial_constraints_product(
        m : gp.Model, 
        z : gp.tupledict, 
        beta : gp.tupledict, 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        K : int, 
        n : List[int], 
        ytrue : int, 
        epsilon_adv : float):
    """ Ajoute contrainte adversariale Adv2 (premiere partie de la contrainte)"""
    # Contrainte : betaj * zj >= betaj * ztrue
    for j in range(n[K]):
        if j != ytrue:
            m.addConstr( ((gp.quicksum(W[K-1][j][i] * z[K-1, i] for i in range(n[K-1])) + b[K-1][j]) * beta[j]) >=
                        ((1 + epsilon_adv) * beta[j] *
                        (gp.quicksum(W[K-1][ytrue][i] * z[K-1, i] for i in range(n[K-1])) + b[K-1][ytrue])) 
            ) 

def add_adversarial_constraints_product_complet_Adv3(
        m : gp.Model, 
        z : gp.tupledict, 
        beta : gp.tupledict, 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        K : int, 
        n : List[int], 
        ytrue : int, 
        epsilon_adv : float):
    """ Contrainte adversariale Adv3 (premiere partie de la contrainte)"""
    # Contrainte : betaj * zj >= betaj * ztrue
    for j in range(n[K]):
        if j != ytrue:
            for j2 in range(n[K]) :
                if j==j2:
                    continue
                m.addConstr( ((gp.quicksum(W[K-1][j][i] * z[K-1, i] for i in range(n[K-1])) + b[K-1][j]) 
                                * beta[j]) >= ((1 + epsilon_adv) * beta[j] *
                            (gp.quicksum(W[K-1][j2][i] * z[K-1, i] for i in range(n[K-1])) + b[K-1][j2])) 
                ) 


def add_adversarial_constraints_product_betas_discrets(
        m : gp.Model, 
        beta : gp.tupledict, 
        K : int, 
        n : List[int], 
        ytrue : int):
    # ******* Contrainte : betai * betaj = 0 ************************************
    for j in range(n[K]):
        if j != ytrue:
            for j2 in range(j):
                if j2!=ytrue : 
                    m.addConstr(beta[j] * beta[j2] == 0) 

def add_adversarial_constraints_ycible(
        m : gp.Model,
        z : gp.tupledict, 
        K : int, 
        n : List[int], 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        rho : float, 
        ycible : int):
    """ Contrainte adversariale du modele de Fischetti """
    for j in range(n[K]):
        if j != ycible:
            m.addConstr( (gp.quicksum(W[K-1][ycible][i] * z[K-1, i] for i in range(n[K-1])) + b[K-1][ycible]) >= 
                        (1 + rho) * 
            (gp.quicksum(W[K-1][j][i] * z[K-1, i] for i in range(n[K-1])) + b[K-1][j]) )  

def add_somme_beta_egale_1(
        m : gp.Model, 
        beta : gp.tupledict, 
        K : int, 
        n : List[int],
        ytrue : int):
    m.addConstr(gp.quicksum(beta[j] for j in range(n[K]) if j!=ytrue) == 1) 

def add_somme_beta_superieure_1(
        m : gp.Model, 
        beta : gp.tupledict, 
        K : int, 
        n : List[int], 
        ytrue : int):
    m.addConstr(gp.quicksum(beta[j] for j in range(n[K]) if j!=ytrue) >= 1) 
            

