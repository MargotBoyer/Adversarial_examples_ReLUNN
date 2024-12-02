import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Chemin vers le fichier .ilp
file_path = 'MILP/Code_Margot/model_dist.ilp'

# Lire le modèle depuis le fichier .ilp
model = gp.read(file_path)


def est_inversible(matrice):
    # Calculer le déterminant de la matrice
    det = np.linalg.det(matrice)
    
    # Vérifier si le déterminant est différent de zéro
    return det != 0

# Extraire les contraintes
constraints = model.getConstrs()

# Initialiser les listes pour W et b
W = []
b = []

for constr in constraints:
    # Obtenir l'expression de la contrainte
    expr = model.getRow(constr)
    
    # Obtenir le terme constant de la contrainte
    const = -constr.getAttr(GRB.Attr.RHS)
    
    # Créer un vecteur de coefficients pour cette contrainte
    coeffs = [0] * model.NumVars
    for i in range(expr.size()):
        var = expr.getVar(i)
        coeff = expr.getCoeff(i)
        coeffs[var.index] = coeff
    
    W.append(coeffs)
    b.append(const)



W = np.array(W)
b = np.array(b)

print("Matrice W :")
print(W)
print("shape W : ", W.shape)
print("Vecteur b : ")
print(b)
print("shape b  : ", b.shape)

if est_inversible(W):
    print("W est inversible ! ")
else :
    print("W est non inversible !")
