#import cplex
from cplex.exceptions import CplexError
from docplex.mp.model import Model
import time


def solveGlover(K, n, x0, ycible, U, L, W, b, epsilon, epsilon_adv, relax, verbose=True):
    print("epsilon  : ", epsilon)

    m = Model(name='Glover_cplex')

    d = {(j): m.continuous_var(lb =0, name='d_{0}'.format(j))  for j in range(n[0])}
    alpha = {(k,j): m.binary_var(name='alpha_{0}_{1}'.format(k,j)) for k in range(1,K) for j in range(n[k])}
    z = {(k,j): m.continuous_var(lb = L[0][0], name='z_{0}_{1}'.format(k,j)) for k in range(K+1) for j in range(n[k])}

    # Contraintes sur la boule autour de x
    for i in range(n[0]):
        m.add_constraint(d[i] >= z[0,i] - x0[i], "distance__{}".format(i))  
        m.add_constraint(d[i] >= x0[i] - z[0,i], "distance_{}".format(i)) 
        m.add_constraint(d[i] <= epsilon, "distance_epsilon_{}".format(i))  

    # Contrainte ReLU pour les couches intermédiaires
    for k in range(1,K):
        for j in range(n[k]): 
            # sub_z = [z[k-1, i] for i in range(n[k-1])]  
            m.add_constraint(z[k,j] >= 0, "ReLU_neurone_{}_{}_>=".format(k,j))
            m.add_constraint(m.sum(W[k-1][j][i] * z[k-1,i] for i in range(n[k-1])) + b[k-1][j] <= U[k-1][j] * alpha[k,j], "ReLU_neurone_{}_{}".format(k,j))
            m.add_constraint(m.sum(W[k-1][j][i] * z[k-1,i] for i in range(n[k-1])) + b[k-1][j] >= L[k-1][j] * (1-alpha[k,j]), "ReLU_neurone_{}_{}".format(k,j))
            
            m.add_constraint(z[k,j] <= m.sum(W[k-1][j][i] * z[k-1,i] for i in range(n[k-1])) + b[k-1][j] + U[k-1][j] * (1-alpha[k,j]), "ReLU_neurone_{}_{}".format(k,j))
            m.add_constraint(z[k,j] >= m.sum(W[k-1][j][i] * z[k-1,i] for i in range(n[k-1])) + b[k-1][j] + L[k-1][j] * (1-alpha[k,j]), "ReLU_neurone_{}_{}".format(k,j))
            m.add_constraint(z[k,j] <= U[k-1][j] * alpha[k,j], "ReLU_neurone_{}_{}".format(k,j))
            m.add_constraint(z[k,j] >= L[k-1][j] * alpha[k,j], "ReLU_neurone_{}_{}".format(k,j))


    # Contrainte ReLU sur la derniere couche
    for j in range(n[K]):
        m.add_constraint(z[K,j] == m.sum(W[K-1][j][i] * z[K-1,i] for i in range(n[K-1])) + b[K-1][j], "ReLU_neurone_{}_{}".format(k,j))

    # Contrainte definissant un exemple adverse
    for j in range(n[K]):
        if j!=ycible :
            m.add_constraint(z[K,ycible] >= (1+rho) * z[K,j] , "Contrainte adverse_ycible={}_j={}".format(ycible,j))


    # Imprimer les contraintes
    print("Contraintes du modèle :")
    for ct in m.iter_constraints():
        print(ct)

    m.set_objective("min", m.sum(d))
    if verbose :
        m.print_information()
    
    # Essayer de résoudre le modèle et vérifier si c'est faisable
    start_time = time.time()
    solution = m.solve(log_output=True)
    end_time = time.time()

    execution_time = end_time - start_time
    if verbose : 
        print("Temps d'exécution du modèle : {:.2f} secondes".format(execution_time))

    
    opt = 0
    status = 4
    Sol = []
    if solution is None:
        if verbose : 
            print("Le modèle n'a pas été résolu avec succès.")
        status = 1
        Sol, opt, status, execution_time
    else:
        
        for j in range(n[0]):
            Sol.append(z[0, j].solution_value)
        sortie = []
        for j in range(n[K]):
            sortie.append(z[K,j].solution_value)
        if verbose : 
            solution.display()
            # m.print_solution()
            print("Solution : ", Sol)
            print("Classes prédites : ", sortie)
        status = 1

    return Sol, opt, status, execution_time



# Example usage:
K = 3
W = [[[0.3,-0.2],[-0.15,-0.5],[0.3,0.5]],[[-0.3,0.1,0.15,0.8],[0.3,-0.2,-0.15,0.6]], [[0.3,-0.2,0.4],[-0.1,-0.15,0.6],[-0.5,-0.2,-0.2],[0.3,0.5,0.2]]]
W_reverse = [ [ [couche[i][j]  for i in range(len(couche))]  for j in range(len(couche[0]))  ] for couche in W]
b = [[-0.2,0.1],[0.3,-0.2,0.3,-0.1],[0.2,-0.5,0.1]]
n = [3,2,4,3]
x0 = [0.6,-0.3,0.4]
uns = [1,1,1]
u = 1500
lb = -1500
y0 = 0
ycible = 2
U = [[u]*len(layer) for layer in W_reverse]
L = [[lb]*len(layer) for layer in W_reverse]
print("U : ", U)

rho = 0.1
epsilon = 10
relax = 0
Sol, opt, status, execution_time = solveGlover(K, n, x0, ycible, U, L, W_reverse, b, epsilon, rho, relax, verbose=1)