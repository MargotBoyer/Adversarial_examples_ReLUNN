import numpy as np
import mosek
from typing import List, Tuple
inf = 10e5


def contrainte_recurrence_matrices_couches(
        task : mosek.Task,
        K : int,
        n : List[int],
        num_contrainte : int):
    # Contrainte  : Pk[zk+1] == Pk+1[zk+1] ***********************
    # ***** Nombre de contraintes : sum(n[1:K]) *****************
    for k in range(1,K):
        for j in range(n[k]):
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],  
                [k-1, k], 
                [1 + n[k-1] + j, 1+ j],  
                [0, 0],
                [1 / 2, -1/2],
            )
            # Bornes
            task.putconboundlist(
                [num_contrainte], [mosek.boundkey.fx], [0], [0]
            )
            num_contrainte += 1

    return num_contrainte