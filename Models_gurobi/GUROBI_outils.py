import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List




parametres_gurobi = {"FeasibilityTol": 1e-2, 
                         "NumericFocus" : 3,  #3
                         "Aggregate" : None,  #0
                         "NonConvex" : None,  #2
                         "Presolve": None, # 0  La désactivation du presolve peut aider à trouver des solutions faisables plus rapidement
                                        # Le presolve simplifie le problème
                         "TimeLimit" : 300,
                         "MinRelNodes" : None,   # 0,
                         "PumpPasses" : None, # 100,
                         "BarConvTol" : None, #1e-6,
                         "ImproveStartTime" : None,   #10,
                         "MIPFocus" : None,  #0, 
                         "StartNodeLimit" : None, # 1000,
                         "Heuristics" : None, # 0.5,  # Contrôle le pourcentage de temps dédié aux heuristiques dans gurobi
                         "Method" : None, # Choix de l'algorithme utilisé pour la résolution (1 : simplexe)
                        "DualReduction" : 0
                         }



def adapt_parametres_gurobi(m : gp.Model, parametres_gurobi : Dict):
    if parametres_gurobi["FeasibilityTol"] is not None : 
        m.setParam(GRB.Param.FeasibilityTol, parametres_gurobi["FeasibilityTol"])

    if parametres_gurobi["NumericFocus"] is not None : 
        m.setParam(GRB.Param.NumericFocus, parametres_gurobi["NumericFocus"])

    if parametres_gurobi["Aggregate"] is not None : 
        m.setParam(GRB.Param.Aggregate, parametres_gurobi["Aggregate"])

    if parametres_gurobi["NonConvex"] is not None :
        m.setParam(GRB.Param.NonConvex, parametres_gurobi["NonConvex"])

    if parametres_gurobi["Presolve"] is not None : 
        m.setParam(GRB.Param.Presolve, parametres_gurobi["Presolve"])

    if parametres_gurobi["TimeLimit"] is not None : 
        m.setParam(GRB.Param.TimeLimit, parametres_gurobi["TimeLimit"])

    if parametres_gurobi["MinRelNodes"] is not None : 
        print("ajout du parametre minrelnodes")
        m.setParam(GRB.Param.MinRelNodes, parametres_gurobi["MinRelNodes"])

    if parametres_gurobi["PumpPasses"] is not None : 
        print("ajout du parametre PumpPasses")
        m.setParam(GRB.Param.PumpPasses, parametres_gurobi["PumpPasses"])

    if parametres_gurobi["BarConvTol"] is not None : 
        print("ajout du parametre BarConvTol")
        m.setParam(GRB.Param.BarConvTol, parametres_gurobi["BarConvTol"])

    if parametres_gurobi["ImproveStartTime"] is not None : 
        print("ajout du parametre ImproveStartTime")
        m.setParam(GRB.Param.ImproveStartTime, parametres_gurobi["ImproveStartTime"])

    if parametres_gurobi["MIPFocus"] is not None : 
        print("ajout du parametre MIPFocus")
        m.setParam(GRB.Param.MIPFocus, parametres_gurobi["MIPFocus"])

    if parametres_gurobi["StartNodeLimit"] is not None : 
        print("ajout du parametre StartNodeLimit")
        m.setParam(GRB.Param.StartNodeLimit, parametres_gurobi["StartNodeLimit"])

    if parametres_gurobi["Heuristics"] is not None : 
        print("ajout du parametre Heuristics")
        m.setParam(GRB.Param.Heuristics, parametres_gurobi["Heuristics"])

    if parametres_gurobi["Method"] is not None : 
        print("ajout du parametre Method")
        m.setParam(GRB.Param.Method, parametres_gurobi["Method"])

    m.setParam("InfUnbdInfo", 1)

    if parametres_gurobi["DualReduction"] is not None :
        m.setParam("DualReductions", parametres_gurobi["DualReduction"])


    
def retourne_valeurs_solutions_bornes(
        m : gp.Model,
        z : gp.tupledict,
        couche : int,
        neurone : int,
        n : List[int],
        nom_borne : str,
        verbose:bool= False):
    
    if nom_borne == "U":
        Sol_borne = 1e8
    elif nom_borne == "L":
        Sol_borne = -1e8
    else :
        return
    status = -1
    time_execution = m.runtime
    opt = None
    nb_nodes = m.NodeCount
    if m.Status == GRB.OPTIMAL:
        opt = m.ObjVal
        # if verbose:
        #     print("Valeur optimale : ", round(opt, 4))
        #     print("CPU time :", m.runtime)
        Sol_borne = opt
        status = 1
        # if verbose:
        #     print(f"{nom_borne} : ", Sol_borne)
        return Sol_borne, status, time_execution, {"Number_Nodes" : nb_nodes}

    elif m.Status == GRB.INFEASIBLE:
        if verbose:
            print("Modele infaisable !")
        m.computeIIS()
        m.write(f'Models_gurobi\lp\{m.ModelName}.ilp')
        status = 3
    elif m.status == GRB.TIME_LIMIT:
        print("Temps limite atteint, récupération de la meilleure solution réalisable")
        print("Gap : ", m.MIPGap)
        if m.SolCount > 0:
            print("Solution réalisable disponible")
            #Sol_borne = z[couche, neurone].X
            return m.ObjBound, 2, time_execution, {"Number_Nodes" : nb_nodes}
        status = 2
    else:
        if verbose:
            print("Statut modele : ", m.Status)
        print("Bound calculee : ", m.ObjBoundC)
    return opt, status, time_execution, {"Number_Nodes" : nb_nodes}





































# def apply_feasRelax(model):
#     """Applique feasRelax à un modèle infaisable pour obtenir une solution faisable relaxée."""
#     # Créez une liste de pénalités pour les variables et les contraintes
#     vars = model.getVars()
#     constrs = model.getConstrs()
    
#     lbpen = [1.0] * len(vars)  # Pénalités pour la relaxation des bornes inférieures des variables
#     ubpen = [1.0] * len(vars)  # Pénalités pour la relaxation des bornes supérieures des variables
#     rhspen = [1.0] * len(constrs)  # Pénalités pour la relaxation des termes constants des contraintes

#     # Utiliser feasRelax pour relaxer les contraintes et trouver une solution faisable
#     model.feasRelax(
#         relaxobjtype=1,  # Minimiser la somme des violations
#         minrelax=True,   # Minimise la somme des relaxations nécessaires pour rendre le modèle faisable
#         vars=vars,
#         lbpen=lbpen,
#         ubpen=ubpen,
#         constrs=constrs,
#         rhspen=rhspen
#     )

#     # Optimiser le modèle relaxé
#     model.optimize()

#     if model.status == GRB.OPTIMAL:
#         print("Une solution faisable relaxée a été trouvée.")
#         solution = {v.varName: v.x for v in model.getVars()}
        
#         print("Relaxations des variables :")
#         for v in model.getVars():
#             lb_relax = model.relaxed[v.varName].SAObjLow if v.varName in model.relaxed else 0
#             ub_relax = model.relaxed[v.varName].SAObjUp if v.varName in model.relaxed else 0
#             if lb_relax != 0 or ub_relax != 0:
#                 print(f"{v.varName}: LBRelax={lb_relax}, UBRelax={ub_relax}")
                
#         print("Relaxations des contraintes :")
#         for c in model.getConstrs():
#             rhs_relax = model.relaxed[c.constrName].SARHS if c.constrName in model.relaxed else 0
#             if rhs_relax != 0:
#                 print(f"{c.constrName}: RHSRelax={rhs_relax}")
#     else:
#         print("Impossible de trouver une solution faisable relaxée.")
#         return None, None, model.status, model.Runtime



def apply_feasRelax(model):
    """Applique feasRelax à un modèle infaisable pour obtenir une solution faisable relaxée et stocke les relaxations dans un dictionnaire."""
    try:
        # Utiliser feasRelax pour relaxer les contraintes et trouver une solution faisable
        model.feasRelax(
            relaxobjtype=1,  # Minimiser la somme des violations
            minrelax=True,   # Minimise la somme des relaxations nécessaires pour rendre le modèle faisable
            vars=model.getVars(),
            lbpen=None,
            ubpen=None,
            constrs=model.getConstrs(),
            rhspen=None
        )

        # Créer un dictionnaire pour stocker les relaxations
        relaxed_values = {}

        # Stocker les relaxations des variables
        for v in model.getVars():
            relaxed_values[v.varName] = {
                'LBRelax': v.getAttr(GRB.Attr.SAObjLow),
                'UBRelax': v.getAttr(GRB.Attr.SAObjUp)
            }

        # Stocker les relaxations des contraintes
        for c in model.getConstrs():
            relaxed_values[c.constrName] = {
                'RHSRelax': c.getAttr(GRB.Attr.SARHS)
            }

        # Optimiser le modèle relaxé
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print("Une solution faisable relaxée a été trouvée.")
            solution = {v.varName: v.x for v in model.getVars()}
            return solution, model.ObjVal, model.status, model.Runtime, relaxed_values
        else:
            print("Impossible de trouver une solution faisable relaxée.")
            return None, None, model.status, model.Runtime, relaxed_values

    except gp.GurobiError as e:
        print(f"Erreur Gurobi: {e}")
        return None, None, model.status, model.Runtime, {}


