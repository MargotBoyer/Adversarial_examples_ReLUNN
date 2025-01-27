import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import pickle

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
from load import retourne_weights, load_data
from reseau_train import Reseau, architectures_modele

from typing import List


def plot_grid_with_reseau(
        data : np.array, 
        labels : np.array, 
        adverses : List = None, 
        Res : torch.nn.Module = None, 
        x_min : float =0.5, 
        x_max : float = 4.5, 
        y_min : float = 1, 
        y_max : float = 3.5, 
        title : str = ""):
    """_summary_

    Args:
        data (np.array): Données à plotter
        labels (np.array): Labels des données
        n (List[int], optional): Liste de la taille des couches du réseau de neurones 
                                (avec à l'indice 0 la taille de l'input)
        K (int, optional): Nombre de couches du réseau
        adverses (List, optional): Liste d'exemple adverses (points mal prédits par le réseau)
        Res (torch.nn.Module, optional): Réseau de neurones
        x_min, x_max, y_min, y_max (float, optional): Bornes de l'intervalle de définition des données
        title (str, optional): Titre du graphique
    """
    # Convertir les données en NumPy pour utilisation avec matplotlib
    data_np = data
    labels_np = labels
    labels_max = np.max(labels_np)
    colors = ["yellow", "green", "blue","pink","purple","#483D8B","#8FBC8F","#B22222", "#4682B4", "#FFDB58"]
    adv_colors = ["orange", "darkgreen", "black"]

    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))


    for label in range(labels_max+1):
        idx = labels_np == label
        plt.scatter(
            data_np[idx, 0], data_np[idx, 1], c=colors[label], label=f"Label {label}"
        )

    if Res is not None : 
        X0 = torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T)
        print("XO shape: ", X0.shape)
        y0 = Res(X0)
        y0 = torch.argmax(y0,1)
        display = DecisionBoundaryDisplay(xx0=XX, xx1=YY, response=y0.view(100,100).data.numpy())
        display.plot(cmap='coolwarm')

    if adverses is not None and adverses != []: 
        for sample, label, x0, ytrue in adverses:
            print(f"Exemple adverse : {sample} et de classe {label}")
            
            plt.plot([sample[0], x0[0]], [sample[1], x0[1]], color='red')
            plt.scatter(
            sample[0], sample[1], c=adv_colors[label]
            )
            plt.scatter(
            x0[0], x0[1], c="black"
            )


    plt.legend()
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title)
    plt.show()



def vizualize(
        data : np.array, 
        labels : np.array, 
        adverses : List = None, 
        Res : torch.nn.Module = None, 
        x_min : float = 0.5, 
        x_max : float = 4.5, 
        y_min : float = 1, 
        y_max : float = 3.5, 
        title : str = ""):
    plot_grid_with_reseau(data, labels, adverses, Res, x_min, x_max, y_min, y_max, title)  # plt.title(f'esp2 = {eps2}')




def get_min_max_data(data_modele):
    """Renvoie les bornes de l'intervalle de définition de l'ensemble de données
    """
    if data_modele == "BLOB":
        data_file= "BLOB_dataset.pt"
        adverses_file = f"{data_modele}_{optimization_model}_adverses.pkl"
        x_min=0.5
        x_max=4.5
        y_min=0
        y_max=5
    elif data_modele == "MOON" :
        data_file= "MOON_dataset.pt"
        adverses_file = f"{data_modele}_{optimization_model}_adverses.pkl"
        frontiere_decision_file = f"{data_modele}_{optimization_model}_frontiere_decision.pkl"
        x_min=0.5
        x_max=4.5
        y_min=1
        y_max=3.5
    elif data_modele == "MULTIPLE_BLOB" :
        data_file= "MULTIPLE_BLOB.pt"
        adverses_file = f"{data_modele}_{optimization_model}_adverses.pkl"
        frontiere_decision_file = f"{data_modele}_{optimization_model}_frontiere_decision.pkl"
        n_circles = 5
        n = int(n_circles * n_circles / 4)
        x_min= -n
        x_max= n 
        y_min= -n
        y_max= n 
    elif data_modele == "CIRCLE" :
        data_file= "CIRCLE_dataset.pt"
        adverses_file = f"{data_modele}_{optimization_model}_adverses.pkl"
        frontiere_decision_file = f"{data_modele}_{optimization_model}_frontiere_decision.pkl"
        x_min=0
        x_max=2.5
        y_min=0
        y_max=2.5
    elif data_modele == "MULTIPLE_CIRCLES" :
        data_file= "MULTIPLE_CIRCLES_dataset.pt"
        adverses_file = f"{data_modele}_{optimization_model}_adverses.pkl"
        frontiere_decision_file = f"{data_modele}_{optimization_model}_frontiere_decision.pkl"
        x_min=-2
        x_max=2
        y_min=-2
        y_max=2
    elif data_modele == "MULTIPLE_DIM_GAUSSIANS" :
        data_file= "MULTIPLE_DIM_GAUSSIANS_dataset.pt"
        adverses_file = f"{data_modele}_{optimization_model}_adverses.pkl"
        frontiere_decision_file = f"{data_modele}_{optimization_model}_frontiere_decision.pkl"
        x_min=-10
        x_max=10
        y_min=-10
        y_max=10

    return x_min,x_max,y_min,y_max



if __name__ == "__main__":
   
    data_modele = "MULTIPLE_CIRCLES"
    optimization_model = "Mix_SDP"

    x_min, x_max, y_min, y_max = get_min_max_data(data_modele)
    
    n, K = architectures_modele(data_modele)
    file, data = load_data(data_modele)

    # print("n : ", n)
    # W, b = retourne_weights(K, n, file)
    # W_reverse = [[ [couche[i][j] for i in range(len(couche))] for j in range(len(couche[0]))] for couche in W]
    # Res = Reseau(K, n, W, b)

    # data = torch.load(f"MILP/Code_Margot/tests/{data_modele}/{data_file}")
    # print("Data loaded !  \n ", data[0])

    # with open(f'MILP/Code_Margot/tests/{data_modele}/{adverses_file}', 'rb') as f:
    #     print("On ouvre les adverses")
    #     adverses = pickle.load(f)
    Res = None
    
    adverses = []
    batch_size = 500
    trainloader = DataLoader(data[0], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(data[1], batch_size=batch_size, shuffle=True)

    first_batch = next(iter(trainloader))
    inputs, labels = first_batch

    for batch in tqdm(trainloader):
        inputs, labels = batch
        if adverses == []:
            vizualize(inputs.numpy(), labels.numpy(), adverses = None, Res = Res, 
                x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)
        else: 
            vizualize(inputs.numpy(), labels.numpy(), adverses, Res, 
                x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)
        break

