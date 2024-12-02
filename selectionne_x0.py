import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.neighbors import BallTree

import torch
from torch.utils.data import Dataset, DataLoader

from comparaison_modeles_outils import load_data


def get_x0():
    res = pd.read_csv("datasets/MNIST/Benchmark/MNIST_epsilon=0.7_taille=9/MNIST_epsilon=0.7_neurones=50_taille=9_benchmark.csv")
    x0 = ast.literal_eval(res.iloc[2]["x0"])
    print('x0 récupéré!')
    print("x0", x0)
    return x0


def plot_points_critiques(donnees2D, labels, df):
    #Visualisation du résultat
    # Créer une figure et un axe
    fig, ax = plt.subplots()
    nb_classes = len(set(labels))
    print("Nombre de classes : ", nb_classes)
    # Premier scatterplot avec les points colorés en fonction de 'label'
    sns.scatterplot(
        x=donnees2D[:, 0], 
        y=donnees2D[:, 1], 
        hue=labels, 
        palette=sns.hls_palette(nb_classes), 
        legend='full', 
        ax=ax
    )

    # Deuxième scatterplot avec les points critiques entourés de noir
    sns.scatterplot(
        x=df[df["Critique"]]["Coordonnée_0"], 
        y=df[df["Critique"]]["Coordonnée_1"], 
        hue=df[df["Critique"]]["label"], 
        palette=sns.hls_palette(nb_classes), 
        legend=False,  # Désactive la légende supplémentaire
        ax=ax,
        edgecolor='black',  # Contour noir
        linewidth=0.5,  # Épaisseur du contour
        s=50  # Taille des points pour mieux voir les contours (ajuster si besoin)
    )

    # Afficher le plot
    plt.show()


def selectionne_points_critiques(
        donnees2D : np.array, 
        labels : np.array, 
        seuil : float = 0.5, 
        epsilon : float = 1,
        vraiesdonnees : np.array = None
        ):
    
    # Création du BallTree avec les coordonnées des points
    tree = BallTree(donnees2D, metric='euclidean')
    # Recherche des paires de points dont la distance est inférieure à epsilon=1
    epsilon = 1
    indices = tree.query_radius(donnees2D, r=epsilon, return_distance=False)  
            # Liste : A chaque élement d'indice i correspond tout l'array des voisins de l'image numero i

    indices_labels = np.array([labels[sub_array] for sub_array in indices], dtype=object)
        # Liste : A chaque élement d'indice i correspond tout l'array des LABELS des voisins de l'image numero i
    indices_prop = ([np.mean(indices_labels[i] == labels[i]) if len(indices_labels[i]) > 0 else 0 for i in range(len(indices_labels)) ])
    indices_prop = np.array(indices_prop)
        # Liste : A chaque élement d'indice i correspond tout la proportion des voisins de l'image i qui ont le meme label que i
    # Selection des points critiques
    points_critiques = np.where(indices_prop < seuil)[0]
    print("Nombre de points critiques : ", points_critiques.shape)

    df_donnees = pd.DataFrame(columns = ["Coordonnée_0","Coordonnée_1"], index = range(donnees2D.shape[0]))
    df_donnees["Coordonnée_0"] = donnees2D[:,0]
    df_donnees["Coordonnée_1"] = donnees2D[:,1]
    print("df_donnes : ", df_donnees)
    if vraiesdonnees is not None :
        df_donnees["Donnees"] = vraiesdonnees
    df_donnees['label'] = labels
    df_donnees["Critique"] = df_donnees.index.isin(points_critiques)
    print(df_donnees[df_donnees["Critique"]]["label"].value_counts())
    return df_donnees


if __name__ == "__main__":
    data_modele = "MULTIPLE_CIRCLES"
    file, data = load_data(data_modele)
    X = []
    y = []

    dataloader = DataLoader(data[0], batch_size=100, shuffle=False)
    for batch, label in dataloader:
        print("batch : ", batch)
        X.extend(batch)  # Conversion des tensors en numpy
        y.extend(label.tolist() if isinstance(label, torch.Tensor) else label)

    # Conversion en arrays NumPy
    X = np.array(X)
    y = np.array(y)
    print("X : ", X)
    df = selectionne_points_critiques(X,y,0.2, epsilon = 0.3)
    plot_points_critiques(X,y,df)