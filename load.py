import torch
from typing import List
import random

def load_data(data_modele : str, architecture : str = None):
    if data_modele == "MOON":
        data = torch.load('datasets/MOON/MOON_dataset.pt')

    elif data_modele == "BLOB":
        data = torch.load('datasets/BLOB/BLOB_dataset.pt')

    elif data_modele == "MULTIPLE_BLOB":
        data = torch.load('datasets/MULTIPLE_BLOB/MULTIPLE_BLOB_dataset.pt')

    elif data_modele == "CIRCLE" :
        data = torch.load('datasets/CIRCLE/CIRCLE_dataset.pt')

    elif data_modele == "MULTIPLE_CIRCLES" :
        print("on selectionne des donnees de type multiple circles")
        data = torch.load('datasets/MULTIPLE_CIRCLES/MULTIPLE_CIRCLES_noise=0.15_n_circles=2_dataset.pt')

    elif data_modele == "MULTIPLE_DIM_GAUSSIANS":
        data = torch.load('datasets/MULTIPLE_DIM_GAUSSIANS/MULTIPLE_DIM_GAUSSIANS_dataset.pt')

    elif data_modele == "MNIST" :
        # transform = transforms.Compose([transforms.ToTensor())])
        # train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        # test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        # data = [train_dataset, test_dataset]
        data = torch.load('datasets/MNIST/critical_points_dataset_seuil=0.5.pt')
    print("Data loaded !  \n ", data[0])
    return data


def load_file_weights(data_modele : str, architecture : str = None):
    if data_modele == "MOON":
        file = 'datasets/MOON/Res_Moon_weights.pth'

    elif data_modele == "BLOB":
        file = 'datasets/BLOB/Res_BLOB_weights.pth'

    elif data_modele == "MULTIPLE_BLOB":
        file = 'datasets/MULTIPLE_BLOB/Res_MULTIPLE_BLOB_weights.pth'

    elif data_modele == "CIRCLE" :
        file = 'datasets/CIRCLE/Res_CIRCLE_weights.pth'

    elif data_modele == "MULTIPLE_CIRCLES" :
        print("on selectionne des donnees de type multiple circles")
        file = 'datasets/MULTIPLE_CIRCLES/Res_MULTIPLE_CIRCLES_weights.pth'

    elif data_modele == "MULTIPLE_DIM_GAUSSIANS":
        file = 'datasets/MULTIPLE_DIM_GAUSSIANS/Res_MULTIPLE_DIM_GAUSSIANS_weights.pth'

    elif data_modele == "MNIST" :
        file = 'datasets/MNIST/Res_MNIST_6x100_weights.pth'
        if architecture is not None : 
            file = f'datasets/MNIST/Res_MNIST_{architecture}_weights.pth'
    return file

def cherche_ycible(ytrue : int,
                   nK : int):
    """ Renvoie une classe cible aléatoire """
    possible_values = [i for i in range(nK) if i != ytrue]
    return random.choice(possible_values)

def liste_ycible(ytrue : int,
                   nK : int):
    """ Renvoie une liste avec toutes les classes cibles adversariales possibles """
    return [i for i in range(nK) if i != ytrue]



def retourne_weights(K : int, 
                     n : List[int], 
                     file : str):
    parametres = torch.load(file)
    W = []
    b = []
    print("Retourne weights activé")
    for k in range(K):
        weight_k = f"layers.{2 * k}.weight"
        bias_k = f"layers.{2 * k}.bias"
        W.append(parametres[weight_k].tolist())
        b.append(parametres[bias_k].tolist())
    return W, b


