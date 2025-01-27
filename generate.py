import argparse
import random
import numpy as np
import torch
#import torch.optim as optim
#from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_blobs, make_circles, make_moons, make_gaussian_quantiles
from torch.utils import data
#from torchvision import datasets, transforms
from plotdata import vizualize

#from model import DenseModel

def generate_gaussian_circles_concentriques(n_samples, n_circles, noise=0.1, random_state=None):
    np.random.seed(random_state)
    
    X_total = []
    y_total = []

    for i in range(n_circles):
        radius = (i + 1) / n_circles
        # Générer des angles uniformément répartis
        angles = np.random.uniform(0, 2 * np.pi, n_samples)
        # Générer des rayons avec une distribution normale centrée sur `radius`
        radii = np.random.normal(loc=radius, scale=noise, size=n_samples)
        
        # Convertir en coordonnées cartésiennes
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        
        X = np.vstack((x, y)).T
        y_labels = np.full(n_samples, i)
        
        X_total.append(X)
        y_total.append(y_labels)

    X_total = np.vstack(X_total)
    y_total = np.concatenate(y_total)

    return X_total, y_total


def generate_multiple_circles(n_circles, n_samples):
    n_samples_per_circle = int(n_samples / n_circles)
    n_samples_left = n_samples - n_circles * n_samples_per_circle

    n = int(n_circles * n_circles / 4)
    cov = [[1, 0], [0, 1]]  

    means = []
    samples = []
    labels = []

    for i in range(n_circles):
        mean = [random.randint(-n, n), random.randint(-n, n)]  
        means.append(mean)
        if i == 0 : 
            sample = np.random.multivariate_normal(mean, cov, n_samples_per_circle + n_samples_left)   
            labels.append([i for j in range(n_samples_per_circle + n_samples_left)])
        else :
            sample = np.random.multivariate_normal(mean, cov, n_samples_per_circle)
            labels.append([i for j in range(n_samples_per_circle)])
        samples.append(sample)


    X_train = np.concatenate(samples, axis = 0)
    y_train = np.concatenate(labels, axis = 0)
    return X_train, y_train


def generate_X_y(name="MOON", n_samples=100, noise=0.05):
    if name == "MOON":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=0)
        X += 2
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y)
        return X,y, ""

    elif name == "CIRCLE":
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=0)
        X += 1
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y)
        return X,y, ""

    elif name == "BLOB":
        centers = [(1, 4), (2, 2), (4, 1)]
        X, y = make_blobs(
            n_samples=n_samples, cluster_std=noise, random_state=0, centers=centers
        )
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y)
        return X,y, ""

    elif name == "MULTIPLE_BLOB":
        n_circles = 5
        X, y= generate_multiple_circles(n_circles, n_samples)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y)
        return X,y, f"n_circles={n_circles}"


    elif name == "MULTIPLE_CIRCLES":
        n_circles = 2  # Number of circles
        X, y = generate_gaussian_circles_concentriques(n_samples=n_samples // n_circles, 
                                         n_circles=n_circles, noise=noise, random_state=42)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).long()
        print("y : ", y)
        print("X : ", X)
        return X,y, f"n_circles={n_circles}"

    elif name == "MULTIPLE_DIM_GAUSSIANS" :
        n_features = 8
        n_classes = 5
        X, y = make_gaussian_quantiles(n_samples=1000, n_features=n_features, n_classes=n_classes, random_state=42)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).long()
        centre = np.mean(X.numpy(), axis=0)
        return X,y, f"n_features={n_features}_n_classes={n_classes}"

    return None, None


def generate_toy(name="BLOB", n_samples=1000, noise=0.05, split=0.8, save=True):
    X,y,annotations = generate_X_y(name,n_samples,noise)

    dataset = data.TensorDataset(X, y)  # create the datset
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    if save:
        torch.save(dataset, f"datasets/{name}/{name}_noise={noise}_{annotations}_dataset.pt")
    vizualize(X.numpy(),y.numpy())


def main():
    generate_toy(name="BLOB", n_samples=10000, noise=0.15)

    # Si on veut lancer par ligne de commande  
    # python generate.py --name BLOB --n_samples 100 --noise 0.1

    # parser = argparse.ArgumentParser(description="balaba")
    # parser.add_argument("--name", type=str, default=1, help="Name of the dataset.")
    # parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
    # parser.add_argument("--noise", type=float, default=0.5, help="Data variance")
    # args = parser.parse_args()
    # generate_toy(args.name, args.n_samples, args.noise)
    
    

    

if __name__ == "__main__":
    main()
