import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm

from load import load_file_weights, retourne_weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

def retourne_reseau():
    K=3
    W=[[[0.3,-0.2],[-0.15,-0.5],[0.3,0.5]],[[0.3,-0.2,-0.15,0.6],[0.3,-0.2,-0.15,0.6]] ,[[0.3,-0.2,0.4],[-0.1,-0.15,0.6],[-0.5,-0.2,-0.2],[0.3,0.5,0.2]]]
    b=[[-0.2,0.1],[0.3,-0.2,0.3,-0.1],[0.2,-0.5,0.1]]
    n=[3,2,4,3]
    return K,W,b,n


    


class Reseau(nn.Module): 
    def __init__(self, K, n, W=None, b=None):
        super(Reseau, self).__init__()
        self.K = K
        self.W = W
        self.b = b
        self.n = torch.tensor(n)

        self.layers = nn.ModuleList()
        for k in range(1,K+1):
            couche = nn.Linear(in_features=self.n[k-1], out_features=self.n[k])
            if W is not None:
                couche.weight = nn.Parameter(torch.tensor(W[k-1], dtype=torch.float32).clone().detach(), requires_grad=False)
                couche.bias = nn.Parameter(torch.tensor(b[k-1], dtype=torch.float32).clone().detach(), requires_grad=False)
            self.layers.append(couche)
            if k < K:
                self.layers.append(nn.ReLU(inplace=True))

    @classmethod
    def create_with_file(cls, data_modele, architecture = None):
        n, K = architectures_modele(data_modele,architecture)
        file = load_file_weights(data_modele, architecture)
        W, b = retourne_weights(K, n, file)
        return cls(K, n, W, b)

    def forward(self, x, verbose = False, return_last_hidden=False):
        x = torch.tensor(x, dtype=torch.float32).to(device)

        if x.dim() >= 3:  # Si les données sont des images (batch_size, channels, height, width)
            x = x.view(x.size(0), -1)  # Aplatir les images (batch_size, 784)
        n_couche = 1
        for i,layer in enumerate(self.layers):
            x = layer(x)

            if (i == len(self.layers) - 2) and return_last_hidden:  # Avant-dernière couche (couche sans ReLU)
                print("Derniere couche retournée")
                return x
            
            if verbose : 
                if n_couche%2==1:
                    print(f"Couche n° {n_couche//2} : ", x)
                elif n_couche%2==0:
                    print(f"Couche n° {n_couche//2} ReLU: ", x, " \n ")
            n_couche += 1
            
        return x

    def retourne_label(self, x):
        res = self.forward(x)
        # print("Sortie du reseau : ", res)
        label = torch.argmax(res)
        return label.item()

    def retourne_bornes(self, data_modele = None):
        
        ub = 1.5*10e2
        lb = -1.5*10e2

        U = [[ub for _ in range(self.n[k])] for k in range(self.K+1)]
        L = [[lb for _ in range(self.n[k])] for k in range(self.K+1)]

        if data_modele == "MNIST" :
            U = [[1500 for _ in range(self.n[k])] for k in range(self.K+1)]
            L = [[-1500 for _ in range(self.n[k])] for k in range(self.K+1)]
            for i in range(self.n[0]) :
                U[0][i] = 1
                L[0][i] = -1
        
        return L, U



def train(model, trainloader, num_epochs=100, lr=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("Entraînement du modèle ...")

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {running_loss/len(trainloader)}")
            evaluate(Res, testloader)


def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')
    

def plot_weights(model):
    for name, param in model.named_parameters():
        print(f"layer {name} : ", param)
        if 'weight' in name:
            plt.figure()
            plt.hist(param.cpu().detach().numpy().flatten(), bins=20)
            plt.title(f'Histogram of {name}')
            plt.show()





def architectures_modele(modele, architecture = None):
    if modele == "MOON" :
        n = [2, 5, 5, 5, 2]
        K = 4
    elif modele == "BLOB" :
        n = [2, 2, 3]
        K = 2
    elif modele == "MULTIPLE_BLOB" :
        n = [2, 10, 10, 5]
        K = 3
    elif modele == "CIRCLE" :
        n = [2, 10, 10, 2]
        K = 3
    elif modele == "MULTIPLE_CIRCLES" :
        n = [2, 5, 5, 2]
        K = 3
    elif modele == "MULTIPLE_DIM_GAUSSIANS" :
        n = [8, 30, 30, 5]
        K = 3
    elif modele == "MNIST" and architecture is None:
        n = [784, 8, 8, 8, 10]  
        K = 4
    elif modele == "MNIST" and architecture == "2x20":
        n = [784, 20, 20, 10]  
        K = 3
    elif modele == "MNIST" and architecture == "6x100":
        n = [784, 100, 100, 100, 100, 100, 100, 10]  
        K = 7
    elif modele == "MNIST" and architecture == "9x100":
        n = [784, 100, 100, 100, 100, 100, 100, 100, 100, 100, 10]  
        K = 10
    elif modele == "MNIST" and architecture == "8x1024":
        n = [784, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]  
        K = 9
    else : 
        print("Le modele n'existe pas dans les donnees.")
    return n, K








if __name__ == "__main__":
    modele = "MOON"
    architecture = None

    if modele == "MOON" :
        data = torch.load('datasets/MOON/MOON_dataset.pt')
    elif modele == "BLOB" :
        data = torch.load('datasets/BLOB/BLOB_dataset.pt')
    elif modele == "MULTIPLE_BLOB" :
        data = torch.load('datasets/MULTIPLE_BLOB/MULTIPLE_BLOB_dataset.pt')
    elif modele == "CIRCLE" :
        data = torch.load('datasets/CIRCLE/CIRCLE_dataset.pt')
    elif modele == "MULTIPLE_CIRCLES" :
        data = torch.load('datasets/MULTIPLE_CIRCLES/MULTIPLE_CIRCLES_noise=0.2_n_circles=2_dataset.pt')
    elif modele == "MULTIPLE_DIM_GAUSSIANS" :
        data = torch.load('datasets/MULTIPLE_DIM_GAUSSIANS/MULTIPLE_DIM_GAUSSIANS_dataset.pt')
    elif modele == "MNIST" :
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        data = [train_dataset, test_dataset]
        
        
    n, K = architectures_modele(modele, architecture)
    print("Data loaded!\n", data[0])
    print("n : ", n)
    
    Res = Reseau(K, n)
    #summary_model(Res)

    batch_size = 500
    trainloader = DataLoader(data[0], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(data[1], batch_size=batch_size, shuffle=True)
    
    # Calcul de la valeur maximum sur le dataloader
    # max = -100
    # min = 100
    # for i in range(100):
    #     train_features, train_labels = next(iter(trainloader))
    #     print(f"Feature batch shape: {train_features.size()}")
    #     print(f"Labels batch shape: {train_labels.size()}")
    #     print("train features : ", train_features)
    #     print("train features min : ", train_features.min())
    #     print("train features max : ", train_features.max())
    #     if max < train_features.max():
    #         max = train_features.max()
    #     if min > train_features.min():
    #         max = train_features.min()
    # print("Maximum sur MNIST train : ", max)
    # print("Minimum sur MNIST train : ", min)


    if modele == "MNIST" :
        train(Res, trainloader, num_epochs = 50)
    else : 
        train(Res, trainloader)
    evaluate(Res, testloader)
    #plot_weights(Res)

    if architecture is not None : 
        torch.save(Res.state_dict(), f'datasets/{modele}/Res_{modele}_{architecture}_weights.pth')
    else :
        torch.save(Res.state_dict(), f'datasets/{modele}/Res_{modele}_weights.pth')
