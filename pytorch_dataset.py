#%%
import pandas as pd
import numpy as np
import inspect
import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from dataset import AirbnbNightlyPriceImageDataset
import modelling

class LinearRegression(torch.nn.Module, config):
    def __init__(self)->None:
        super().__init__()

        #initialise parameters
        
        self.linear_layers=torch.nn.Sequential(
            torch.nn.Linear(9,9),
            torch.nn.ReLU(),
            torch.nn.Linear(9,1)
        )
        self.double()

    def forward(self, features):
    #use layers to process features
        return self.linear_layers(features)

    def __repr__(cls):
        return 'Pytorch'

def train(model, train_loader, val_loader, epochs=100, lr=0.0001, optimiser=torch.optim.SGD):
    optimiser=optimiser(model.parameters(),lr=lr)
    
    #writer=SummaryWriter()
    
    batch_idx=0
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            labels=labels.unsqueeze(1)
            
            predicted=model(features)
            
            loss=F.mse_loss(predicted, labels)
            loss.backward()

            print(loss.item())

            #optimization step
            optimiser.step()
            #reset grad params for next backward step
            optimiser.zero_grad()
            #writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx+=1
            #if batch_idx%epochs==0:
                #val_loss, val_acc = evaluate(model, val_loader)
                #writer.add_scalar('Loss/Val', val_loss, batch_idx)
                #writer.add_scalar('Accuracy/Val', val_acc, batch_idx)
            
def evaluate(model, dataloader):
    losses=[]
    correct=0
    n_example=0
    for batch in dataloader:
        features, labels=batch
        predicted=model(features)
        loss=F.mse_loss(predicted, labels)
        losses.append(loss.detach())
        correct+=torch.sum(torch.argmax(predicted,dim=1)==labels)
        n_example+=len(labels)
    avg_loss=np.mean(losses)
    accuracy=correct/n_example
    return avg_loss, accuracy


if __name__ == "__main__":
    model=LinearRegression()
    mydata=AirbnbNightlyPriceImageDataset('tabular_data/clean_tabular_data.csv', 'Price_Night')

    #Create train, validation, test sets
    train_count = int(0.7 *mydata.__len__())
    valid_count = int(0.2*mydata.__len__())
    test_count = mydata.__len__ ()- train_count - valid_count
    train_set, valid_set, test_set = random_split(mydata, (train_count, valid_count, test_count))

    #use DataLoader
    batch_size=128
    train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader=DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(test_set, batch_size=batch_size, shuffle=True)

    train(model, train_loader, val_loader)
    modelling.save_model()


#Create a new folder named neural_networks. Inside it, create a folder called regression.
#Adapt your function called save_model so that it detects whether the model is a PyTorch module, and if so, 
# saves the torch model in a file called model.pt, its hyperparameters in a file called hyperparameters.json, 
# and its performance metrics in a file called metrics.json.

#%%

