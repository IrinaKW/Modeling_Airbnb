#%%
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import R2Score
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from dataset import AirbnbNightlyPriceImageDataset
import modelling
import hyperparameteres_grids as hg
import itertools

class LinearRegression(torch.nn.Module):
    def __init__(self, config)->None:
        super().__init__()
        model_depth=config['model_depth']
        width=config['hidden_layer_width']
        layers=[]
        #initialise parameters
        for hidden_layer in range(model_depth):
            layers.append(nn.Linear(width,width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width,1))
        self.layers=torch.nn.Sequential(*layers)

#        self.double()

    def forward(self, features):
        #use layers to process features
        return self.layers(features)


def train(model, data_loader, config):
    training_start_time = time.time()
    epochs=config['epochs']
    lr=config['learning_rate']
    optimiser_class = config['optimiser']
    optimiser_instance = getattr(torch.optim, optimiser_class)
    optimiser = optimiser_instance(model.parameters(), lr=lr)
    loss_train=[]
    r2_score=[]
    #writer=SummaryWriter()
    
    batch_idx=0
    
    for epoch in range(epochs):
        for batch in data_loader:
            features, labels = batch
            features=features.type(torch.float32)
            labels=labels.unsqueeze(1)
            predicted=model(features)
            loss=criterion(predicted, labels.float())
            loss.backward()
            print('Loss', loss.item())
            
            loss_train.append(loss.item)
            r2=criterion2(predicted, labels.float())
            r2_score.append(r2.item())

            #optimization step
            optimiser.step()
            #reset grad params for next backward step
            optimiser.zero_grad()
            #writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx+=1
            #prediction.append(prediction_time)
        
        if batch_idx%epochs==0:
            training_time=time.time()-training_start_time
            avg_loss_train=np.mean(loss_train)
            r2_train=np.mean(r2_score)
            RMSE_loss_train=torch.square(avg_loss_train)
            RMSE_loss_val, r2_val = evaluate(model, val_loader)
            RMSE_loss_test, r2_test = evaluate(model, test_loader)
            #writer.add_scalar('Loss/Val', val_loss, batch_idx)
            #writer.add_scalar('Accuracy/Val', val_acc, batch_idx)

    metrics_dic={}
    number_of_predictions = epochs * len(train_set)
    inference_latency = training_time / number_of_predictions
    metrics_dic['training_time']=training_time
    metrics_dic['inference_latency']=inference_latency
    RMSE_loss=[RMSE_loss_train, RMSE_loss_val, RMSE_loss_test]
    R_squared=[r2_train, r2_val, r2_test]
    metrics_dic["RMSE_loss"] = [loss.item() for loss in RMSE_loss]
    metrics_dic["R_squared"] = [score.item() for score in R_squared]
    return metrics_dic

def evaluate(model, dataloader):
    losses=[]
    r2=[]
    for batch in dataloader:
        features, labels = batch
        features=features.type(torch.float32)
        labels=labels.unsqueeze(1)
        predicted=model(features)
        loss=criterion(predicted, labels.float())
        losses.append(loss.detach())
        r2=criterion2(predicted, labels.float())
        r2.append(r2.detach())
        n_examples+=len(labels)
    avg_loss=np.mean(losses)
    avg_r2=np.mean(r2)
    RMSE_score=torch.square(avg_loss)
    return RMSE_score, avg_r2


def tune_nn_model(model, RMSE_loss_val,grid_dic):
    keys, values = zip(*grid_dic.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_RMSE_loss=RMSE_loss_val
    for params in experiments:
        modelling.update_nn_config(params)
        training_time = train(model, train_loader, config=config)
        metrics_dic = evaluate(model, training_time, params['epochs'])
        if metrics_dic['RMSE_loss'][1] < best_RMSE_loss: 
            best_RMSE_loss=metrics_dic['RMSE_loss'][1]
            best_model=model
            best_config=params
            best_metrics_dic=metrics_dic
    return best_model, best_config, best_metrics_dic

if __name__ == "__main__":
    config=modelling.get_nn_config()
    model=LinearRegression(config)
    criterion=torch.nn.MSELoss()
    criterion2=R2Score()
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

    metrics_dic = train(model, train_loader, config=config)
    print('First Model', metrics_dic)
    RMSE_loss_val=metrics_dic['RMSE_loss'][1]

    #best_model, best_config, best_metrics_dic=tune_nn_model(model, RMSE_loss_val, hg.nn_model_param)
    #print('Best Model', metrics_dic)
    #modelling.save_model(best_model, best_config, best_metrics_dic)

# %%
