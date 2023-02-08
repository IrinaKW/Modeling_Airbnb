#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt 
import itertools
import joblib
import json
import datetime
import yaml
import os
import re
import glob
import torch
from tabular_data import load_airbnb
import hyperparameteres_grids as hg
#%%

def prep_data_sets(X,y):
    xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.2, random_state=1)
    xtrain, xval, ytrain, yval= train_test_split(xtrain, ytrain, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    return xtrain, ytrain, xval, yval, xtest, ytest

def regression_performance(x, y, model):
    ypred=model.predict(x)
    mse=mean_squared_error(y, ypred)
    score=model.score(x,y)
    #graph_original_predictied(y, ypred)
    return mse, score

def classification_performance(x, y, model):
    # pred = model.predict_proba(x)
    # mse=log_loss(y, pred)
    ypred=model.predict(x)
    score=model.score(x,y)
    f1=f1_score(y, ypred, average='micro')
    precision=precision_score(y, ypred, average='micro')
    recall=recall_score(y, ypred,average='micro')
    return f1, precision,recall, score
    
def graph_original_predictied(y_original, y_predicted):
    x_ax = range(len(y_original))
    plt.plot(x_ax, y_original, label="original")
    plt.plot(x_ax, y_predicted, label="predicted")
    plt.title("Test set and predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()

def custom_tune_regression_model_hyperparameters(model_type, grid_dic, data_sets):
    """manual application: performs a grid search over a reasonable range of hyperparameter values

    Args:
        model_type: model class
        grid_dic (dict): A dictionary of hyperparameter names mapping to a list of values to be tried
        data_sets (list): The training, validation, and test sets

    Returns:
        return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics._
    """
    keys, values = zip(*grid_dic.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_mse=10000
    score=0
    performance_metrics_dict={}
    for param in experiments:
        alpha=param['alpha']
        learning_rate= param['learning_rate']
        loss= param['loss']
        penalty=param['penalty']
        max_iter=param['max_iter']
        model=model_type(alpha=alpha, learning_rate=learning_rate, loss=loss, penalty=penalty, max_iter=max_iter)
        model.fit(data_sets[0], data_sets[1])
        mse, score=regression_performance(data_sets[2], data_sets[3], model)
        if mse<best_mse:
            best_mse=mse
            performance_metrics_dict["mse"] = mse
            performance_metrics_dict["rmse"] = mse**(1/2.0)
            performance_metrics_dict["r2"] = score
            best_hyperparameter_values_dict=param
            best_model=model
    
    return best_model, best_hyperparameter_values_dict, performance_metrics_dict

def tune_regression_model_hyperparameters(model_type, grid_dic, data_sets):
    model=model_type()
    params=[grid_dic]
    gs_estimator = GridSearchCV(estimator=model, param_grid=params)
    gs_estimator.fit(data_sets[0], data_sets[1])
    best_hyperparameter_values_dict=gs_estimator.best_params_
    mse, score=regression_performance(data_sets[2], data_sets[3], gs_estimator)
    performance_metrics_dict={}
    performance_metrics_dict["mse"] = mse
    performance_metrics_dict["rmse"] = mse**(1/2.0)
    performance_metrics_dict["r2"] = score
    return gs_estimator, best_hyperparameter_values_dict, performance_metrics_dict

def tune_classification_model_hyperparameters (model_type, grid_dic, data_sets):
    model=model_type()
    params=[grid_dic]
    gs_estimator = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy')
    gs_estimator.fit(data_sets[0], data_sets[1])
    best_hyperparameter_values_dict=gs_estimator.best_params_
    f1, precision, recall, score=classification_performance(data_sets[2], data_sets[3], gs_estimator)
    performance_metrics_dict={}
    performance_metrics_dict["F1"]=f1
    performance_metrics_dict["precision"] = precision
    performance_metrics_dict["recall"] = recall
    performance_metrics_dict["validation_accuracy"] = score
    return gs_estimator, best_hyperparameter_values_dict, performance_metrics_dict


def evaluate_all_models(model, grid_dic, data_sets, sub_folder):
    model_type=model()
    folder=str(model_type.__class__.__name__).replace('Regressor', '')
    folder=str(model_type.__class__.__name__).replace('Classifier', '')
    new_name=re.findall('[A-Z][^A-Z]*', folder)
    new_name='_'.join(new_name).lower()
    if sub_folder=='regression':
        model, param, metrics=tune_regression_model_hyperparameters(model, grid_dic, data_sets)
    else:
        model, param, metrics=tune_classification_model_hyperparameters(model, grid_dic, data_sets)

    path='models/'+sub_folder+'/'+new_name+'/'
    save_model(model, param, metrics, path)
    

def save_model(model, param, metrics, path='new'):
    if isinstance(model,torch.nn.Module):
        exact_time=str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        path= 'neural_networks/regression/'+exact_time+'/'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        model_filename=path+'model.pt'
        torch.save(model.state_dict(), model_filename)

    else:
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        model_filename = path+'model.joblib'
        joblib.dump(model, model_filename)
    
    hyperparam_filename=path+'hyperparameters.json'
    metrics_filename=path+'metrics.json'

    with open(hyperparam_filename, 'w') as fp:
        json.dump(param,fp)
    with open(metrics_filename, 'w') as fp:    
        json.dump(metrics, fp)

    print ('Model is saved')


def find_best_model(reg_or_class):
    #upload metrics, compares scores (validation r2 for regression or validation accuracy for classification), return the highest
    path = "./models/"+reg_or_class
    metrics_files = glob.glob(path + "/**/metrics.json", recursive = True)
    best_score=0
    for file in metrics_files:
        f = open(str(file))
        metrics_dic = json.load(f)
        if reg_or_class=='regression':
            score=metrics_dic['r2']
        else: 
            score=metrics_dic['validation_accuracy']
        if score>best_score:
            best_score=score
            best_name=str(file).split('/')[-2]
    path='./models/'+reg_or_class+'/'+best_name+'/'
    model=joblib.load(path+'model.joblib')
    with open (path+'hyperparameters.json', 'r') as fp:
        param=json.load(fp)
    with open (path+'metrics.json', 'r') as fp:
        metrics=json.load(fp)

    return model, param, metrics

def get_nn_config():
    d= yaml.full_load(open('nn_config.yaml'))
    return d 

def update_nn_config(params):
    fname = "nn_config.yaml"
    stream = open(fname, 'r')
    data = yaml.safe_load(stream)
    data['optimiser']= params['optimiser']
    data['learning_rate'] = params['learning_rate']
    data['hidden_layer_width'] = params['hidden_layer_width']
    data['model_depth'] = params['model_depth']
    data['epochs'] = params['epochs']
    with open(fname, 'w') as yaml_file:
        yaml_file.write( yaml.dump(data, default_flow_style=False))


if __name__ == "__main__":
    df=pd.read_csv('tabular_data/clean_tabular_data.csv')
    X, y = load_airbnb(df, 'Category')
    label_encoder=preprocessing.LabelEncoder()
    y=label_encoder.fit_transform(y)
    X = scale(X)
    #y = scale(y)
    xtrain, ytrain, xval, yval, xtest, ytest= prep_data_sets(X,y)
    data_sets=[xtrain, ytrain, xval, yval, xtest, ytest]
    regression_model_param_dic={SGDRegressor:hg.sgd_param, 
        DecisionTreeRegressor:hg.decision_tree_param, 
        RandomForestRegressor:hg.random_forest_param, 
        GradientBoostingRegressor:hg.gradient_boost_param}
    for pair in regression_model_param_dic.items():
        model=pair[0]()
        model.fit(data_sets[0], data_sets[1])
        print('train',pair[0], regression_performance(data_sets[0], data_sets[1], model))
        print('validation',pair[0],regression_performance(data_sets[2], data_sets[3], model))
        print('test',pair[0],regression_performance(data_sets[4], data_sets[5], model))


    model, param, metrics = find_best_model('regression')
    print ('best regression model is: ', model)
    print('with metrics', metrics)

    # model, param, metrics = find_best_model('classification')
    # print ('best classification model is: ', model)
    # print('with metrics', metrics)

#%%
