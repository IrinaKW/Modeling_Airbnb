#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt 
import itertools
import joblib
import json
import os
import re
import glob
from tabular_data import load_airbnb

#Hyperparameteres grids
sgd_param = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'loss': ['squared_error', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
    'penalty' : ['l2', 'l1', 'elasticnet'],
    'max_iter' : [750, 1000, 1250, 1500]}

decision_tree_param={"splitter":["best","random"],
    "max_depth" : [1,3,5,7,9,11,12],
    "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5],
    "max_features":[1.0, "log2","sqrt",None],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90]}

random_forest_param={'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': [1,0, 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

gradient_boost_param={'n_estimators':[500,1000,2000],
    'learning_rate':[.001,0.01,.1],
    'max_depth':[1,2,4],
    'subsample':[.5,.75,1],
    'random_state':[1]}

logistic_regression_param ={'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-2,2,5),
    'max_iter' : [1000, 1500]}



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


def evaluate_all_models(model, grid_dic, data_sets):
    model_type=model()
    folder=str(model_type.__class__.__name__).replace('Regressor', '')
    new_name=re.findall('[A-Z][^A-Z]*', folder)
    new_name='_'.join(new_name).lower()
    model, param, metrics=tune_regression_model_hyperparameters(model, grid_dic, data_sets)
    path='models/regression/'+new_name+'/'
    save_model(model, param, metrics, path)
    

def save_model(model, param, metrics, path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    model_filename = path+'model.joblib'
    hyperparam_filename=path+'hyperparameters.json'
    metrics_filename=path+'metrics.json'
    joblib.dump(model, model_filename)
    with open(hyperparam_filename, 'w') as fp:
        json.dump(param,fp)
    with open(metrics_filename, 'w') as fp:    
        json.dump(metrics, fp)
    print ('Model is saved')


def find_best_model():
    #upload metrics, compare r2, take the highest
    path = "./models/regression"
    metrics_files = glob.glob(path + "/**/metrics.json", recursive = True)
    best_r2=0
    for file in metrics_files:
        f = open(str(file))
        metrics_dic = json.load(f)
        r2=metrics_dic['r2']
        if r2>best_r2:
            best_r2=r2
            best_name=str(file).split('/')[-2]
    path='./models/regression/'+best_name+'/'
    model=joblib.load(path+'model.joblib')
    with open (path+'hyperparameters.json', 'r') as fp:
        param=json.load(fp)
    with open (path+'metrics.json', 'r') as fp:
        metrics=json.load(fp)

    return model, param, metrics



if __name__ == "__main__":
    df=pd.read_csv('tabular_data/clean_tabular_data.csv')
    X, y = load_airbnb(df, 'Category')
    label_encoder=preprocessing.LabelEncoder()
    y=label_encoder.fit_transform(y)
    X = scale(X)
    #y = scale(y)
    xtrain, ytrain, xval, yval, xtest, ytest= prep_data_sets(X,y)
    data_sets=[xtrain, ytrain, xval, yval, xtest, ytest]
    # model_param_dic={SGDRegressor:sgd_param, 
    #     DecisionTreeRegressor:decision_tree_param, 
    #     RandomForestRegressor:random_forest_param, 
    #     GradientBoostingRegressor:gradient_boost_param}
    #model, param, metrics= find_best_model()

    model=LogisticRegression(random_state=1, C=1.0, max_iter=1000, penalty='l2', solver='lbfgs')
    model.fit(data_sets[0], data_sets[1])
    print( classification_performance(data_sets[2], data_sets[3], model))

    a, b, c =tune_classification_model_hyperparameters(LogisticRegression, logistic_regression_param, data_sets)
    print (b, c)

 


# %%
model.get_config()


