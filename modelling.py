#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt 
import itertools
from tabular_data import load_airbnb

def train_model(X,y):
    xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.2, random_state=1)
    xtrain, xval, ytrain, yval= train_test_split(xtrain, ytrain, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    model_sgdr = SGDRegressor()
    model_sgdr.fit(xtrain, ytrain)
    return model_sgdr, xtrain, ytrain, xval, yval, xtest, ytest

def calculate_scores(x, y, model):
    ypred=model.predict(x)
    mse=mean_squared_error(y, ypred)
    score=model.score(x,y)
    #graph_original_predictied(y, ypred)
    return mse, score
    
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
        mse, score=calculate_scores(data_sets[2], data_sets[3], model)
        if mse<best_mse:
            best_mse=mse
            performance_metrics_dict["mse"] = mse
            performance_metrics_dict["rmse"] = val_mse**(1/2.0)
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
    mse, score=calculate_scores(data_sets[2], data_sets[3], gs_estimator)
    performance_metrics_dict={}
    performance_metrics_dict["mse"] = mse
    performance_metrics_dict["rmse"] = val_mse**(1/2.0)
    performance_metrics_dict["r2"] = score
    return gs_estimator, best_hyperparameter_values_dict, performance_metrics_dict


if __name__ == "__main__":
    df=pd.read_csv('tabular_data/clean_tabular_data.csv')
    X, y = load_airbnb(df, 'Price_Night')
    X = scale(X)
    y = scale(y)
    model_sgdr, xtrain, ytrain, xval, yval, xtest, ytest= train_model(X,y)

    space = {}
    space['alpha']= [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    space['learning_rate']= ['constant', 'optimal', 'invscaling', 'adaptive']
    space['loss']= ['squared_error', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive']
    space['penalty']=['l2', 'l1', 'elasticnet']
    space['max_iter']= [750, 1000, 1250, 1500]

    model_sgdr, xtrain, ytrain, xval, yval, xtest, ytest = train_model(X,y)
    data_sets=[xtrain, ytrain, xval, yval, xtest, ytest]

    custom_model, custom_param, custom_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, space, data_sets)
    tune_model, tune_param, tune_metrics = tune_regression_model_hyperparameters(SGDRegressor, space, data_sets)

    print ('Initial Results')
    train_mse, train_score = calculate_scores(xtrain,ytrain,model_sgdr)
    print(f'Train Set: RMSE: {train_mse**(1/2.0)}, score: {train_score}')
    val_mse, val_score = calculate_scores(xval,yval,model_sgdr)
    print(f'Val Set: RMSE: {val_mse**(1/2.0)}, score: {val_score}')
    test_mse, test_score = calculate_scores(xtest,ytest,model_sgdr)
    
    print(f'Test Set: RMSE: {test_mse**(1/2.0)}, score: {test_score}')
    print(f'Manual tuning: {custom_metrics}')
    print(f'GridSearchCV tuning: {tune_metrics}')

#%%

