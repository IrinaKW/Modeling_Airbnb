import numpy as np

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
    'C' : np.logspace(-2,2,5)}

decision_trees_class_param={}

random_forest_class_param={'n_estimators' : list(range(10,101,10)),
    'max_features' : list(range(6,32,5))}


gradient_boosting_class_param={}