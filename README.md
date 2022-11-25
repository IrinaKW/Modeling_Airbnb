# Modeling_Airbnb
> Modelling Airbnb's property listing dataset 

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- Build a framework that systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team 


## Technologies Used
joblib==1.2.0
matplotlib==3.6.2
numpy==1.23.4
opencv_python==4.6.0.66
pandas==1.5.1
requests==2.25.1
scikit_learn==1.1.3


## Features
-The tabular dataset has the following columns:
    ID: Unique identifier for the listing
    Category: The category of the listing
    Title: The title of the listing
    Description: The description of the listing
    Amenities: The available amenities of the listing
    Location: The location of the listing
    guests: The number of guests that can be accommodated in the listing
    beds: The number of available beds in the listing
    bathrooms: The number of bathrooms in the listing
    Price_Night: The price per night of the listing
    Cleanliness_rate: The cleanliness rating of the listing
    Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
    Location_rate: The rating of the location of the listing
    Check-in_rate: The rating of check-in process given by the host
    Value_rate: The rating of value given by the host
    amenities_count: The number of amenities in the listing
    url: The URL of the listing
    bedrooms: The number of bedrooms in the listing
- Cleaning process of tabular data:
    remove columns with missing values
    modify Description column by turning it into a string
    for "guests", "beds", "bathrooms", and "bedrooms" columns empty entries are replaced with 1.
- Preparation of image data:
    all images are downloaded and processed through usage of ID column of the matching tabular data
    all images are resized, the height of the smallest image is set as the height for all of the other images.
    w by h ratio is preserved
    resized images are saved into the new data/processed_images location
- Model Training to predict the price for the listing per night:
    Use sklearn to compute the key measures of performance for your regression model. That should include the RMSE, and R^2 for both the training and test sets.
    Create a function which performs a grid search over a reasonable range of hyperparameter values manually, followed by CVGrideSearch
    Use decision trees, random forests, and gradient boosting regression models to determine the best model after the CVGrid Search

    

## Screenshots
![Listing Data](./img/listing_table.png)
![Best model outcome](./img/best_model.png)


## Setup
All dependancies are listed in the requirements.txt


## Usage
```
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

```



## Project Status
Project is: _in progress_ 


## Room for Improvement

To do:
- Train the model
- 


## Acknowledgements
- This project was inspired by AiCore program.

## Contact
Created by [@irinakw](irina.k.white@gmail.com) - feel free to contact me!

