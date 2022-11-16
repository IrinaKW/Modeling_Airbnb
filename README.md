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
- opencv_python==4.6.0.66
- pandas==1.5.1
- requests==2.25.1


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
    

## Screenshots
![Listing Data](./img/listing_table.png)


## Setup
All dependancies are listed in the requirements.txt


## Usage



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

