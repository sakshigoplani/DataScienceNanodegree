# New York Airbnb Data Analysis

## Table of contents

- [Project motivation](#project-motivation)
- [Installation](#installation)
- [File descriptions](#file-descriptions)
- [Results](#results)

## Project Motivation

Based on Cross-Industry Standard Process of Data Mining (CRISP-DM), New York Airbnb dataset was collected and investigated.
Three business questions were asked and answered:

- For different property types, what are average prices for different neighbourhoods? 
- What factors contribute towards a host rating? What aspects could possibly help a host get a better rating?  
- How can a host be entitled as a Superhost? How can they set a competitive rental price for their listing? How can a host maintain good review rating for their listing?

## Installation

The code should run with no issues using Python versions 3.* with the libararies as follows.
- numpy, pandas, matplotlib, seaborn, datetime, re, math, sklearn

Two quick start options are available:
- [Download the latest release.](https://github.com/sakshigoplani/DataScienceNanodegree/tree/master/Introduction_To_Data_Science_Project_1)
- Clone the repo: `https://github.com/sakshigoplani/DataScienceNanodegree.git`

## File descriptions

Within the download you'll find the following directories and files.

```text
Introduction_To_Data_Science_Project_1/
├── new_york_airbnb.ipynb
├── listings.csv
├── README.md
└── .ipynb_checkpoints/
    └── new_york_airbnb-checkpoint.ipynb
```

- new_york_airbnb.ipynb ==> Notebook to investigate Airbnb trends in April 2020 in New York.
- listings.csv         ==> Information of listings in New York.

## Results
Results and discussions were published on Medium: https://medium.com/@sakshigoplani9/torque-your-airbnb-spin-in-new-york-557c5902f5de

- In general, the price trend for different types of neighbourhoods: Manhattan > Brooklyn > Queens > Bronx > Staten Island
- In general, the price trend for different types of properties: Hotel Room > Entire Home/Apartment > Private Room > Shared Room
- It is cheaper to stay in an Entire Home/Apartment property than a Hotel Room in New York
- In Bronx, Private Room and Shared Room cost just about the same
- All review scores are highly correlated with each other as it was expected
- Reviews and Availability show negative correlation which was expected as properties with higher review rating are expected to be heavily booked
- It was expected that host_response_time and host_response_rate would have a positive corelation with the review_score_communication but was found otherwise
- It was also expected that host_reponse_rate and host_repsonse_time will have a negative correlation with availability but turned out to be otherwise
- It was unexpected to see that number of years as host has low negative correlation with response time, response rate and acceptance rate
- Linear Regression can be a good enough model for predicting review_score_rating based on the feature set
- The models comply with the findings from heatmap correlation
- Exploration of other models is required in future to better predict host_is_superhost and price features
