# Probabilistic_Impact_Estimation

Master's Thesis

Author: Irem Özalp

1st Examiner: Prof. Dr. Stefan Lessmann

2nd Examiner: Dr. Alona Zharova

Date: 13.10.2023

![](/forecast_comparison.png)

## Summary

Buildings in the EU account for 42% of energy consumption. This makes energy efficiency in buildings a crucial aspect of achieving climate neutrality. Smart home systems play a pivotal role in this transition and offer efficiency optimization through load shifting, which can help transition to energy-neutral buildings. Evaluating the impact of an intervention is challenging due to the absence of data from both treated and untreated scenarios concurrently. To address this, we employ the Deep Probabilistic Counterfactual Prediction Net (DeepProbCP) framework, which employs a global probabilistic forecasting model to estimate the probabilistic impact of an intervention across multiple treated units. We simulate the implementation of a smart home recommendation system and use the DeepAR model to generate counterfactual outcomes, facilitating a controlled evaluation. The results show that the recommendation system has a substantial cumulative impact on cost reduction, and this effect is further confirmed through the use of counterfactual time series generated by the DeepAR model.

**Keywords:** recommendation system, load shifting, counterfactual analysis, global forecasting, probabilistic forecasting, neural networks, deep learning

## Working with the repo

### Dependencies

Python version 3.8.12 was used to build the project.

### Setup

1. Clone this repository

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Installing the required MXNet package, version 1.9.1, on a MacBook with an M1 chip processor can be challenging. However, this issue is not encountered when using an M2 chip or Intel-based Macs. While it is possible to build the package from source (as described in the MXNet documentation: https://mxnet.apache.org/versions/1.9.1/get_started/osx_setup.html), this process is somewhat unstable.

In contrast, I faced no issues while installing all the necessary dependencies on a Linux system.

Although I did not have the opportunity to test it on a Microsoft system, there are no issues reported on the internet regarding MXNet installation for this platform.

4. Download the necessary data
The REFIT data files can be accesed from the following link:
https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z
Or you can go to https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned and download the data by clicking "CLEAN_REFIT_081116.7z".


After downloading the data, please copy the csv files to ./data

5. Unzip the pickle_files folder, which contains all the pickle files used in the notebooks
This folder needs to be inside ./data 

## Reproducing results

1. The first Notebook 1_Creating_Recommendations.ipynb creates recommendations for each shiftable device in each household using the Recommendation Agent of the Smart Home Recommendation System, and creates simulated treatment data by introducing the recommendation acceptance algorithm which accepts recommendations by shifting usage loads.
2. The second Notebook 2_Building_the_DeepAR_Model.ipynb creates essential datasets for training the DeepAR forecasting model, establishes a hyperparameter tuning framework, and conducts model training
3. The third Notebook 3_Creating_and_Evaluating_Forecasts.ipynb introduces the evaluation metrics employed to assess the performance of the binary forecasting model and generates forecasts using the trained moded.
4. The fourth and last Notebook 4_Forecast_Analyses creates counterfactual load profiles by inserting device usage data to binary forecasts. Then it compares the treatment, control and counterfactual time series and conducts hypothesis tests to prove a significant decrease in costs after the implementation of the recommendation system.


## Project Structure
````
├── README.txt                                    # this readme file                                                
├── forecast_comparison.png                       # figure of forecast comparison         
├── 1_Creating_Recommendations.ipynb              # Notebook 1
├── 2_Building_the_DeepAR_Model.ipynb             # Notebook 2
├── 3_Creating_and_Evaluating_Forecasts.ipynb     # Notebook 3     
├── 4_Forecast_Analysis.ipynb                     # Notebook 4
├── Thesis_Classes.py 					                  # All the classes used in the Notebooks
├── agents.py                                     # agents of the recommendation system (Zharova et al., 2022)
├── helper_functions.py                           # helper functions of the recommendation system (Zharova et al., 2022)
├── helper_functions_thesis.py                    # helper functions used in the Notebooks
├── requirements.txt
├── data                                                        
│   ├── CLEAN_House1.csv                          # household data (Murray et al., 2017, household 1 to 10)     
│   ├── [...]                                                       
│   ├── CLEAN_House10.csv                                           
│   ├── REFIT_Readme.txt              
│   ├── Day-ahead Prices_201501010000-201601010000.csv  # day-ahead prices provided by ENTSO-E, n.d.        
│   └── pickle_files	                           # pickle files used in the Notebooks to speed up the run time
│        ├── activity_dict.pickle
│        ├── deepar_model.pickle
│        ├── forecast_samples_rescaled.pickle
│        ├── forecast_samples_test.pickle
│        ├── forecast_samples_val.pickle
│        ├── load_dict.pickle
│        ├── load_post_recommendation_usage_added.pickle
│        ├── load_post_recommendation.pickle
│        ├── recommendations_dict_test.pickle
│        ├── recommendations_dict.pickle
│        └── usage_dict.pickle
````

