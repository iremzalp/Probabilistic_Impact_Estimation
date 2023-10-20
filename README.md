# Probabilistic_Impact_Estimation

Master's Thesis

Author: Irem Özalp

1st Examiner: Prof. Dr. Stefan Lessmann

2nd Examiner: Dr. Alona Zharova

Date: 13.10.2023

![](/forecast_comparison.png)

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

**Full text**: [include a link that points to the full text of your thesis]
*Remark*: a thesis is about research. We believe in the [open science](https://en.wikipedia.org/wiki/Open_science) paradigm. Research results should be available to the public. Therefore, we expect dissertations to be shared publicly. Preferably, you publish your thesis via the [edoc-server of the Humboldt-Universität zu Berlin](https://edoc-info.hu-berlin.de/de/publizieren/andere). However, other sharing options, which ensure permanent availability, are also possible. <br> Exceptions from the default to share the full text of a thesis require the approval of the thesis supervisor.  

## Working with the repo

### Dependencies

Python version 3.8.12 is required to run the project.

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

4. Download the necessary data
The REFIT data files can be accesed from the following link:
https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z

After downloading the data, please copy the csv files to ./data


## Reproducing results

Describe steps how to reproduce your results.

Here are some examples:
- [Paperswithcode](https://github.com/paperswithcode/releasing-research-code)
- [ML Reproducibility Checklist](https://ai.facebook.com/blog/how-the-ai-community-can-get-serious-about-reproducibility/)
- [Simple & clear Example from Paperswithcode](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) (!)
- [Example TensorFlow](https://github.com/NVlabs/selfsupervised-denoising)

### Training code

Does a repository contain a way to train/fit the model(s) described in the paper?

### Evaluation code

Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

### Pretrained models

Does a repository provide free access to pretrained model weights?

## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?


## Project Structure
````
├── README.txt                                    # this readme file                                                
├── forecast_comparison.png                       # figure of forecast comparison         
├── 1_Creating_Recommendationy.ipynb              # Script 1
├── 2_Building_the_DeepAR_Model.ipynb             # Script 2
├── 3_Creating_and_Evaluating_Forecasts.ipynb     # Script 3     
├── 4_Forecast_Analysis.ipynb                     # Script 4
├── Thesis_Classes.py 					                  # All the classes created in the scripts
├── agents.py                                     # agents of the recommendation system (Zharova et al., 2022)
├── helper_functions.py                           # helper functions of the recommendation system (Zharova et al., 2022)
├── helper_functions_thesis.py                    # helper functions created in the scripts
├── requirements.txt
├── data                                                        
│   ├── CLEAN_House1.csv                          # household data (Murray et al., 2017, household 1 to 10)     
│   ├── [...]                                                       
│   ├── CLEAN_House10.csv                                           
│   ├── REFIT_Readme.txt              
│   ├── Day-ahead Prices_201501010000-201601010000.csv  # day-ahead prices provided by ENTSO-E, n.d.        
│   └── pickle_files	                           # pickle files used in the script to speed up the run time
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

