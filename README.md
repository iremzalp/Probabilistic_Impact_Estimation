# Probabilistic_Impact_Estimation

Master's Thesis

Author: Irem Özalp

1st Examiner: Prof. Dr. Stefan Lessmann

2nd Examiner: Dr. Alona Zharova

Date: 13.10.2023

![](/forecast_comparison.png)


## Project Structure
````
├── README.txt                                  # this readme file                                                
├── forecast_comparison.png                     # figure of forecast comparison         
├── 1_Creating_Recommendationy.ipynb            # Script 1
├── 2_Building_the_DeepAR_Model.ipynb           # Script 2
├── 3_Creating_and_Evaluating_Forecasts.ipynb   # Script 3     
├── 4_Forecast_Analysis.ipynb                   # Script 4
├── Thesis_Classes.py 					                # All the classes created in the scripts
├── agents.py                                   # agents of the recommendation system (Zharova et al., 2022)
├── helper_functions.py                         # helper functions of the recommendation system (Zharova et al., 2022)
├── helper_functions_thesis.py                  # helper functions created in the scripts
├── requirements.txt
├── data                                                        
│   ├── CLEAN_House1.csv                        # household data (Murray et al., 2017, household 1 to 10)     
│   ├── [...]                                                       
│   ├── CLEAN_House10.csv                                           
│   ├── REFIT_Readme.txt              
│   ├── Day-ahead Prices_201501010000-201601010000.csv  # day-ahead prices provided by ENTSO-E, n.d.        
│   └── pickle_files	                         # pickle files used in the script to speed up the run time
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

