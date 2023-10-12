import pandas as pd
import matplotlib.pyplot as plt

class Helper_Functions_Thesis:
    
    @staticmethod
    def open_pickle_file(file_path, file_name):
        import os
        import pickle
        
        with open(os.path.join(file_path, file_name), 'rb') as handle:
            return pickle.load(handle)
        
    @staticmethod
    def create_shiftable_devices_dict(devices):
        
        shiftable_devices_list = {}

        for dev in devices.values():
            household_id = dev['hh']
            device_name = dev['dev_name']

            if household_id not in shiftable_devices_list:
                shiftable_devices_list[household_id] = []

            shiftable_devices_list[household_id].append(device_name)
            00
        return shiftable_devices_list
    
    @staticmethod
    def create_date_list_daily(start, length):
        from datetime import datetime
        
        date_list_daily = pd.date_range(datetime.strptime(start,'%Y-%m-%d'),
                         periods=length,
                         freq='d').date
        
        return date_list_daily
        
    def create_date_list_hourly(start, length):
        from datetime import datetime
        
        date_list_hourly = pd.date_range(datetime.strptime(start,'%Y-%m-%d'),
                         periods=length*24,
                         freq='h')
        
        return date_list_hourly

    def create_target_values(recommendation_start, recommendation_phase, dataset_dict, binary_usage=True):
        import pandas as pd
        target = {}
        
        date_list_hourly = Helper_Functions_Thesis.create_date_list_hourly(recommendation_start, recommendation_phase)
        
        for dev, data in dataset_dict.items():
            usage_data = data['usage_bin'] if binary_usage else data['usage']
            target[dev] = pd.DataFrame({'usage': usage_data.loc[date_list_hourly]})
        
        return target

    def get_timespan(df, start, timedelta_params):

        start = pd.to_datetime(start) if type(start) != type(pd.to_datetime('1970-01-01')) else start 
        end = start + pd.Timedelta(**timedelta_params)
        return df[start:end]

    def create_recommendations(date_list, hh, recommendation_agent_dict, activity_prob_threshold=0.5, usage_prob_threshold=0.6):
        import pandas as pd
        
        recommendations_table = pd.DataFrame(columns=["recommendation_date", "device", "recommendation"])

        for date in date_list:
            print(date)
            recommendation_df = recommendation_agent_dict[hh].pipeline(
                date=str(date),
                activity_prob_threshold=activity_prob_threshold,
                usage_prob_threshold=usage_prob_threshold,
                evaluation=False,
                weather_sel=True
            )[
                ["recommendation_date", "device", "recommendation"]
            ]

            recommendations_table = pd.concat([recommendations_table, recommendation_df])

            recommendations_table.recommendation = pd.to_numeric(recommendations_table.recommendation,
                                                                            downcast='integer',
                                                                            errors='coerce')
        
        return recommendations_table

    def optimize_deepar_hyperparameters(
        prediction_length, freq, start_dataset, start_validation, recommendation_length, devices, dataset_list, n_trials=30, seed = 0
    ):
        import numpy as np
        import time
        import optuna
        import random
        from Thesis_Classes import DeepAR_Tuning_Objective
        
        random.seed(seed)
        np.random.seed(seed)

        best_model_deepar = None
        predictor = None

        def callback(study, trial):
            nonlocal best_model_deepar
            if study.best_trial == trial:
                best_model_deepar = predictor

        time_start = time.time()
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(
            DeepAR_Tuning_Objective(
                prediction_length, freq, start_dataset, start_validation, recommendation_length, devices, dataset_list
            ),
            n_trials=n_trials,
            callbacks=[callback]
        )

        print('Number of finished trials: {}'.format(len(study.trials)))

        print('Best trial:')
        trial_p = study.best_trial

        print('  Value: {}'.format(trial_p.value))

        print('  Params: ')
        for key, value in trial_p.params.items():
            print('    {}: {}'.format(key, value))
        print('Elapsed time:', time.time() - time_start)

        return best_model_deepar
    
    @staticmethod
    def add_load_profile(devices, recommendation_start, load_dict):
        from agents import Load_Agent
        from helper_functions_thesis import Helper_Functions_Thesis
        from copy import deepcopy
        
        devices_with_loads = deepcopy(devices)
            
        shiftable_devices_dict = Helper_Functions_Thesis.create_shiftable_devices_dict(devices_with_loads)

        load_profiles_dict = {}
        for hh in shiftable_devices_dict.keys():
            
            # Load profile calculated from data start until chosen date (recommendation start)
            load_agent = Load_Agent(hh[2:])
            load_profiles_dict[hh] = load_agent.pipeline(load_dict[hh], recommendation_start, shiftable_devices_dict[hh])
                
        for dev in devices_with_loads.keys():
            
            hh = devices_with_loads[dev]['hh']
            dev_name = devices_with_loads[dev]['dev_name']
            devices_with_loads[dev]['load_profile'] = load_profiles_dict[hh].loc[dev_name]
            
        return devices_with_loads
            


