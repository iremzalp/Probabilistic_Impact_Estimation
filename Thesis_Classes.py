import mxnet as mx
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from gluonts.mx.trainer.callback import Callback
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import Dataset
from gluonts.mx import DeepAREstimator
from gluonts.mx.distribution import CategoricalOutput
from gluonts.mx import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

class Activity_Usage_Threshold_Search:
    '''
    A class for streamlining the process of determining optimal thresholds for activity and usage.
    Parameters:
    ----------
    activity_dict : 
        A dictionary containing the output dataframes from the Activity Agent for each household.
    usage_dict :
        A dictionary containing the output dataframes from the Usage Agent for each household.
    load_dict :
        A dictionary containing the output dataframes from the Load Agent for each household.
    devices :
        A dictionary containing device-specific information.
    price_df : 
        A DataFrame with hourly price data in GBP per megawatt-hour.
    recommendation_start :
        The start date of the recommendation validation period in 'yyyy-mm-dd' format.
    recommendation_length : 
        The length of the validation period in days.
    model_type : 
        The machine learning model type used for predicting availability and usage probabilities.
    threshold_dict : 
        A dictionary defining the availability and usage threshold parameter space.
    '''
    def __init__(
        self,
        activity_dict: Dict[str, pd.DataFrame],
        usage_dict: Dict[str, pd.DataFrame],
        load_dict: Dict[str, pd.DataFrame],
        devices: Dict[str, Any],
        price_df: pd.DataFrame,
        recommendation_start: str,
        recommendation_length: int,
        model_type: str = "random forest",
        threshold_dict: Dict[str, Any] = {
            "activity": {"start": 0.0, "end": 0.95, "granularity": 0.05},
            "usage": {"start": 0.0, "end": 0.95, "granularity": 0.05},
        },
    ):
        import pandas as pd
        from helper_functions_thesis import Helper_Functions_Thesis
        self.activity_dict = {dev: df.copy() for dev, df in activity_dict.items()}
        self.usage_dict = {dev: df.copy() for dev, df in usage_dict.items()}
        self.load_dict = {dev: df.copy() for dev, df in load_dict.items()}
        self.price_df = price_df
        self.devices = devices
        self.model_type = model_type
        self.threshold_dict = threshold_dict
        
        self.date_list = Helper_Functions_Thesis.create_date_list_daily(
            recommendation_start, recommendation_length
        )
        
        self.shiftable_devices_dict = Helper_Functions_Thesis.create_shiftable_devices_dict(
            self.devices
        )

    def initialize_recommendation_agent(self):
        from agents import X_Recommendation_Agent
        recommendation_agent_dict = {
            hh: X_Recommendation_Agent(
                self.activity_dict.get(hh),
                self.usage_dict.get(hh),
                self.load_dict.get(hh),
                self.price_df,
                self.shiftable_devices_dict.get(hh),
                model_type=self.model_type
            )
            for hh in self.shiftable_devices_dict.keys()
        }
        return recommendation_agent_dict

    def generate_probability_list(self, params):
        import numpy as np
        start = params['start']
        end = params['end']
        granularity = params['granularity']
        
        prob_list = [round(prob, 2) for prob in np.arange(start, end, granularity)]
        
        return prob_list

    def create_threshold_values(self):
        usage_params = self.threshold_dict['usage']
        activity_params = self.threshold_dict['activity']
        
        usage_prob_list = self.generate_probability_list(usage_params)
        activity_prob_list = self.generate_probability_list(activity_params)

        return usage_prob_list, activity_prob_list

    def grid_search_probability_thresholds(self, usage_prob_list, activity_prob_list, recommendation_agent_dict):
        from helper_functions_thesis import Helper_Functions_Thesis
        recommendations_dict = {}
        
        for hh in recommendation_agent_dict.keys():
            recommendations_dict[hh] = {}

            for i in activity_prob_list:
                for j in usage_prob_list:
                    print(hh,i,j)
                    recommendations_dict[hh][f"{i}_{j}"] = Helper_Functions_Thesis.create_recommendations(
                        date_list=self.date_list,
                        hh=hh,
                        recommendation_agent_dict=recommendation_agent_dict,
                        activity_prob_threshold=i,
                        usage_prob_threshold=j
                    )

        return recommendations_dict

    def create_accuracy_grid(self, usage_prob_list, activity_prob_list, recommendations_dict):
        accuracy_grid = {}

        for hh, device in self.shiftable_devices_dict.items():
            accuracy_grid[hh] = pd.DataFrame(index=activity_prob_list, columns=usage_prob_list)
            accuracy_grid[hh].index.name = 'Usage Probabilities'
            accuracy_grid[hh].columns.name = 'Activity Probabilities'

            for a in activity_prob_list:
                for u in usage_prob_list:
                    cols = [x + '_usage' for x in device]
                    
                    # Transform recommendaiton data to binary, daily usage patterns
                    recommended_usage_pattern = recommendations_dict[hh][f"{a}_{u}"].copy()
                    recommended_usage_pattern['recommendation'] = recommended_usage_pattern['recommendation'].apply(
                        lambda x: 1 if pd.notnull(x) else 0
                    )

                    recommended_usage_pattern = recommended_usage_pattern.set_index('recommendation_date')
                    recommended_usage_pattern = (
                        recommended_usage_pattern.groupby([recommended_usage_pattern.index, 'device'])
                        ['recommendation']
                        .aggregate('first')
                        .unstack()[device]
                    )
                    
                    recommended_usage_pattern.columns.name = None
                    
                    # Retrieve true device usage data
                    true_usage_pattern = self.usage_dict[hh][cols].loc[self.date_list]
                    true_usage_pattern.columns = device
                    true_usage_pattern.index = recommended_usage_pattern.index

                    # Compare recommended vs. true usage patterns and calculate differences
                    usage_comparison = recommended_usage_pattern.compare(true_usage_pattern)

                    if usage_comparison.empty:
                        accuracy_grid[hh].at[u, a] = 0
                    else:
                        difference = usage_comparison.xs('self', axis=1, level=1, drop_level=False).count().sum()
                        accuracy_grid[hh].at[u, a] = difference

            accuracy_grid[hh] = accuracy_grid[hh].astype(int)

        return accuracy_grid

    def return_best_recommendations(self, accuracy_grid, daily_recommendations_dict):
        threshold_dict = {}

        for hh in self.shiftable_devices_dict.keys():
            min_value = float("inf")
            min_index = None

            for col in accuracy_grid[hh].columns:
                for index, value in enumerate(accuracy_grid[hh][col]):
                    if value < min_value:
                        min_value = value
                        min_index = (index, col)

            usage_threshold = accuracy_grid[hh].index[min_index[0]]
            activity_threshold = min_index[1]

            threshold_dict[hh] = {
                'usage': usage_threshold,
                'activity': activity_threshold
            }

        return threshold_dict

    def pipeline(self):
    
        recommendation_agent_dict = self.initialize_recommendation_agent()
        
        usage_prob_list, activity_prob_list = self.create_threshold_values()
            
        daily_recommendations_dict = self.grid_search_probability_thresholds(
            usage_prob_list,
            activity_prob_list,
            recommendation_agent_dict)
            
        accuracy_grid = self.create_accuracy_grid(
            usage_prob_list,
            activity_prob_list,
            daily_recommendations_dict)
            
        best_thresholds = self.return_best_recommendations(
            accuracy_grid,
            daily_recommendations_dict)
            
        return best_thresholds

class Create_Accept_Recommendations:
    '''
    A class for generating recommendations and simulating device usage patterns through the test phase
    Parameters:
    ----------
    recommendation_agent_dict : 
        A dictionary containing household recommendation agents to create recommendations
    devices :
        A dictionary containing device-specific information
    recommendation_start :
        The start date of the recommendation validation period in 'yyyy-mm-dd' format
    recommendation_length : 
        The length of the recommendation validation period in days
    threshold_dict : 
        A dictionary defining the availability and usage threshold parameter space
    load_dict :
        A dictionary containing the output dataframes from the Load Agent for each household
    '''
    def __init__(
        self,
        recommendation_agent_dict,
        devices: Dict[str, Any],
        recommendation_start: str,
        recommendation_length: int,
        best_thresholds,
        load_dict: Dict[str, pd.DataFrame],
    ):
        import pandas as pd
        from helper_functions_thesis import Helper_Functions_Thesis
        self.recommendation_agent_dict = recommendation_agent_dict
        self.devices = devices
        self.best_thresholds = best_thresholds
        self.recommendation_start = recommendation_start
        self.recommendation_length = recommendation_length
        self.shiftable_devices_dict = Helper_Functions_Thesis.create_shiftable_devices_dict(devices)
        self.date_list = Helper_Functions_Thesis.create_date_list_daily(
            recommendation_start, recommendation_length
        )
        self.load_dict = {dev: df.copy() for dev, df in load_dict.items()}

    def generate_daily_recommendations(self):
        from agents import X_Recommendation_Agent
        from helper_functions_thesis import Helper_Functions_Thesis
        best_daily_recommendations_dict = {}

        for hh in self.shiftable_devices_dict.keys():
            activity_threshold = self.best_thresholds[hh]['activity']
            usage_threshold = self.best_thresholds[hh]['usage']
            print(f"Household: {hh}, Activity Threshold: {activity_threshold}, Usage Threshold: {usage_threshold}")

            best_daily_recommendations_dict[hh] = Helper_Functions_Thesis.create_recommendations(
                date_list=self.date_list,
                hh=hh,
                recommendation_agent_dict=self.recommendation_agent_dict,
                activity_prob_threshold=activity_threshold,
                usage_prob_threshold=usage_threshold
            )
        
        return best_daily_recommendations_dict

    def update_device_information(self, recommendations_dict):
        from agents import Load_Agent
        
        for dev, device_information in self.devices.items():
            hh = device_information['hh']
            dev_name = device_information['dev_name']

            load_agent = Load_Agent(hh[2:])
            load_profiles = load_agent.pipeline(
                self.load_dict[hh],
                self.recommendation_start,
                self.shiftable_devices_dict[hh]) 

            device_information['usage'] = self.load_dict[hh][dev_name]
            device_information['load_profile'] = load_profiles.loc[dev_name]
            device_information['recommendation'] = recommendations_dict[hh][recommendations_dict[hh]['device'] == dev_name]

    def accept_recommendations(self, is_usage_added=False, is_info_displayed=False):
        from datetime import timedelta
        import numpy as np
        import pandas as pd
        
        load_post_recommendation = {}
        
        # Iterate over each device
        for dev in self.devices.keys():
            device_usage = self.devices[dev]['usage'].copy()
            load_profile = self.devices[dev]['load_profile'].copy()
            recommendation = self.devices[dev]['recommendation'].copy()
        
            device_usage = device_usage[str(self.date_list[0]):str(self.date_list[-1])]
            output = device_usage.copy()

            # Iterate over each date in the specified date list
            for date in self.date_list:

                date_before = date - timedelta(days=1)
                date = str(date)

                daily_usage = device_usage.loc[date].copy()
                daily_usage_extended = device_usage.loc[date:][:48].copy()
                
                # Retrieve the recommended start hour
                recommended_start_hour = recommendation[(recommendation.recommendation_date == date)]['recommendation'].iloc[0]
                
                if np.isnan(recommended_start_hour):
                    print(date, "No recommendation") if is_info_displayed else None
                    continue

                recommended_start_hour = int(recommended_start_hour)
                
                # Extract the usage start hours
                shifted_daily_usage = daily_usage.shift()

                if date != str(self.date_list[0]):
                    shifted_daily_usage.iloc[0] = device_usage.loc[str(date_before)].iloc[-1]

                start_hours_usage = daily_usage[
                    (daily_usage.values > 0) & (shifted_daily_usage.values == 0)
                ].index.hour.tolist()

                if start_hours_usage == []:
                    if is_usage_added:
                        print(date, "No usage, Insert average load to the recommended usage hour:", recommended_start_hour) if is_info_displayed else None
                        daily_usage_extended[recommended_start_hour:recommended_start_hour+3] = load_profile[0:3]
                        daily_usage_slice = daily_usage_extended[recommended_start_hour:recommended_start_hour+3]
                        output = daily_usage_slice.combine_first(output)
                        continue
                    else:
                        print(date, "No usage") if is_info_displayed else None
                        continue

                # Determine the closest start hour to the recommended hour
                distance_of_closest_start = int(min(np.full(shape=len(start_hours_usage), fill_value=recommended_start_hour) - start_hours_usage, key=abs, default=0))
                shifted_usage_start = int(recommended_start_hour - distance_of_closest_start)
                
                if (len(start_hours_usage) != 0) and (shifted_usage_start == recommended_start_hour):
                    print(date, "Device usage starts at the recommended hour") if is_info_displayed else None
                    continue

                # Calculate the length of device usage
                usage_length = (
                    ((daily_usage_extended.iloc[shifted_usage_start:] == 0).idxmax() - daily_usage_extended.index[shifted_usage_start])
                        // pd.Timedelta(minutes=60))

                print(date, "Shift load starting at", shifted_usage_start, "to the recommended hour", recommended_start_hour) if is_info_displayed else None

                usage_section = daily_usage_extended.iloc[shifted_usage_start:shifted_usage_start + usage_length]
                usage_shifted = usage_section.shift(periods=distance_of_closest_start, freq="60T")
                output[date:].iloc[shifted_usage_start:shifted_usage_start + usage_length] = 0
                output = usage_shifted.combine_first(output)

            # Store the results for the current device
            output = pd.DataFrame({'usage': output})
            load_post_recommendation[dev] = output
        
        return load_post_recommendation

    def pipeline(self):
        recommendations_test_dict = self.generate_daily_recommendations()

        self.update_device_information(
            recommendations_test_dict)
        
        load_post_recommendation = self.accept_recommendations(
            is_usage_added=False,
            is_info_displayed=False)
        
        load_post_recommendation_usage_added = self.accept_recommendations(
            is_usage_added=True,
            is_info_displayed=False)
        
        return load_post_recommendation, load_post_recommendation_usage_added

class Create_Dataset:
    '''
    A class for constructing datasets for each device with all the necessary information to train
    the forecasting model.
    Parameters:
    ----------
    start_dataset:
        A string ('yyyy-mm-dd') to determine the start of the dataset.
    end_dataset:
        A string ('yyyy-mm-dd') to determine the end of the dataset.
    devices:
        A dictionary containing device-specific information.
    load_dict:
        A dictionary containing the output dataframes from the Load Agent for each household.
    usage_dict:
        A dictionary containing the output dataframes from the Usage Agent for each household.
    activity_dict:
        A dictionary containing the output dataframes from the Activity Agent for each household.
    fill_na:
        Bool, controls if missing values should be filled with zeros.
    '''

    def __init__(
        self,
        start_dataset: str,
        end_dataset: str,
        devices: Dict[str, Any],
        load_dict: Dict[str, pd.DataFrame],
        usage_dict: Dict[str, pd.DataFrame],
        activity_dict: Dict[str, pd.DataFrame],
        fill_na = True,
    ):
        from helper_functions_thesis import Helper_Functions_Thesis
        self.start_dataset = start_dataset
        self.end_dataset = end_dataset
        self.devices = devices
        self.load_dict = load_dict
        self.usage_dict = usage_dict
        self.activity_dict = activity_dict
        self.hh_list: List[str] = list(Helper_Functions_Thesis.create_shiftable_devices_dict(self.devices).keys())
        self.fill_na = fill_na

    def slice_datasets(self) -> None:
        import pandas as pd

        for hh in self.hh_list:
            self.load_dict[hh] = self.load_dict[hh].loc[self.start_dataset:self.end_dataset, :]
            self.usage_dict[hh] = self.usage_dict[hh].loc[self.start_dataset:self.end_dataset, :]
            self.activity_dict[hh] = self.activity_dict[hh].loc[self.start_dataset:self.end_dataset, :]

    def fill_na_function(self) -> None:
        import pandas as pd

        for hh in self.hh_list:
            self.load_dict[hh] = self.load_dict[hh].fillna(0)
            self.usage_dict[hh] = self.usage_dict[hh].fillna(0)
            self.activity_dict[hh] = self.activity_dict[hh].fillna(0)

    def create_activity_probability(self, activity_dict: pd.DataFrame) -> np.ndarray:
        import numpy as np
        from datetime import timedelta
        from agents import Activity_Agent
            
        activity = Activity_Agent(activity_dict)
        X_train, y_train, X_test, y_test = activity.train_test_split(
            activity_dict, str(activity_dict.index[-1].date()+timedelta(1))
        )
        X_train = X_train.fillna(0)
        model = activity.fit(X_train, y_train, 'random forest')
        activity_probability = model.predict_proba(X_train)[:, 1]
        return activity_probability

    def create_usage_probability(self, usage_dict: pd.DataFrame, dev_name: str) -> np.ndarray:
        import numpy as np
        from agents import Usage_Agent
        
        usage = Usage_Agent(usage_dict, dev_name)
        X_train, y_train, X_test, y_test = usage.train_test_split(
            usage_dict, self.end_dataset, train_start=self.start_dataset
        )
        X_train = X_train.fillna(0) 
        model = usage.fit(X_train, y_train.values.ravel(), 'random forest')
        usage_probability = model.predict_proba(X_train)[:, 1]
        return usage_probability

    def resample_daily_to_hourly_data(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd
        from agents import Activity_Agent

        dummy_day = daily_data.index[-1] + pd.Timedelta(days=1)
        dummy_row = pd.DataFrame(index=pd.DatetimeIndex([dummy_day]))
        daily_data_with_dummy = pd.concat([daily_data, dummy_row])
        hourly_data = daily_data_with_dummy.resample("H").ffill().iloc[:-1, :]
        return hourly_data

    def create_deepar_dataset(self) -> Dict[str, pd.DataFrame]:
        import numpy as np
        import pandas as pd

        deepar_dataset = {}

        for dev in self.devices.keys():
            hh = self.devices[dev]['hh']
            dev_name = self.devices[dev]['dev_name']

            # Create a DataFrame with required columns
            deepar_dataset[dev] = pd.DataFrame({
                'temp': self.activity_dict[hh]['temp'],
                'dwpt': self.activity_dict[hh]['dwpt'],
                'rhum': self.activity_dict[hh]['rhum'],
                'wdir': self.activity_dict[hh]['wdir'],
                'wspd': self.activity_dict[hh]['wspd'],
                'usage': self.load_dict[hh][dev_name],
                'usage_bin': np.where(self.load_dict[hh][dev_name] > 0, 1, 0),
                'hh': hh,
                'dev': self.devices[dev]['dev'],
                'activity_prob': self.create_activity_probability(self.activity_dict[hh])
            })

            # add usage-related features
            columns = ['periods_since_last_activity', f'periods_since_last_{dev_name}_usage']
            daily_features = self.usage_dict[hh][columns].copy()
            daily_features.columns = daily_features.columns.str.replace(f'{dev_name}_', '', regex=False)
            daily_features['usage_prob'] = self.create_usage_probability(self.usage_dict[hh].copy(), dev_name)

            # Resample daily data to hourly and concatenate
            hourly_features = self.resample_daily_to_hourly_data(daily_features)
            deepar_dataset[dev] = pd.concat([deepar_dataset[dev], hourly_features], axis=1)

            # Select final columns
            columns = [
                'usage', 'usage_bin', 'hh', 'dev',
                'periods_since_last_activity', 'periods_since_last_usage',
                'activity_prob', 'usage_prob',
                'temp', 'dwpt', 'rhum', 'wdir', 'wspd'
            ]

            deepar_dataset[dev] = deepar_dataset[dev][columns]

        return deepar_dataset



    def pipeline(self) -> Dict[str, pd.DataFrame]:
    
        self.slice_datasets()
        
        if self.fill_na:
                self.fill_na_function()

        deepar_dataset = self.create_deepar_dataset()

        return deepar_dataset

class Transform_Dataset:
    '''
    A class for reshaping datasets into a format compatible with GluonTS, the forecasting model.
    
    Parameters:
    ----------
    start_training:
        A string ('yyyy-mm-dd') determining the start of the training period.
    start_validation:
        A string ('yyyy-mm-dd') determining the start of the testing period.
    devices:
        A dictionary containing device-specific information.
    dataset_dict:
        A dictionary with target time series data along with covariates.
    freq:
        A string specifying the frequency of the time series data (e.g., '1H' for hourly data).
    prediction_length:
        An integer representing the desired prediction horizon (default is 24).
    '''
    def __init__(
        self,
        start_training: str,
        start_validation: str,
        devices: Dict[str, Any],
        dataset_dict: Dict[str, pd.DataFrame],
        freq: str = '1H',
        prediction_length: int = 24,
    ):
        self.start_training = str(start_training)
        self.start_validation = str(start_validation)
        self.hh_list: List[str] = []
        self.devices = devices
        self.dataset_dict = {hh: df.copy() for hh, df in dataset_dict.items()}
        self.freq = freq
        self.prediction_length = prediction_length
        
    def reshape_training_data(self):
        import pandas as pd
        import numpy as np

        transposed_list = {}
        
        for dev in self.devices.keys():
            
            self.dataset_dict[dev] = self.dataset_dict[dev].loc[:self.start_validation]
            usage_bin_df = self.dataset_dict[dev]['usage_bin']
            usage_bin_df = usage_bin_df.reindex()
            usage_bin_df = pd.DataFrame(usage_bin_df)
            usage_bin_df.index = ['d' + str(n) for n in np.arange(usage_bin_df.shape[0])]
            usage_bin_df = pd.DataFrame(usage_bin_df).T

            usage_bin_df['dev'] = self.devices[dev]['dev']
            usage_bin_df['hh'] = str(self.devices[dev]['hh'])
            usage_bin_df['id'] = usage_bin_df['hh'] + '_' + usage_bin_df['dev']
            transposed_list[dev] = usage_bin_df

        training_dataset = pd.concat(transposed_list.values()).reset_index(drop=True)

        return training_dataset

    def create_dynamic_real_features(self):
        import pandas as pd
        import numpy as np

        train_dynamic_real_features_list = []
        val_dynamic_real_features_list = []
        test_dynamic_real_features_list = []

        for dev in self.devices.keys():
            dynamic_real_columns = self.dataset_dict[dev].drop(columns=['usage','usage_bin','hh','dev']).columns

            train_dynamic_real_features = (
                self.dataset_dict[dev]
                .iloc[: -self.prediction_length * 2, :][dynamic_real_columns]
                .T.to_numpy()
            )
            val_dynamic_real_features = (
                self.dataset_dict[dev]
                .iloc[:-self.prediction_length, :][dynamic_real_columns]
                .T.to_numpy()
            )
            test_dynamic_real_features = self.dataset_dict[dev][
                dynamic_real_columns
            ].T.to_numpy()

            train_dynamic_real_features_list.append(train_dynamic_real_features)
            val_dynamic_real_features_list.append(val_dynamic_real_features)
            test_dynamic_real_features_list.append(test_dynamic_real_features)

        return (
            train_dynamic_real_features_list,
            val_dynamic_real_features_list,
            test_dynamic_real_features_list,
        )

    def create_static_categorical_features(self, training_dataset: pd.DataFrame) -> Tuple[Any, Tuple[int, int]]:
        import numpy as np
        
        hh_ids = training_dataset['hh'].astype('category').cat.codes.values
        hh_ids_unique = np.unique(hh_ids)

        dev_ids = training_dataset['dev'].astype('category').cat.codes.values
        dev_ids_unique = np.unique(dev_ids)

        stat_cat_list = [hh_ids, dev_ids]

        stat_cat = np.concatenate(stat_cat_list)
        stat_cat = stat_cat.reshape(len(stat_cat_list), len(dev_ids)).T
            
        stat_cat_cardinalities: Tuple[int, int] = (len(hh_ids_unique), len(dev_ids_unique))
            
        return stat_cat, stat_cat_cardinalities

    def split_data(
    self,
    training_dataset: pd.DataFrame,
    train_dynamic_real_features_list: List[Any],
    val_dynamic_real_features_list: List[Any],
    test_dynamic_real_features_list: List[Any],
    stat_cat: Any
) -> Tuple[Any, Any, Any]:
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from gluonts.dataset.field_names import FieldName

        train_df = training_dataset.drop(['id', 'hh', 'dev'], axis=1)
        train_target_values = train_df.values

        test_target_values = val_target_values = train_target_values.copy()

        train_target_values = [ts[:-self.prediction_length * 2] for ts in train_target_values]
        val_target_values = [ts[:-self.prediction_length] for ts in val_target_values]

        dates = [pd.Timestamp(self.start_training) for _ in range(len(training_dataset))]

        def create_list_dataset(target_values, dynamic_real_features_list):
            return ListDataset(
                [
                    {
                        FieldName.TARGET: target,
                        FieldName.START: start,
                        FieldName.FEAT_DYNAMIC_REAL: fdr,
                        FieldName.FEAT_STATIC_CAT: fsc,
                    }
                    for (target, start, fdr, fsc) in zip(
                        target_values,
                        dates,
                        dynamic_real_features_list,
                        stat_cat,
                    )
                ],
                freq=self.freq,
            )

        train_ds = create_list_dataset(train_target_values, train_dynamic_real_features_list)
        val_ds = create_list_dataset(val_target_values, val_dynamic_real_features_list)
        test_ds = create_list_dataset(test_target_values, test_dynamic_real_features_list)

        return train_ds, val_ds, test_ds

    def pipeline(self) -> Tuple[Any, Any, Any]:
        import pandas as pd
        import numpy as np
        from gluonts.dataset.common import ListDataset
        from gluonts.dataset.field_names import FieldName

        training_dataset = self.reshape_training_data()

        (
            train_dynamic_real_features_list,
            val_dynamic_real_features_list,
            test_dynamic_real_features_list,
        ) = self.create_dynamic_real_features()
            
        stat_cat = self.create_static_categorical_features(training_dataset)[0]

        train_ds, val_ds, test_ds = self.split_data(
            training_dataset,
            train_dynamic_real_features_list,
            val_dynamic_real_features_list,
            test_dynamic_real_features_list,
            stat_cat
        )

        return train_ds, val_ds, test_ds

class Metric_Inference_Early_Stopping(Callback):
    '''
    Early Stopping mechanism based on the prediction network.
    Can be used to base the Early Stopping directly on a metric of interest, instead of on the training/validation loss.
    In the same way as test datasets are used during model evaluation,
    the time series of the validation_dataset can overlap with the train dataset time series,
    except for a prediction_length part at the end of each time series.
    Parameters
    ----------
    validation_dataset
        An out-of-sample dataset which is used to monitor metrics
    predictor
        A gluon predictor, with a prediction network that matches the training network
    evaluator
        The Evaluator used to calculate the validation metrics.
    metric
        The metric on which to base the early stopping on.
    patience
        Number of epochs to train on given the metric did not improve more than min_delta.
    min_delta
        Minimum change in the monitored metric counting as an improvement
    verbose
        Controls, if the validation metric is printed after each epoch.
    minimize_metric
        The metric objective.
    restore_best_network
        Controls, if the best model, as assessed by the validation metrics is restored after training.
    num_samples
        The amount of samples drawn to calculate the inference metrics.
    '''

    def __init__(
        self,
        validation_dataset: Dataset,
        estimator: DeepAREstimator,
        evaluator: Evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9],allow_nan_forecast=True),
        metric: str = 'MSE',
        patience: int = 20,
        min_delta: float = 0.0,
        verbose: bool = False,
        minimize_metric: bool = True,
        restore_best_network: bool = True,
        num_samples: int = 100,
    ):
        assert (
            patience >= 0
        ), 'EarlyStopping Callback patience needs to be >= 0'
        assert (
            min_delta >= 0
        ), 'EarlyStopping Callback min_delta needs to be >= 0.0'
        assert (
            num_samples >= 1
        ), 'EarlyStopping Callback num_samples needs to be >= 1'

        self.validation_dataset = list(validation_dataset)
        self.estimator = estimator
        self.evaluator = evaluator
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_network = restore_best_network
        self.num_samples = num_samples

        if minimize_metric:
            self.best_metric_value = np.inf
            self.is_better = np.less
        else:
            self.best_metric_value = -np.inf
            self.is_better = np.greater

        self.validation_metric_history: List[float] = []
        self.best_network = None
        self.n_stale_epochs = 0

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: mx.gluon.nn.HybridBlock,
        trainer: mx.gluon.Trainer,
        best_epoch_info: dict,
        ctx: mx.Context
    ) -> bool:
        should_continue = True
        
        transformation = self.estimator.create_transformation()
        predictor = self.estimator.create_predictor(transformation=transformation, trained_network=training_network)

        from gluonts.evaluation.backtest import make_evaluation_predictions

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.validation_dataset,
            predictor=predictor,
            num_samples=self.num_samples,
        )

        agg_metrics, item_metrics = self.evaluator(ts_it, forecast_it)
        current_metric_value = agg_metrics[self.metric]
        self.validation_metric_history.append(current_metric_value)

        if self.verbose:
            print(
                f'Validation metric {self.metric}: {current_metric_value}, best: {self.best_metric_value}'
            )

        if self.is_better(current_metric_value, self.best_metric_value):
            self.best_metric_value = current_metric_value

            if self.restore_best_network:
                training_network.save_parameters('best_network.params')

            self.n_stale_epochs = 0
        else:
            self.n_stale_epochs += 1
            if self.n_stale_epochs == self.patience:
                should_continue = False
                print(
                    f'EarlyStopping callback initiated stop of training at epoch {epoch_no}.'
                )

                if self.restore_best_network:
                    print(
                        f'Restoring best network from epoch {epoch_no - self.patience}.'
                    )
                    training_network.load_parameters('best_network.params')

        return should_continue

class DeepAR_Tuning_Objective:
    '''
    A class for hyperparameter tuning using Optuna.

    Parameters:
    ----------
    prediction_length:
        An integer representing the desired prediction horizon.
    freq:
        A string specifying the frequency of the time series data.
    start_training:
        A string ('yyyy-mm-dd') determining the start of the training period.
    start_validation:
        A string ('yyyy-mm-dd') determining the start of the validation period.
    recommendation_length:
        The length of the test period in days.
    devices:
        A dictionary containing device-specific information.
    dataset_dict:
        A dictionary with target time series data along with covariates.
    metric_type:
        A string specifying the type of metric to optimize during tuning (default is 'MASE').
    '''
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        start_training: str,
        start_validation: str,
        recommendation_length: int,
        devices: Dict[str, Any],
        dataset_dict: Dict[str, pd.DataFrame],
        metric_type='MASE', 
    ):
        self.prediction_length = prediction_length
        self.freq = freq
        self.start_training = start_training
        self.start_validation = start_validation
        self.recommendation_length = recommendation_length
        self.devices = devices
        self.dataset_dict = dataset_dict
        self.metric_type = metric_type
        
    def get_params(self, trial) -> dict:
        return {
            # Number of time steps in the context window (3, 4, or 5 weeks)
            'context_length': trial.suggest_categorical('context_length', [504, 672, 840]),
            
            # Number of layers in the network
            'num_layers': trial.suggest_int('num_layers', 3, 4),
            
            # Number of cells in each layer
            'num_cells': trial.suggest_int('num_cells', 10, 50),
            
            # Dropout rate to prevent overfitting
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            
            # Learning rate for optimization (log scale)
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            
            # Number of training epochs
            'epochs': trial.suggest_int('epochs', 25, 100),
            
            # Batch size for training
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            
            # Number of batches per epoch
            'num_batches_per_epoch': trial.suggest_categorical('num_batches_per_epoch', [40, 50, 60, 70, 80, 90]),
        }

    def __call__(self, trial):
        params = self.get_params(trial)

        transformer = Transform_Dataset(
            self.start_training,
            self.start_validation,
            self.devices,
            self.dataset_dict,
            self.freq, 
            self.prediction_length)
        
        training_dataset = transformer.reshape_training_data()
        
        # Create static categorical features
        cardinality = transformer.create_static_categorical_features(training_dataset)[1]

        # Initialize the DeepAR estimator
        estimator = DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=params['context_length'],
            freq=self.freq,
            distr_output=CategoricalOutput(2),
            use_feat_dynamic_real=True,
            use_feat_static_cat=True,
            cardinality=cardinality,
            num_layers=params['num_layers'],
            num_cells=params['num_cells'],
            batch_size=params['batch_size'],
            dropout_rate=params['dropout_rate'],
        )
        
        evaluator = Evaluator(allow_nan_forecast=True)

        train_ds, val_ds, _ = transformer.pipeline()
        
        es_callback = Metric_Inference_Early_Stopping(
            validation_dataset=val_ds,
            estimator=estimator,
            metric='RMSE',
            patience=20,
            evaluator=evaluator
        )

        trainer = Trainer(
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            num_batches_per_epoch=params['num_batches_per_epoch'],
            callbacks=[es_callback],
            hybridize=False
        )
        
        datelist_daily = pd.date_range(self.start_validation, freq='d', periods=self.recommendation_length)

        estimator.trainer = trainer
        global predictor
        daily_error = 0
        

        predictor = estimator.train(train_ds)

        for date in datelist_daily:
            
            transformer = Transform_Dataset(
                self.start_training,
                date,
                self.devices,
                self.dataset_dict,
                self.freq, 
                self.prediction_length)

            _, _, test_ds = transformer.pipeline()

            forecast_deepar, ts_deepar = make_evaluation_predictions(
                dataset=test_ds,
                predictor=predictor,
                num_samples=100
            )
            forecasts = list(forecast_deepar)
            tss = list(ts_deepar)

            forecasts_bin = {}
            for i in range(0, len(forecasts)):

                forecasts_bin[i] = np.where(forecasts[i].quantile(0.5) >= 0.5, 1, 0)
                daily_error += abs(np.where(forecasts_bin[i].sum() == 0, 0, 1) - np.where(tss[i][-24:].sum()[0] == 0, 0, 1))
                
        trial.set_user_attr(key='best_model', value=predictor)

        return daily_error

class Evaluation_Metrics:
    '''
    A class for evaluating binary forecasts against target data.
    It calculates various metrics including misclassification error, accuracy, precision,
    recall, and F1-score for binary forecasts at both hourly and daily levels.
    Parameters:
    ----------
    forecast
        A dictionary of forecasted dataframes for each device
    target
        A dictionary of target dataframes for each device
    devices
        A dictionary containing device-specific information
    '''

    def __init__(
        self,
        forecast: Dict[str, pd.DataFrame],
        target: Dict[str, pd.DataFrame],
        devices: Dict[str, Any],

    ):
        self.forecast =  {dev: df.copy() for dev, df in forecast.items()}
        self.target = {dev: df.copy() for dev, df in target.items()} 
        self.date_list = np.unique(next(iter(self.target.values())).index.date)
        self.devices = devices

    def update_forecast_with_tolerance(self):
    
        import copy

        arr_dict = {}

        forecast_with_tolerance = copy.deepcopy(self.forecast)

        for dev in forecast_with_tolerance.keys():
            length = len(forecast_with_tolerance[dev])

            positive_forecasts = np.where(forecast_with_tolerance[dev] == 1)[0]

            if positive_forecasts.size > 0:

                forecast_with_tolerance[dev].iloc[np.maximum(positive_forecasts - 1, 0)] = 1
                forecast_with_tolerance[dev].iloc[np.maximum(positive_forecasts - 2, 0)] = 1
                forecast_with_tolerance[dev].iloc[np.minimum(positive_forecasts + 1, length - 1)] = 1
                forecast_with_tolerance[dev].iloc[np.minimum(positive_forecasts + 2, length - 1)] = 1
                
        return forecast_with_tolerance

    def total_misclassification_error(self, forecast):
        total_error = 0

        for dev in self.target.keys():
            FP = sum((self.forecast[dev].values == 1) & (self.target[dev].values == 0))
            FN = sum((forecast[dev].values == 0) & (self.target[dev].values == 1))

            total_error += FP + FN
        
        return total_error[0]

    def total_daily_misclassification_error(self, forecast):
        
        total_daily_error = 0
        for dev in self.target.keys():
                
            daily_forecast = forecast[dev].groupby(forecast[dev].index.date).sum()
            daily_target = self.target[dev].groupby(self.target[dev].index.date).agg(lambda x: np.nan if x.isnull().any() else x.sum())
            
            daily_forecast['usage'] = daily_forecast['usage'].apply(lambda x: 1 if x > 0 else 0)
            daily_target['usage'] = daily_target['usage'].apply(lambda x: 1 if (not pd.isna(x) and x > 0) else x)
                
            FP = sum((daily_forecast.values == 1) & (daily_target.values == 0))
            FN = sum((daily_forecast.values == 0) & (daily_target.values == 1))

            total_daily_error += FP + FN
        
        return total_daily_error[0]

    def hourly_accuracy(self, forecast):
        x = self.total_misclassification_error(forecast)
        
        sum_notna_hours = 0
        for i in self.target.keys():
            sum_notna_hours += self.target[i].notna().sum()[0]
        
        accuracy = 1 - x / sum_notna_hours
        return round(accuracy,3)

    def daily_accuracy(self, forecast):
        tme_daily = self.total_daily_misclassification_error(forecast)
            
        sum_notna_days = 0

        for i in self.target.keys():
            daily_target = self.target[i].groupby(self.target[i].index.date).agg(lambda x: np.nan if x.isnull().any() else x.sum())
            sum_notna_days += daily_target.notna().sum()
            
        accuracy = (1 - tme_daily / sum_notna_days)[0]
        return round(accuracy,3)

    def precision(self,forecast):
    
        total_precision = 0
        
        for dev in self.target.keys():
            
            TP = sum((forecast[dev].values == 1) & (self.target[dev].values == 1))
            FP = sum((self.forecast[dev].values == 1) & (self.target[dev].values == 0))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            total_precision += precision
            
        average_precision = total_precision[0]/len(self.devices)

        return round(average_precision,3)

    def recall(self,forecast):
    
        total_recall = 0
        
        for dev in self.target.keys():
            
            TP = sum((forecast[dev].values == 1) & (self.target[dev].values == 1))
            FN = sum((forecast[dev].values == 0) & (self.target[dev].values == 1))

            # Calculate Precision and Recall
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            total_recall += recall
        
        average_recall = total_recall[0]/len(self.devices)

        return round(average_recall,3)
        
    def f1_score(self,forecast):
    
        total_f1_score = 0
        
        for dev in self.target.keys():
            
            TP = sum((forecast[dev].values == 1) & (self.target[dev].values == 1))
            FN = sum((forecast[dev].values == 0) & (self.target[dev].values == 1))
            FP = sum((self.forecast[dev].values == 1) & (self.target[dev].values == 0))
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
            if precision == 0 and recall == 0:
                f1_score =  0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)
                
            total_f1_score += f1_score
            
        average_f1_score = total_f1_score[0]/len(self.devices)  

        return round(average_f1_score,3)

class Generate_Forecasts:
    '''
    A class for generating forecasts using a specified DeepAR predictor model.
    Parameters:
    ----------
    start_dataset :
        The starting date of the dataset used for training the predictor.
    start_forecast :
        The start date of the forecasting period in 'yyyy-mm-dd' format.
    length_forecast :
        The length of the forecasting period in days.
    predictor :
        The machine learning model used for making forecasts.
    num_samples :
        The number of samples to generate for each device's forecast.
    devices :
        A dictionary containing device-specific information.
    target :
        The target variable for the forecasts.
    dataset_dict :
        A dictionary with target time series data along with covariates.
    '''
    def __init__(
        self,
        start_dataset,
        start_forecast,
        length_forecast,
        predictor,
        num_samples,
        devices,
        target,
        dataset_dict
    ):
        from helper_functions_thesis import Helper_Functions_Thesis
        self.start_dataset = start_dataset
        self.date_list_daily = Helper_Functions_Thesis.create_date_list_daily(start_forecast, length_forecast)
        self.date_list_hourly = Helper_Functions_Thesis.create_date_list_hourly(start_forecast, length_forecast)
        self.predictor = predictor
        self.num_samples = num_samples
        self.devices = devices
        self.target = target
        self.dataset_dict = dataset_dict

    def create_forecast_samples(self):
    
        from Thesis_Classes import Transform_Dataset
        from datetime import timedelta
        from gluonts.evaluation.backtest import make_evaluation_predictions
        
        forecasts = {}

        for day, date in enumerate(self.date_list_daily):

            transformer = Transform_Dataset(self.start_dataset, 
                                            date, 
                                            self.devices, 
                                            self.dataset_dict)
            
            _, _, test_ds = transformer.pipeline()

            forecast_deepar, _ = make_evaluation_predictions(
                dataset=test_ds,
                predictor=self.predictor,
                num_samples=self.num_samples
            )
            
            forecast = list(forecast_deepar)

            forecast_dict = {}
            for i, device_key in enumerate(self.devices.keys()):
                forecast_dict[device_key] = forecast[i]

            forecasts[day] = forecast_dict
            
        return forecasts

    def create_forecast_lists(self, forecasts, threshold=0.5):
        forecasts_list = {}
        quantile = 1-threshold
        
        for i in self.devices.keys():
            forecasts_list[i] = pd.DataFrame()
            
            for date in forecasts.keys():
                quantile_forecast = np.quantile(forecasts[date][i].samples, quantile, axis=0)
                quantile_df = pd.DataFrame(quantile_forecast.astype(int))
                forecasts_list[i] = pd.concat([forecasts_list[i], quantile_df])
            
            forecasts_list[i].index = self.date_list_hourly
            forecasts_list[i].columns = ['usage']

        return forecasts_list

    def evaluate_forecast_samples(self, forecast_samples, metric_name):

        supported_metrics = [
            'total_misclassification_error',
            'total_daily_misclassification_error',
            'hourly_accuracy',
            'daily_accuracy',
            'precision',
            'recall',
            'f1_score'
        ]

        if metric_name not in supported_metrics:
            raise ValueError(f"Unsupported metric '{metric_name}'. Please choose a metric from the list: {supported_metrics}")

        metric_values = []
        thresholds = np.arange(0, 1, 0.01)

        for threshold in thresholds:
            forecast = self.create_forecast_lists(forecast_samples, threshold)
            metrics = Evaluation_Metrics(forecast, self.target, self.devices)
            metric_function = getattr(metrics, metric_name)
            metric_result = metric_function(forecast)
            metric_values.append(metric_result)
            
        if metric_name in ['total_misclassification_error', 'total_daily_misclassification_error']:
            best_metric_result = min(metric_values)
        else:
            best_metric_result = max(metric_values)
        
        best_threshold_idx = metric_values.index(best_metric_result)
        best_threshold = thresholds[best_threshold_idx]

        return metric_values, best_threshold, best_threshold_idx

    def rescale_probabilities(self, prob, threshold):
        if prob<=threshold:
            x = prob/(2*threshold)
        else:
            x = (prob-threshold)/(2*(1-threshold))+0.5
        return x

    def create_shifted_probabilities(self, samples, threshold):
        problist = {}
        for dev in self.devices.keys():
            problist[dev] = []
            for date in range(0,len(samples)):
                problist[dev] = problist[dev] + [self.rescale_probabilities(p,threshold) for p in samples[date][dev].samples.mean(axis=0)]
        return problist

    def rescale_forecast_samples(self, shifted_probability_list, forecast_samples):

        from gluonts.model.forecast import SampleForecast

        forecast_samples_rescaled = {}
        for dev in self.devices.keys():
            forecast_samples_rescaled[dev] = SampleForecast(samples=np.array([[0] * len(self.date_list_hourly)] * self.num_samples),
                                                        start_date=forecast_samples[0][dev].start_date)

            for sample in range(0,len(forecast_samples_rescaled[dev].samples)):
                forecast_samples_rescaled[dev].samples[sample] = [np.random.choice([0, 1], p=[1 - x, x]) for x in shifted_probability_list[dev]]
        
        return forecast_samples_rescaled

    @staticmethod
    def create_forecast_quantiles(forecast_samples, quantile = 0.5):
        from helper_functions_thesis import Helper_Functions_Thesis
        forecast_quantile = {}
        first_key = next(iter(forecast_samples))
        start_forecast = forecast_samples[first_key].start_date.strftime('%Y-%m-%d')
        length_forecast = len(forecast_samples[first_key].quantile(0.5)) / 24
        date_list_hourly = Helper_Functions_Thesis.create_date_list_hourly(start_forecast, length_forecast)

        for dev in forecast_samples.keys():

            forecast_quantile[dev] = pd.DataFrame(forecast_samples[dev].quantile(quantile),
                                                columns=['usage'],index=date_list_hourly)
            
        return forecast_quantile

class Create_Counterfactual_Time_Series:
    '''
    A class for generating counterfactual load usage patterns from binary forecasts
    Parameters:
    ----------
    forecast
        A dictionary of forecasted dataframes for each device
    forecast_samples
        A Forecast object, where the predicted distribution is represented internally as samples
    target
        A dictionary of target dataframes for each device
    devices
        A dictionary containing device-specific information
    price_df
         A DataFrame with hourly price data in GBP per megawatt-hour.
    '''
    def __init__(
        self,
        forecast_samples,
        forecast: Dict[str, pd.DataFrame],
        target: Dict[str, pd.DataFrame],
        devices: Dict[str, Any],
        price_df: pd.DataFrame,

    ):
        import numpy as np
        self.forecast_samples = forecast_samples
        self.forecast =  {dev: df.copy() for dev, df in forecast.items()}
        self.target = {dev: df.copy() for dev, df in target.items()} 
        self.devices = devices
        self.date_list = np.unique(next(iter(self.forecast.values())).index.date)
        self.price_df = price_df
        
    def _calculate_usage_length(self, data, start_dataset):
        import pandas as pd
        return (
            ((data.iloc[start_dataset:] == 0).idxmax() - data.index[start_dataset])
                // pd.Timedelta(minutes=60)
            )[0]

    def _insert_usage_values_from_target(
    self,
    daily_forecast_extended,
    daily_target_extended,
    start_hours_forecast,
    target_start_hour,
    load_profile,
):
        import pandas as pd
            
        nearest_forecast_start_hour = min(
            start_hours_forecast, key=lambda x: abs(x - target_start_hour)
        )

        time_shift = nearest_forecast_start_hour - target_start_hour

        usage_length_target = self._calculate_usage_length(
            daily_target_extended, target_start_hour
        )
        usage_length_forecast = self._calculate_usage_length(
            daily_forecast_extended, nearest_forecast_start_hour
        )
        
        usage_length_forecast = min(usage_length_forecast,24)

        target_hours_pre_shift = daily_target_extended.iloc[
            target_start_hour:target_start_hour + usage_length_forecast
        ]
        
        target_hours_post_shift = target_hours_pre_shift.shift(periods=time_shift, freq="60T")
        if usage_length_forecast > usage_length_target:

            target_hours_post_shift.iloc[
                usage_length_target: min(usage_length_forecast,len(target_hours_post_shift)),0
            ] = load_profile[usage_length_target:min(usage_length_forecast,len(target_hours_post_shift))]
        
        daily_forecast_extended.iloc[
            nearest_forecast_start_hour:nearest_forecast_start_hour
            + usage_length_forecast
        ] = 0
        daily_forecast_extended = target_hours_post_shift.combine_first(
            daily_forecast_extended
        )

        start_hours_forecast.remove(nearest_forecast_start_hour)
        
        return daily_forecast_extended, start_hours_forecast

    def _insert_usage_values_from_load_profile(
    self,
    load_profile,
    daily_forecast_extended,
    forecast_start_dataset,
):
        usage_length_forecast = self._calculate_usage_length(
            daily_forecast_extended, forecast_start_dataset
        )
        
        daily_forecast_extended.iloc[
            forecast_start_dataset:forecast_start_dataset
            + usage_length_forecast,
            0,
        ] = load_profile[0:usage_length_forecast]
            
        return daily_forecast_extended

    def create_counterfactual_time_series(self, is_info_displayed=False):
        from datetime import timedelta
        import pandas as pd
        
        results = {}
        
        # Iterate through each device
        for dev in self.devices.keys():
            forecast = self.forecast[dev].copy()
            target = self.target[dev]
            load_profile = self.devices[dev]['load_profile']
            
            # Initialize an output DataFrame
            output = pd.DataFrame(index=forecast.index, columns=['usage'])
            
            if is_info_displayed:
                print('\n', 'device nr: ', dev)
            
            # Iterate over each date in the specified date list
            for date in self.date_list:
                date_before = date - timedelta(days=1)
                date = str(date)

                # Extract daily forecast and target data
                daily_forecast = forecast.loc[date]
                daily_target = target.loc[date]

                daily_target_extended = target.loc[date:][:48]
                daily_forecast_extended = forecast.loc[date:][:48]

                # Check if there is device usage on the forecast day
                if daily_forecast.values.sum() == 0:
                    if is_info_displayed:
                        print(date, "no device usage forecasted")
                    continue

                # Create shifted versions of daily forecast and target to calculate start hours
                shifted_daily_forecast = daily_forecast.shift()
                shifted_daily_target = daily_target.shift()
                    
                if date != str(self.date_list[0]):
                    shifted_daily_forecast.iloc[0] = forecast.loc[str(date_before)].iloc[-1]
                    shifted_daily_target.iloc[0] = target.loc[str(date_before)].iloc[-1]
                    
                start_hours_forecast = daily_forecast[
                    (daily_forecast.values == 1) & (shifted_daily_forecast.values == 0)
                ].index.hour.tolist()

                # Continue only if there is usage start on the forecast day
                if start_hours_forecast == []:
                    continue

                # Identify the start hours for device usage in target
                start_hours_target = daily_target[
                    (daily_target.values > 0) & (shifted_daily_target.values == 0)
                ].index.hour.tolist()

                # Sort target start hours based on proximity to forecast start hours
                sorted_start_hours_target = sorted(
                    start_hours_target,
                    key=lambda x: min(abs(x - r) for r in start_hours_forecast),
                )
                
                if is_info_displayed:
                    if len(start_hours_target) > len(start_hours_forecast):
                        print(date, 'More device start hours than forecasted')
                    else:
                        print(date, 'Less than or equal device start hours than forecasted')
                        
                # Iterate through sorted target start hours
                for target_start_hour in sorted_start_hours_target:
                    if start_hours_forecast == []:
                        break
                    
                    # Insert target usage values into forecast
                    daily_forecast_extended, start_hours_forecast = self._insert_usage_values_from_target(
                        daily_forecast_extended,
                        daily_target_extended,
                        start_hours_forecast,
                        target_start_hour,
                        load_profile,
                    )

                # Handle remaining start hours by inserting load profile data
                for forecast_start_dataset in start_hours_forecast:
                    if daily_forecast_extended['usage'].iloc[forecast_start_dataset] <= 1:
                        daily_forecast_extended = self._insert_usage_values_from_load_profile(
                            load_profile,
                            daily_forecast_extended,
                            forecast_start_dataset,
                        )
                        
                # Combine daily adjusted usage values into the output DataFrame
                output = daily_forecast_extended[daily_forecast_extended['usage'] > 1].combine_first(output)
                    
            results[dev] = output.fillna(0)

        return results

    def calculate_total_cost(self, usage_df):
        import pandas as pd
        usage_df = pd.DataFrame(usage_df)
        usage_df = usage_df.join(self.price_df, how="inner")
        usage_df["total_cost"] = usage_df["usage"] * usage_df["Price"]

        return usage_df

    def agg_to_day_level(self, time_series):
        import pandas as pd
        agg_time_series = {}
        for i in time_series.keys():
            agg_time_series[i] = 0
            agg_time_series[i] = time_series[i].groupby(pd.Grouper(freq='D')).sum()

        return agg_time_series

    def agg_to_dev_level(self, time_series):
        import numpy as np
        agg_time_series = {}
        count_per_dev = {}

        for dev in np.unique([self.devices[x]["dev"] for x in self.devices.keys()]):
            agg_time_series[dev] = 0
            count_per_dev[dev] = 0

            for i in [x for x in self.devices.keys() if self.devices[x]["dev"] == dev]:
                agg_time_series[dev] += time_series[i]
                count_per_dev[dev] += 1  # Increment the count for this device

            agg_time_series[dev] /= count_per_dev[dev]

        return agg_time_series

    def agg_to_hh_level(self, time_series):
        import numpy as np
        agg_time_series = {}
        count_per_hh = {}
        for hh in np.unique([self.devices[x]["hh"] for x in self.devices.keys()]):
            agg_time_series[hh] = 0
            count_per_hh[hh] = 0

            for i in [x for x in self.devices.keys() if self.devices[x]["hh"] == hh]:
                agg_time_series[hh] += time_series[i]
                count_per_hh[hh] += 1
            
            agg_time_series[hh] /= count_per_hh[hh]

        return agg_time_series

    def agg_complete(self, time_series):

        agg_time_series = list(time_series.values())[0].copy()
        agg_time_series = agg_time_series*0

        for dev in self.devices.keys():
            agg_time_series += time_series[dev].copy()

        agg_time_series_dict = {}
        agg_time_series_dict["all"] = agg_time_series / len(self.devices)
        return agg_time_series_dict 

    def pipeline(self,is_info_displayed=False,daily_agg=True,agg_level=None):

        counterfactual_usage = self.create_counterfactual_time_series(is_info_displayed)
        
        counterfactual_cost = {}
        for dev in self.devices.keys():
            counterfactual_cost[dev] = self.calculate_total_cost(counterfactual_usage[dev])

        if agg_level == "dev":
            counterfactual_output = self.agg_to_dev_level(counterfactual_cost)
        elif agg_level == "hh":
            counterfactual_output = self.agg_to_hh_level(counterfactual_cost)
        elif agg_level == "all":
            counterfactual_output = self.agg_complete(counterfactual_cost)
        elif agg_level is None:
            counterfactual_output = counterfactual_cost
        else:
            raise ValueError("Invalid aggregation option. Please choose from 'dev', 'hh', 'all', or None.")
                
        if daily_agg:
            counterfactual_output = self.agg_to_day_level(counterfactual_output)
                
        return counterfactual_output

    def generate_aggregated_counterfactuals(self):
        scenarios = {}
        
        agg_levels = ['hh', 'dev', 'all', None]
        daily_agg_options = [True, False]
        
        for agg_level in agg_levels:
            for daily_agg in daily_agg_options:
                
                scenario_key = f'{"daily" if daily_agg else "hourly"}_{agg_level}'
                
                scenarios[scenario_key] = self.pipeline(
                    is_info_displayed=False, daily_agg=daily_agg, agg_level=agg_level
                )
        
        return scenarios

    def generate_aggregated_loads(self, loads):
        from copy import deepcopy
    
        scenarios = {}
        
        aggregated_loads = deepcopy(loads)
        
        for i in self.devices.keys():  
            aggregated_loads[i] = self.calculate_total_cost(aggregated_loads[i])
        
        scenarios = {}
        
        scenarios['hourly_None'] = aggregated_loads
        scenarios['hourly_dev'] = self.agg_to_dev_level(aggregated_loads)
        scenarios['hourly_hh'] = self.agg_to_hh_level(aggregated_loads)
        scenarios['hourly_all'] = self.agg_complete(aggregated_loads)
        
        scenarios['daily_None'] = self.agg_to_day_level(scenarios['hourly_None'])
        scenarios['daily_dev'] = self.agg_to_day_level(scenarios['hourly_dev'])
        scenarios['daily_hh'] = self.agg_to_day_level(scenarios['hourly_hh'])
        scenarios['daily_all'] = self.agg_to_day_level(scenarios['hourly_all'])
        
        return scenarios

    def time_series_plot(self, quantile_list, show_control=False, show_treatment_with_usage=False, show_treatment_without_usage=True, daily_agg=True, agg_level=None):
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from Thesis_Classes import Generate_Forecasts

        cmap = LinearSegmentedColormap.from_list('CustomGradient', ['#ffdfdb', '#ff82c4', '#d6163b'])

        aggregation = f'{"daily" if daily_agg else "hourly"}_{agg_level}'

        forecast_quantile_list = {}

        for q in quantile_list:
            print('Generating forecast for the {}-quantile...'.format(q))
            
            forecast_quantile = Generate_Forecasts.create_forecast_quantiles(self.forecast_samples, quantile=q)
            create_counterfactuals = Create_Counterfactual_Time_Series(self.forecast_samples, forecast_quantile, self.target, self.devices, self.price_df)
            forecast_quantile_list[q] = create_counterfactuals.pipeline(
                is_info_displayed=False, daily_agg=daily_agg, agg_level=agg_level
            )

        for item, values in forecast_quantile_list[q].items():
            width = 16
            height = 5.71

            plt.figure(figsize=(width, height))

            if show_control:
                control_costs = int(self.control_aggregations[aggregation][item]['total_cost'].sum())

                plt.plot(
                    self.control_aggregations[aggregation][item]['total_cost'],
                    label=f"control                                       {control_costs:,}",
                    color='blue'
                )
            
            if show_treatment_with_usage:
                treatment = self.treatment_add_usage_aggregations
                treatment_costs = int(treatment[aggregation][item]['total_cost'].sum())
                        
                if show_control:
                    # Calculate relative cost difference between treatment and control
                    relative_cost_diff_treatment = (treatment_costs - control_costs) / control_costs
                    treatment_label = f"treatment (w. added usage)      {treatment_costs:,} ({relative_cost_diff_treatment:.1%})"
                else:
                    treatment_label = f"treatment (w. added usage)      {treatment_costs:,}"
                        
                plt.plot(
                    treatment[aggregation][item]['total_cost'],
                    label=treatment_label,
                    color='green'
                )
                        
            if show_treatment_without_usage:
                treatment = self.treatment_aggregations
                treatment_costs = int(treatment[aggregation][item]['total_cost'].sum())
                        
                if show_control:
                    # Calculate relative cost difference between treatment and control
                    relative_cost_diff_treatment = (treatment_costs - control_costs) / control_costs
                    treatment_label = f"treatment (w.o. added usage)   {treatment_costs:,} ({relative_cost_diff_treatment:.1%})"
                else:
                    treatment_label = f"treatment (w.o. added usage)   {treatment_costs:,}"
                        
                plt.plot(
                    treatment[aggregation][item]['total_cost'],
                    label=treatment_label,
                    color='mediumaquamarine'
                )
            for q in quantile_list:
                color = cmap((1 - q) * 2) if q > 0.5 else cmap(q * 2)
                counterf_costs = int(forecast_quantile_list[q][item]['total_cost'].sum())
                
                if show_control:
                    # Calculate relative cost difference between counterfactual and control
                    relative_cost_diff_counterfactual = (counterf_costs - control_costs) / control_costs
                    counterfactual_label = f"counterf. {q:.2f}-q                         {counterf_costs:,} ({relative_cost_diff_counterfactual:.1%})"
                else:
                    counterfactual_label = f"counterf. {q:.2f}-q                         {counterf_costs:,}"

                plt.plot(
                    forecast_quantile_list[q][item]['total_cost'],
                    label=counterfactual_label,
                    color=color
                )

            plt.title('Hourly Energy Costs - {}'.format(item), fontsize=14)
            plt.xlabel('Day', fontsize=12)
            plt.ylabel('Energy Cost', fontsize=12)
            plt.grid()
            plt.legend(loc=1, title='Total Cost', fontsize=12)
            plt.xticks(rotation=20, fontsize=12)
            plt.tight_layout()
            plt.show()

class Quantile_Distribution:
    '''
    A class for computing and plotting quantile distributions of energy costs in different scenarios
    Parameters:
    ----------
    control
        A dictionary of control dataframes for each device.
    treatment
        A dictionary of treatment dataframes for each device.
    treatment_added_usage
        A dictionary of treatment dataframes with added usage for each device.
    counterfactual
        A dictionary of counterfactual dataframes for each device.
    '''
    def __init__(
        self,
        control,
        treatment,
        treatment_added_usage,
        counterfactual,

    ):
        import numpy as np
        self.control =  {dev: df.copy() for dev, df in control.items()}
        self.treatment = {dev: df.copy() for dev, df in treatment.items()} 
        self.treatment_added_usage = {dev: df.copy() for dev, df in treatment_added_usage.items()} 
        self.counterfactual = {dev: df.copy() for dev, df in counterfactual.items()} 
        self.quantile_range = np.arange(0, 1.01, 0.01)
    
    @staticmethod
    def calculate_quantile(data, p):
        import numpy as np

        n = len(data)
        rank = p * (n - 1) 
        floor_rank = np.floor(rank).astype(int)
        ceil_rank = np.ceil(rank).astype(int)
        
        if floor_rank == ceil_rank:
            return data[floor_rank]
        else:
            lower_value = data[floor_rank]
            upper_value = data[ceil_rank]
            weight = rank - floor_rank
            return lower_value + (upper_value - lower_value) * weight
        
    def calculate_empirical_quantiles(self, data):
        import numpy as np
        
        quantile_values = []
        for q in self.quantile_range:
            
            sorted_data = np.sort([x for x in data if str(x) != 'nan'])
            quantile_value = self.calculate_quantile(sorted_data, q)
            quantile_values.append(quantile_value)
            
        return quantile_values
    
    def calculate_aggregated_quantiles(self, data_aggregations,daily_agg=True, agg_level=None, is_nonzero = True):
        from copy import deepcopy

        aggregation = f'{"daily" if daily_agg else "hourly"}_{agg_level}'
                
        data_agg = deepcopy(data_aggregations[aggregation])
            
        quantile_values = {}
            
        for ax, item in enumerate(data_agg.keys()):
            if is_nonzero:
                data_agg[item] = data_agg[item]['total_cost'][data_agg[item]['total_cost'] > 0]
            else:
                data_agg[item] = data_agg[item]['total_cost']
            quantile_values[item] = self.calculate_empirical_quantiles(data_agg[item])
            
        return quantile_values
    
    @staticmethod
    def extract_selected_quantiles(quantile):
        selected_quantiles = []
        quantile_list = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
            
        for index in quantile_list: 
            selected_quantiles.append(quantile[index])
            
        return selected_quantiles
    
    def create_quantile_table(self, t_quantile, t_u_quantile, c_quantile, cf_quantile):
        import pandas as pd

        t_quantile_df = pd.DataFrame({'treatment': self.extract_selected_quantiles(t_quantile)})
        t_u_quantile_df = pd.DataFrame({'treatment_with_usage': self.extract_selected_quantiles(t_u_quantile)})
        c_quantile_df = pd.DataFrame({'control': self.extract_selected_quantiles(c_quantile)})
        cf_quantile_df = pd.DataFrame({'counterfactual': self.extract_selected_quantiles(cf_quantile)})

        result_df = pd.concat([c_quantile_df, cf_quantile_df, t_quantile_df, t_u_quantile_df], axis=1).astype(int)
        result_df['counterfactual_diff'] = round(100*(result_df['counterfactual'] - result_df['control']) / result_df['control'], 2)
        result_df['treatment_with_usage_diff'] = round(100*(result_df['treatment_with_usage'] - result_df['control']) / result_df['control'], 2)
        result_df['treatment_diff'] = round(100*(result_df['treatment'] - result_df['control']) / result_df['control'], 2)
        result_df.index = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
        
        result_df = result_df.iloc[:,-3:].T.fillna(0)

        return result_df
    
    def plot_quantile_distributions(self, daily_agg=True, agg_level='all', is_nonzero=False):
        import matplotlib.pyplot as plt
        import numpy as np
        from pandas.plotting import table
        
        treatment_quantile = self.calculate_aggregated_quantiles(self.treatment, daily_agg=daily_agg, agg_level=agg_level, is_nonzero=is_nonzero)
        treatment_u_quantile = self.calculate_aggregated_quantiles(self.treatment_added_usage, daily_agg=daily_agg, agg_level=agg_level,  is_nonzero=is_nonzero)
        control_quantile = self.calculate_aggregated_quantiles(self.control, daily_agg=daily_agg, agg_level=agg_level, is_nonzero=is_nonzero)
        counterfactual_quantile = self.calculate_aggregated_quantiles(self.counterfactual, daily_agg=daily_agg, agg_level=agg_level, is_nonzero=is_nonzero)

        num_cols = 1  # Set the number of columns to 1
        num_rows = len(control_quantile)

        subplot_size = (16, 6)
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(subplot_size[0], num_rows * (subplot_size[1]+6)))
        
        if num_rows == 1:
            axes = [axes]
        
        self.quantile_range = np.arange(0, 1.01, 0.01)

        for ax, item in enumerate(control_quantile.keys()):

            # Plotting the quantile distributions
            axes[ax].plot(self.quantile_range, control_quantile[item], linestyle='-', label='control', color='blue')
            axes[ax].plot(self.quantile_range, treatment_u_quantile[item], linestyle='-', label='treatment (w. added usage)', color='green')
            axes[ax].plot(self.quantile_range, treatment_quantile[item], linestyle='-', label='treatment (w.o. added usage)', color='mediumaquamarine')
            axes[ax].plot(self.quantile_range, counterfactual_quantile[item], linestyle='-', label='counterfactual', color='red')
            axes[ax].set_title('Quantile Distribution - {}'.format(item))
            plt.xlabel('Quantiles')
            plt.ylabel('Energy Cost')
            axes[ax].legend(fontsize=11)
            axes[ax].grid()
            
            # Creating a subplot within the current axes for the table
            subax = fig.add_subplot(axes[ax].get_subplotspec(), frame_on=False)

            subax.set_xticks([])
            subax.set_yticks([])
            subax.set_xticklabels([])
            subax.set_yticklabels([])

            plt_dataframe = self.create_quantile_table(treatment_quantile[item],
                                                    treatment_u_quantile[item],
                                                    control_quantile[item], 
                                                    counterfactual_quantile[item])

            tab = table(subax, plt_dataframe,
                        loc='bottom', cellLoc='center', bbox=[0, -0.15, 1, 0.1], edges='horizontal')

            tab.auto_set_font_size(False)
            tab.set_fontsize(12)

        plt.show()

class Hypothesis_Testing:
    '''
    A class for conducting hypothesis testing on cost differences
    Parameters:
    ----------
    control
        A dictionary of control dataframes for each device.
    treatment_usage
        A dictionary of treatment dataframes for each device.
    treatment_added_usage
        A dictionary of treatment dataframes with added usage for each device.
    counterfactual
        A dictionary of counterfactual dataframes for each device.
    '''
    def __init__(
        self,
        control,
        treatment_usage,
        treatment_added_usage,
        counterfactual,

    ):
        import numpy as np
        self.control =  {dev: df.copy() for dev, df in control.items()}
        self.treatment_usage = {dev: df.copy() for dev, df in treatment_usage.items()} 
        self.treatment_added_usage = {dev: df.copy() for dev, df in treatment_added_usage.items()} 
        self.counterfactual = {dev: df.copy() for dev, df in counterfactual.items()} 
        self.quantile_range = np.arange(0, 1.01, 0.01)

    @staticmethod
    def calculate_cost_differences(treatment_values, cf_values, c_values):
        differences_dict = {}
        differences_dict['true_diff_abs'] = (treatment_values - c_values)
        differences_dict['true_diff_rel'] = ((treatment_values - c_values) / c_values)
        differences_dict['pred_diff_abs'] = (treatment_values - cf_values)
        differences_dict['pred_diff_rel'] = ((treatment_values - cf_values)/ cf_values)
        differences_dict['diff_rel_counterf'] = (cf_values - c_values) / c_values
        return differences_dict
        
    @staticmethod
    def calculate_mean_relative_difference(treatment_values, c_values, cf_values):
        true_diff_rel_mean = (treatment_values.sum() - c_values.sum()) / c_values.sum()
        pred_diff_rel_mean = (treatment_values.sum() - cf_values.sum()) / cf_values.sum()
        return true_diff_rel_mean, pred_diff_rel_mean
        
    @staticmethod
    def replace_nan(array):
        import numpy as np
        return array.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    @staticmethod
    def trim_outliers(diff, trim_percent=1):
        import numpy as np
        lower_threshold = np.percentile(diff, trim_percent)
        upper_threshold = np.percentile(diff, 100 - trim_percent)
        trimmed_diff = np.clip(diff, lower_threshold, upper_threshold)
        return trimmed_diff

    def create_difference_table(self, agg_level='all', treatment_with_added_usage = False):
        from copy import deepcopy
        import numpy as np
        
        aggregation = f'{"daily"}_{agg_level}'
        
        if treatment_with_added_usage:
            treatment_agg = deepcopy(self.treatment_added_usage[aggregation])
        else:
            treatment_agg = deepcopy(self.treatment_usage[aggregation])
        counterfactual_agg = deepcopy(self.counterfactual[aggregation])
        control_agg = deepcopy(self.control[aggregation])
        
        true_diff_abs= {}
        true_diff_rel= {}
        pred_diff_abs= {}
        pred_diff_rel= {}
        true_diff_rel_mean= {}
        pred_diff_rel_mean= {}
        diff_rel_counterf= {}
        
        for item in control_agg.keys():

            t_values = treatment_agg[item]['total_cost']
            cf_values = counterfactual_agg[item]['total_cost']
            c_values = control_agg[item]['total_cost']
            
            difference_arrays = self.calculate_cost_differences(t_values,cf_values,c_values)
            true_diff_rel_mean[item], pred_diff_rel_mean[item] = self.calculate_mean_relative_difference(t_values, c_values, cf_values)
                
            for key, value in difference_arrays.items():
                difference_arrays[key] = self.replace_nan(value)
                difference_arrays[key] = self.trim_outliers(difference_arrays[key])
        
            true_diff_abs[item] = difference_arrays['true_diff_abs']
            pred_diff_abs[item] = difference_arrays['pred_diff_abs']
            true_diff_rel[item] = difference_arrays['true_diff_rel']
            pred_diff_rel[item] = difference_arrays['pred_diff_rel']
            diff_rel_counterf[item] = difference_arrays['diff_rel_counterf']

        self.rel_diff_true = true_diff_rel
        self.rel_diff_pred = pred_diff_rel
        
        self.rel_diff_counterfactual = diff_rel_counterf
            
        diff_data = {
            'True diff Abs': [int(np.mean(true_diff_abs[item])) for item in control_agg],
            'Pred diff Abs': [int(np.mean(pred_diff_abs[item])) for item in control_agg],
            'True diff Rel': [round(true_diff_rel_mean[item],3) for item in control_agg],
            'Pred diff Rel': [round(pred_diff_rel_mean[item],3) for item in control_agg],
        }

        result_table = pd.DataFrame(diff_data).T
        result_table.columns = [f'{item}' for item in control_agg]
        self.result_table = result_table
        
    def plot_cost_changes(self):
        import numpy as np
        import matplotlib.pyplot as plt 
        
        num_cols = 1
        num_rows = len(self.rel_diff_true)

        subplot_size = (10, 6)
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(subplot_size[0], num_rows * subplot_size[1]))
        
        if num_rows == 1:
            axes = [axes]
            
        for ax, item in enumerate(self.rel_diff_true.keys()):

            axes[ax].plot(self.rel_diff_true[item],label='true relative difference,     mean={}'.format(round(np.mean(self.rel_diff_true[item]),2)),color='green')
            axes[ax].plot(self.rel_diff_pred[item],label='pred relative difference,    mean={}'.format(round(np.mean(self.rel_diff_pred[item]),2)),color='lightgreen')
            axes[ax].plot(self.rel_diff_counterfactual[item],label='relative forecast error,       mean={}'.format(round(np.mean(self.rel_diff_counterfactual[item]),2)),color='red')
            axes[ax].axhline(0, color='black', linestyle='--')
            axes[ax].legend()
            
            axes[ax].set_title('Relative Difference to Control Set - {}'.format(item))
            plt.xlabel('Day')
            plt.ylabel('Daily Change in %')
            axes[ax].legend(fontsize=11)
            axes[ax].grid()

    @staticmethod
    def check_normality(data, alpha=0.05):
        import scipy.stats as st
        
        return st.shapiro(data).pvalue
    
    def perform_significance_test(self, data1, data2=None, alpha=0.05):
        import scipy.stats as st
        import numpy as np
        
        if (data1 == 0).all():
            # "No change in treatment, skip"
            return np.nan
        
        is_data1_normal = self.check_normality(data1)
        
        if data2 is None:
            if is_data1_normal > alpha:
                # Data1 is normally distributed, perform a t-test
                _, p_value = st.ttest_ind(data1, np.zeros_like(data1))
            else:
                # Data1 is not normally distributed, perform a Wilcoxon test
                _, p_value = st.wilcoxon(data1)
        else:
            is_data2_normal = self.check_normality(data2)
            
            if is_data1_normal > alpha and is_data2_normal > alpha:
                # Both data1 and data2 are normally distributed, perform a t-test
                _, p_value = st.ttest_ind(data1, data2)
            else:
                # Either data1 or data2 (or both) is not normally distributed, perform a Wilcoxon test
                _, p_value = st.wilcoxon(data1, data2)
        
        p_value = round(p_value, 3)
        
        return p_value
    
    def perform_hypothesis_test(self, alpha=0.05):
        import numpy as np
        
        p_rel_diff_true = {}
        p_rel_diff_pred = {}
        p_rel_diff_counterfactual = {}
        
        for key in self.rel_diff_true.keys():
            
            p_rel_diff_true[key] = self.perform_significance_test(self.rel_diff_true[key], data2=None, alpha=0.05)
            p_rel_diff_pred[key] = self.perform_significance_test(self.rel_diff_pred[key], data2=None, alpha=0.05)
            p_rel_diff_counterfactual[key] = self.perform_significance_test(self.rel_diff_pred[key], data2=self.rel_diff_counterfactual[key], alpha=0.05)
        
        result_table_complete = self.result_table.copy().T
        result_table_complete['True H(1) p-value'] = p_rel_diff_true.values()
        result_table_complete['Pred H(1) p-value'] = p_rel_diff_pred.values()
        result_table_complete['Pred H(2) p-value'] = p_rel_diff_counterfactual.values()
        
        result_table_complete['True Significant'] = np.where(
            (result_table_complete['True H(1) p-value']<= alpha)&(result_table_complete['True diff Abs']<0),
            u'\N{check mark}', 'x')    
        result_table_complete['Pred Significant'] = np.where(
            (result_table_complete['Pred H(1) p-value']<= alpha)&(result_table_complete['Pred H(2) p-value']<= alpha)&(result_table_complete['Pred diff Abs']<0)
            , u'\N{check mark}', 'x')    
            
        return result_table_complete.T

    def pipeline(self, agg_level='all', treatment_with_added_usage = False, alpha=0.05, plot_changes = True):
    
        self.create_difference_table(agg_level=agg_level,
                                    treatment_with_added_usage = treatment_with_added_usage)
        
        if plot_changes:
            self.plot_cost_changes()
        
        print(self.perform_hypothesis_test(alpha))
