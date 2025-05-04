# Quick Start: Running Feed Forward model on gift-eval benchmark

# This notebook shows how to run the Feed Forward model on the gift-eval benchmark.
# Make sure you download the gift-eval benchmark and set the `GIFT-EVAL` environment 
# variable correctly before running this notebook.
# We will use the `Dataset` class to load the data and run the model. 
# If you have not already please check out the [dataset.ipynb](./dataset.ipynb) notebook
# to learn more about the `Dataset` class. 
# We are going to just run the model on two datasets for brevity. 
# But feel free to run on any dataset by changing the `short_datasets` and `med_long_datasets` variables below.
# We will use the `SimpleFeedForwardEstimator` class from gluonts 
# to run the Feed Forward model. It demonstrates how to use the `gluonts` estimator 
# interface to train and evaluate a deep learning model.

import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
short_datasets = "m4_weekly"

# med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "bizitobs_l2c/H"

# Get union of short and med_long datasets
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

dataset_properties_map = json.load(open("dataset_properties.json"))

from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    MSIS,
    RMSE,
    NRMSE,
    ND,
    MeanWeightedSumQuantileLoss,
)

# Instantiate the metrics
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]




## Training and Evaluation

# We will use the `evaluate_model` function to evaluate the model. 
# This function is a helper function to evaluate the model on the test data 
# and return the results in a dictionary. 
# We are going to follow the naming conventions explained in the [README](../README.md) file to store the results in a csv file called `all_results.csv` under the `results/feedforward` folder.

# The first column in the csv file is the dataset config name which is a combination of the dataset name, 
# frequency and the term:
# ```python
# f"{dataset_name}/{freq}/{term}"
# ```

from gluonts.model import evaluate_model
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
import csv
import os
import time
from gluonts.time_feature import get_seasonality
from gift_eval.data import Dataset

# Iterate over all available datasets

output_dir = "../results/feedforward"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the path for the CSV file
csv_file_path = os.path.join(output_dir, "all_results.csv")

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow(
        [
            "dataset",
            "model",
            "eval_metrics/MSE[mean]",
            "eval_metrics/MSE[0.5]",
            "eval_metrics/MAE[0.5]",
            "eval_metrics/MASE[0.5]",
            "eval_metrics/MAPE[0.5]",
            "eval_metrics/sMAPE[0.5]",
            "eval_metrics/MSIS",
            "eval_metrics/RMSE[mean]",
            "eval_metrics/NRMSE[mean]",
            "eval_metrics/ND[0.5]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "domain",
            "num_variates",
        ]
    )

for ds_name in all_datasets:
    ds_key = ds_name.split("/")[0]
    print(f"Processing dataset: {ds_name}")
    terms = ["short", "medium", "long"]
    for term in terms:
        if (
            term == "medium" or term == "long"
        ) and ds_name not in med_long_datasets.split():
            continue

        if "/" in ds_name:
            ds_key = ds_name.split("/")[0]
            ds_freq = ds_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = ds_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = dataset_properties_map[ds_key]["frequency"]

        ds_config = f"{ds_key}/{ds_freq}/{term}"
        # Initialize the dataset
        to_univariate = (
            False
            if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
        season_length = get_seasonality(dataset.freq)
        # Create mapping from dataset properties CSV

        estimator = SimpleFeedForwardEstimator(
            prediction_length=dataset.prediction_length,
            context_length=dataset.prediction_length,
            trainer_kwargs=dict(
                max_epochs=1,
            ),
        )
        predictor = estimator.train(dataset.validation_dataset)

        # Measure the time taken for evaluation
        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=512,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )

        # Append the results to the CSV file
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    ds_config,
                    "feedforward",
                    res["MSE[mean]"][0],
                    res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0],
                    res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0],
                    res["sMAPE[0.5]"][0],
                    res["MSIS"][0],
                    res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0],
                    res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    dataset_properties_map[ds_key]["domain"],
                    dataset_properties_map[ds_key]["num_variates"],
                ]
            )

        print(f"Results for {ds_name} have been written to {csv_file_path}")



## Results
# Running the above cell will generate a csv file called `all_results.csv` 
# under the `results/feedforward` folder containing the results for the Feed Forward model on the gift-eval benchmark. The csv file will look like this:

import pandas as pd
df = pd.read_csv("../results/feedforward/all_results.csv")
print(df)