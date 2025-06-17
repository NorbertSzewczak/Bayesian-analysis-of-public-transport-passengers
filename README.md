# Bayesian Analysis of Public Transport Passengers

This project aims to create two Bayesian models for predicting the number of public transport passengers in London based on available weather data. The analysis uses historical data on passenger counts and weather conditions to build predictive models and compare their performance.

## Project Overview

The goal of this project is to analyze the relationship between weather conditions and public transport usage in London. By applying Bayesian statistical methods, we can quantify uncertainty in our predictions and gain insights into how different weather factors influence passenger numbers.

## Data Sources

The project uses two main data sources:
- **TfL (Transport for London) data**: Daily passenger counts for tube and bus journeys from 2019-2023
- **London weather data**: Historical weather measurements from 1979-2023, including temperature, precipitation, pressure, humidity, and cloud cover

## Project Structure

- **preprocessing.ipynb**: Data cleaning, feature engineering, and preparation
- **model_1.ipynb**: Implementation of the first Bayesian model (simple linear regression)
- **model_2.ipynb**: Implementation of the second Bayesian model (hierarchical model with day-of-week effects)
- **comparison.ipynb**: Comparison of both models using information criteria and performance metrics

## Methodology

1. **Data Preprocessing**: 
   - Cleaning and merging TfL passenger data with weather data
   - Feature engineering (e.g., creating day-of-week and month indicators)
   - Normalizing numerical features
   - Train-test split (2019-2022 for training, 2023 for testing)

2. **Model 1**: 
   - Simple Bayesian linear regression
   - All predictors treated independently
   - Informative priors based on domain knowledge

3. **Model 2**: 
   - Hierarchical Bayesian model
   - Day-of-week effects modeled with a hierarchical structure
   - Allows for correlation between similar days

4. **Model Comparison**:
   - WAIC (Widely Applicable Information Criterion)
   - PSIS-LOO (Pareto-Smoothed Importance Sampling Leave-One-Out cross-validation)
   - Posterior predictive checks
   - Out-of-sample prediction accuracy

## Results

The models provide insights into how weather conditions affect public transport usage in London, with the hierarchical model capturing day-of-week patterns more effectively. Detailed results and model comparison can be found in the respective notebooks.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- cmdstanpy
- arviz

## Usage

1. Run `preprocessing.ipynb` to prepare the data
2. Run `model_1.ipynb` and `model_2.ipynb` to fit the models
3. Run `comparison.ipynb` to compare model performance