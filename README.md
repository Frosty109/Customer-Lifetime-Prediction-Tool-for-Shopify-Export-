# **Customer Lifetime Value Analysis Tool**

A Python-based tool for analyzing customer purchasing behavior and predicting customer lifetime value using Bayesian statistical models.
Features

Transaction data analysis and visualization
Customer purchase frequency analysis
Time between purchases analysis
Customer retention probability predictions
Expected future purchase predictions
Customer lifetime value (CLV) calculations
Automated export of analysis results and visualizations

## **Requirements**

Copypython
pandas
numpy
matplotlib
pymc
arviz
pymc_marketing
pytensor
Input Data Format
The tool expects a CSV file with the following columns:

Email
Created at
Payment Reference
Total
Billing Name

## **Outputs** 

The tool generates several exports in the exports/ directory:
CSV Reports

Expected number of purchases
Expected average order value
Customer lifetime value summary

## **Visualizations**

Frequency distribution of customers
Average days between purchases
Probability matrices
Expected purchase predictions
Customer repurchase probability
Average order value analysis
Customer lifetime value analysis

## **Usage**

Place your transaction data CSV in the same directory as the script
Run the script:
bashCopypython main.py

Check the exports/ directory for results

## **Statistical Models Used**

BG/NBD (Beta-Geometric/Negative Binomial Distribution) model for purchase frequency
Gamma-Gamma model for monetary value prediction
Bayesian inference for parameter estimation
