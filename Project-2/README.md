# Project 2

Here you will find the python notebook we wrote for BDCC's second project, predicting lenght of stay (LOS) of patients in a hospital using the MIMIC-III dataset.  

You will find 3 notebooks:
- setup
- data
- models

setup does an initial conversion of the files, converting them from csv format to parquet format.
data covers the exploration and pre-processing of the data and the building of the dataset for the ML models.
models builds a few machine learning models on the built dataset and does predictions and analysis of the results.

And if you want to run them, althought they are already provided with results below the code, you should run them in that order and will also need to download a few of the csv files of the dataset, namely:

- CHARTEVENTS
- ICUSTAYS
- D_ICD_DIAGNOSES
- PATIENTS
- DIAGNOSES_ICD
- ADMISSIONS