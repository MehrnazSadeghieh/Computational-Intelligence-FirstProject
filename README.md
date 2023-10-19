# Computational-Intelligence-FirstProject

## Seizure Detection from EEG Signals

This project aims to implement an intelligent system for detecting seizures in epileptic patients from EEG signals. The primary tasks involve preprocessing the EEG signals, feature extraction, feature enhancement, and signal classification. The final goal is to achieve accurate seizure detection through the application of machine learning and deep learning methods.

## Dataset Description

The provided dataset contains 500 signal segments, each 23.6 seconds long, sampled at 173.61 Hz. Each data file consists of 4097 data points. These data files are categorized into five different groups. For a detailed description of the groups and access to the dataset, please refer to the [link](https://www.upf.edu/web/ntsa/downloads/-/asset_publisher/xvT6E4pczrBw/content/2001-indications-of-nonlinear-deterministic-and-finite-dimensional-structures-in-time-series-of-brain-electrical-activity-dependence-on-recording-regi?inheritRedirect=false&redirect=https%3A%2F%2Fwww.upf.edu%2Fweb%2Fntsa%2Fdownloads%3Fp_p_id%3D101_INSTANCE_xvT6E4pczrBw%26p_p_lifecycle%3D0%26p_p_state%3Dnormal%26p_p_mode%3Dview%26p_p_col_id%3Dcolumn-1%26p_p_col_count%3D1![image](https://github.com/MehrnazSadeghieh/Computational-Intelligence-FirstProject/assets/68302873/931bea17-b5dd-4d34-8197-277f7f360091)
).

## Data Preprocessing and Loading

Efficient data handling and preprocessing are essential for this project. We recommend using the provided data loading solution to avoid unnecessary I/O operations. Data preprocessing is primarily related to signal processing and is crucial for achieving accurate results. 

## Feature Engineering - Assignment One

Feature engineering is a critical step in building an intelligent system. In the first assignment, the goal is to extract meaningful features from the EEG signals. You need to extract at least 15 features from three different categories, including:
- Statistics
- Innovative
- Entropies
- LBP-based features
- Time domain
- Frequency Domain

Proper feature engineering will provide a strong foundation for signal classification and seizure detection.

## Classification of Signals - The Second Task

After feature extraction, the next step is to classify the signals. You are required to implement three different classification algorithms and evaluate their performance. The evaluation should include metrics like accuracy, precision, and recall. We use k-fold cross-validation with k=5 to ensure robust evaluation.

The algorithms to be implemented are:
- Support Vector Machine (SVM)
- Random Forest
- k-Nearest Neighbors (KNN)

You will explore the impact of different classification algorithms and parameter tuning on the results. The final report should include:
- Parameter settings
- Accuracy, precision, and recall values
- Receiver Operating Characteristic (ROC) curves
- Confusion matrices

## Note
Normalization of the input data is also part of the process to ensure data consistency.

## Authors

- [Mehrnaz Sadeghieh](#)
