# Credit_Risk_Analysis
Credit risk analysis by using imbalanced-learn and scikit-learn libraries to build and evaluate machine learning models.

## Overview
Credit risk analysis was performed by machine learning models built using imbalanced-learn and scikit-learn libraries. Analysis aims to demonstrate the effectiveness of machine learning models in the following steps:
1. Oversample data with the <RandomOverSampler> and <SMOTE> algorithms
2. Undersample the data with <ClusterCentroids> algorithm
3. Combine over- and undersampling methods with <SMOTEENN> algorithm.
4. Compare <BalancedRandomForestClassifier> and <EasyEnsembleClassifier> machine learning models that reduce bias to predict credit risk.

## Results

* Naive Random Oversampling with <RandomOverSampler> algorithm:<br>
![random_oversampling](images/naive_random_oversampling.png)<br>

  * Balanced Accuracy Score: 0.62
  * Scores on high risk credit prediction:
    - Precision: 0.01
    - Sensitivity(Recall): 0.60
    - F1 harmonic: 0.02
    - Predicted True: 52
    - Predicted False: 35
  * Scores on low risk credit prediction:
    - Precision: 1.0
    - Sensitivity(Recall): 0.64
    - F1 harmonic: 0.78
    - Predicted True: 6082
    - Predicted False: 11036

* SMOTE Oversampling:<br>
![smote](images/smote_oversampling.png)<br>

  * Balanced Accuracy Score: 0.65
  * Scores on high risk credit prediction:
    - Precision: 0.01
    - Sensitivity(Recall): 0.63
    - F1 harmonic: 0.02
    - Predicted True: 55
    - Predicted False: 32
  * Scores on low risk credit prediction:
    - Precision: 1.0
    - Sensitivity(Recall): 0.67
    - F1 harmonic: 0.80
    - Predicted True: 5580
    - Predicted False: 11538

* Undersampling with <ClusterCentroids> algorithm:<br>
![undersampling](images/cluster_centroid_undersampling.png)

  * Balanced Accuracy Score: 0.49
  * Scores on high risk credit prediction:
    - Precision: 0.00
    - Sensitivity(Recall): 0.55
    - F1 harmonic: 0.01
    - Predicted True: 48
    - Predicted False: 39
  * Scores on low risk credit prediction:
    - Precision: 0.99
    - Sensitivity(Recall): 0.43
    - F1 harmonic: 0.60
    - Predicted True: 9728
    - Predicted False: 7390

* Combination Sampling with <SMOTEENN> algorithm:<br>
![smoteenn](images/combination_smoteen.png)<br>

  * Balanced Accuracy Score: 0.62
  * Scores on high risk credit prediction:
    - Precision: 0.01
    - Sensitivity(Recall): 0.60
    - F1 harmonic: 0.02
    - Predicted True: 52
    - Predicted False: 32
  * Scores on low risk credit prediction:
    - Precision: 1.00
    - Sensitivity(Recall): 0.65
    - F1 harmonic: 0.79
    - Predicted True: 5995
    - Predicted False: 11123

## Summary
