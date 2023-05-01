# KAMEL
Trajectory imputation is the process of filling the gaps between observed GPS points, which has an important effect in increasing the accuracy of the trajectories and their applications. KAMEL is a scalable BERT-based system for trajectory imputation. KAMEL does not need to know the underlying road network. Instead, KAMEL maps this problem to finding a missing word and NLP, and uses state-of-the-art NLP tools to solve it. KAMEL utilizes the BERT NLP model for this task. However, BERT, as is, does not lend itself to the special characteristics of the trajectory imputation problem. Hence, KAMEL architecture is composed of several modules that adapt the nature of trajectory data to be more fit for BERT input and output, inject spatial-awareness in both the input and output of BERT, and allow imputing multiple points for each trajectory gap.



# Content Overview
This repository contains supplementary materials for the KAMEL paper. In particular, it provides the following: 

## 1. KAMEL code and its imputation algorithm. 
    
This can be found in the `kamel.py` file.

## 2. KAMEL trained models.

This can be found in the `models` directory.

## 3. Driver to run the imputation.
    
This can be found in the `driver.py` file.


# Dataset
The dataset used in the experiments are publicly available and can be obtained from their original sources: 
    
**1. Porto Dataset:** Can be downloaded from the following resources: 

    - https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
    - https://star.cs.ucr.edu/?portotaxi#center=42.23,7.99&zoom=5


**2. Jakarta Dataset:** Can be obtained from the authors of the following paper:

    Xiaocheng Huang, Yifang Yin, Simon Lim, Guanfeng Wang, Bo Hu, Jagannadan Varadarajan, Shaolin Zheng, Ajay Bulusu, and Roger Zimmermann. 2019. Grab-Posisi: An Extensive Real-Life GPS Trajectory Dataset in Southeast Asia. In Proceedings of the 3rd ACM SIGSPATIAL International Workshop on Prediction of Human Mobility (PredictGIS'19). Association for Computing Machinery, New York, NY, USA, 1–10. https://doi.org/10.1145/3356995.3364536
 


# KAMEL Training

KAMEL uses BERT as a black box. Therefore, the code and architecture for training and building the models follow the published BERT model in the pages below:

- Training: 

    - https://github.com/google-research/bert
    - https://colab.research.google.com/drive/1nVn6AFpQSzXBt8_ywfx6XR8ZfQXlKGAz
