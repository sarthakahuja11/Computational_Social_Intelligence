# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 00:15:23 2021

@author: sarth
"""

import pandas as pd                                                    #used for loading data
import math

training_set = pd.read_csv("training-part-2.csv")
test_set = pd.read_csv("test-part-2.csv")

# Calculating the Standard Deviation of feature "i" in class "k" gaussian
training_std = training_set.groupby('Class').std()

# Calculating the Average of feature "i" in class "k" gaussian
training_mean = training_set.groupby('Class').mean()

# List of each facial muscle ['AU_01 r', 'AU02 r' ... ]
facial_muscles = list(test_set.columns.values)[:-1]


# Function takes a facial expression 'smile' or 'frown' and a row instance
# log p(x|Ck)sn

def gaussian_function(facial_expression, feature_vector):
    probability = 0
    
    # Loop over each Activation Unit
    for muscle in facial_muscles:
        # Find log 2pi * std deviation of class ik (left hand side)
        square_root2pi = math.sqrt((2 * math.pi))
        left_hand_side = math.log((square_root2pi * training_std[muscle][facial_expression]))

        # Find Euclidean Distance divided by 2 * standard deviation squared of class ik (right hand side)
        euclidean_distance = (feature_vector[muscle] - training_mean[muscle][facial_expression])**2
        std_dev_squared = 2 * (training_std[muscle][facial_expression]**2)
        right_hand_side = euclidean_distance / std_dev_squared

        # Take left hand side and right hand side and add them to them to the running total
        probability += right_hand_side + left_hand_side
        #Considering the prior to be equal hence - neglecting

    return -float(probability)

# Return Gaussian Decimal Error Rate
def error_rate():
    error_count = []
    # Pass each row[i] of the test data to the gaussian function
    for i in range((test_set.shape[0])):

        test_case = test_set.iloc[[i]]
        correct_class = test_case['Class'].tolist()[0]

        # Get posteriors of each class and choose one with greater posterior
        smile_posterior = gaussian_function('smile', test_case)
        #print(smile_posterior)
        frown_posterior = gaussian_function('frown', test_case)
        classification_decision = ('smile' if smile_posterior > frown_posterior else 'frown')

        # Check if classification matches the correct label and store result
        error_count.append(classification_decision == correct_class)

    return "Error Rate: {}%".format(error_count.count(False) / len(error_count) *100)


print(error_rate())




