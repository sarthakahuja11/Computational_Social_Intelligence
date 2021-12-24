# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 01:33:49 2021

"""
# -*- coding: utf-8 -*-

import csv
import numpy as np
from collections import defaultdict

categories = defaultdict(list)

#Using DictReader to read file
with open('laughter-corpus.csv', newline='') as csvfile:
  dataset = csv.DictReader(csvfile)
  for instance in dataset:
    for (i,j) in instance.items():
      categories[i].append(j)

gender = np.asarray(categories['Gender'])
role = np.asarray(categories['Role'])
duration = np.asarray(categories['Duration'])

#male candidates
male_candidates = gender[np.where(gender=="Male")].size
#print(male_candidates)

#female candidates
female_candidates = gender[np.where(gender=="Female")].size
#print(female_candidates)

#caller candidates
caller_candidates = role[np.where(role=="Caller")].size
#print(caller_candidates)

#receiver candidates
receiver_candidates = role[np.where(role=="Receiver")].size
#print(receiver_candidates)

total_laughter_events = len(categories[i])
print('total number of laughter events:%d'%(total_laughter_events))

#probability of candidate being male
prob_male = 57/120

#probability of candidate being male
prob_female = 63/120

#probability of candidate being a caller or a receiver
prob_caller = prob_receiver = 0.5

#function for compute chi_square
def compute_chi_square(expected_val, observed_val):
  difference = observed_val - expected_val
  difference_squared = np.square(difference)
  comp = difference_squared/expected_val
  chi_square = np.sum(comp)
  return chi_square

def compute_t_value(matrix1, matrix2):

  average_mat1 = np.average(matrix1)
  std_dev1 = np.std(matrix1)
  variance1 = np.square(std_dev1)
  n1 = np.size(matrix1)
  
  average_mat2 = np.average(matrix2)
  std_dev2 = np.std(matrix2)
  variance2 = np.square(std_dev2)
  n2 = np.size(matrix2)

  #formula for t-value
  t_value = np.absolute(average_mat1 - average_mat2)/np.sqrt((np.square(variance1)/n1)+(np.square(variance2)/n2))
  return t_value

#Q.1: Is the number of laughter events higher for women than for men?
observed_val = np.matrix([male_candidates, female_candidates])
expected_val = np.matrix([prob_male*total_laughter_events, prob_female*total_laughter_events])
chi_square = compute_chi_square(expected_val, observed_val)

print(chi_square)

#Q.2: Is the number of laughter events higher for callers than for receivers?
observed_val = np.matrix([caller_candidates, receiver_candidates])
expected_val = np.matrix([prob_caller*total_laughter_events, prob_receiver*total_laughter_events])
chi_square = compute_chi_square(expected_val, observed_val)

print(chi_square)

#Q.3: Are laughter events longer for women?
duration_for_male_candidate = np.matrix(duration[np.where(gender=='Male')]).astype(np.float)
#print(duration_for_male_candidate)

duration_for_female_candidate = np.matrix(duration[np.where(gender=='Female')]).astype(np.float)
#print(duration_for_female_candidate)

#calculate t_value
t_value = compute_t_value(duration_for_male_candidate, duration_for_female_candidate)
print('t_value:%f'%(t_value))

#Q.4: Are laughter events longer for callers?
duration_for_caller = np.matrix(duration[np.where(role=='Caller')]).astype(np.float)
#print(duration_for_caller)

duration_for_receiver = np.matrix(duration[np.where(role=='Receiver')]).astype(np.float)
#print(duration_for_receiver)

#calculate t_value
t_value2 = compute_t_value(duration_for_caller, duration_for_receiver)
print('t_value2:%f'%(t_value2))

