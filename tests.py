import numpy as np
import pandas as pd 
import random
from model import model_fit

def data_from_model(baseline_params, startle_scaling_params, sound_scaling_params, 
                    prepulse_conditions, startle_sounds_each_condition, noise=0.01):
    data = []
    for i, condition in enumerate(prepulse_conditions):
        startle_scaling = startle_scaling_params[i]
        sound_scaling = sound_scaling_params[i]
        for startle_sound in startle_sounds_each_condition[condition]:
            model_point = sigmoid(startle_sound, *baseline_params, startle_scaling, sound_scaling)
            jitter_point = model_point + random.uniform(-noise * baseline_params[0], noise*baseline_params[0])
            data.append([condition[0], condition[1], startle_sound, jitter_point])
    return data



# Should accurately fit baseline and one prepulse condition
data = data_from_model([2, 0.2, 35], [1, 0.8], [1, 0.94], [(0, 100), (14, 100)], 
                        {(0, 100): [0, 20, 30, 50, 60], (14,100): [0, 20, 30, 50, 60]})
model = model_fit(data)


# Should only fit the baseline condition and throw out the other condition because 
# it doesn't have enough startle sound levels
data = data_from_model([2, 0.2, 35], [1, 0.8], [1, 0.94], [(0, 100), (14, 100)], 
                        {(0, 100): [0, 20, 30, 50, 60], (14,100): [0, 20, 30]})
model = model_fit(data)


# Make sure it can handle just a baseline curve
data = data_from_model([2, 0.2, 35], [1], [1], [(0, 100)], 
                        {(0, 100): [0, 20, 30, 50, 60]})
model = model_fit(data)


# Test situation where a baseline condition doesn't come close to covering the startle curve
# Output a warning that the fractional satuaration is low
data = data_from_model([2, 0.2, 35], [1, 0.8], [1, 0.94], [(0, 100), (14, 100)], 
                        {(0, 100): [0, 5, 10, 20, 30], (14,100): [0, 20, 30, 40, 50]})
model = model_fit(data)


# Test situation where a prepulse condition doesn't come close to covering the startle curve
# This should not throw an error, although the situation should be avoided.
data = data_from_model([2, 0.2, 35], [1, 0.8], [1, 0.94], [(0, 100), (14, 100)], 
                        {(0, 100): [0, 20, 30, 50, 60], (14,100): [0, 5, 10, 20]})
model = model_fit(data)


# Test not having a control startle sound for the baseline prepulse condition
data = data_from_model([4, 0.15, 35], [1, 0.8], [1, 0.94], [(0, 100), (14, 100)], 
                        {(0, 100): [20, 30, 50, 60], (14,100): [0, 20, 30, 50, 60]})
model = model_fit(data)



# ---------------------
# Failure cases: uncomment one at a time to run
# --------------------

# # Throw an error if the data isn't an Nx4 array or list
# data = [[0, 1, 2], [0, 1, 2]]
# model = model_fit(data)


# # Throw an error if the input isn't a numpy array or list
# data = 'hello world'
# model = model_fit(data)



# # Throw an error if we see  multiple baseline prepulse conditions
# data = data_from_model([4, 0.15, 35], [1, 1, 0.8], [1, 1, 0.94], [(0, 100), (0, 0), (14, 100)], 
#                         {(0, 100): [0, 20, 30, 50, 60], 
#                          (0, 0): [0, 20, 30, 50, 60], (14,100): [0, 20, 30, 50, 60]})
# model = model_fit(data)



# # Throw an error on the empty input
# data = []
# model = model_fit(data)




# # Should throw error that baseline condition doesn't have enough startle sound levels
# data = data_from_model([2, 0.2, 35], [1, 0.8], [1, 0.94], [(0, 100), (14, 100)], 
#                         {(0, 100): [0, 20, 30], (14,100): [0, 20, 30, 50, 60]})
# model = model_fit(data)




# # Throw an error that data needs to include a baseline condition (prepulse sound = 0 dB above baseline)
# data = data_from_model([2, 0.2, 35], [0.9], [0.94], [(14, 100)], 
#                         {(14, 100): [0, 20, 30, 50, 60]})
# model = model_fit(data)




# # Throw an error that the data must contain at least one control stimulus (startle sound = 0 dB above baseline)
# data = data_from_model([2, 0.2, 35], [1, 0.8], [1, 0.94], [(0, 100), (14, 100)], 
#                         {(0, 100): [20, 30, 50, 60], (14,100): [20, 30]})
# model = model_fit(data)


