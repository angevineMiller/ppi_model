# PPI Model

This repository contains Python code implementing a new model of prepulse inhibition of the acoustic startle response (PPI). This will fit the model to acoustic startle data for an individual animal. Please refer to the associated manuscript for a detailed description of the model and how it can be used. 

In this repo, the Jupyter notebook **'ppi_model_demo.ipynb'** demos how to use the model. Below we have included a short walk-through indicating the requirements for using the model:

## Terminology
* *prepulse sound level* - the intensity of the prepulse sound (dB above baseline)
* *startle sound level* - the intensity of the potentially-startling sound (dB above baseline)
* *delay* - the delay between the prepulse and startle sounds
* *prepulse condition* - a unique combination of prepulse sound level and delay, which can be measured at different startle sound levels
* *baseline prepulse condition* - a prepulse condition in which the prepulse sound level is 0 dB above baseline (i.e. no prepulse sound)
* *control stimulus* - a stimulus in which the startle sound level equals 0 dB above baseline (i.e. no startle sound)

## Data requirements for this model
1) You need exactly one baseline prepulse condition, i.e. with a prepulse sound of 0 dB above baseline.
2) The baseline prepulse condition must be measured at a minimum of four different startle sound levels. To optimally fit the model, you should ideally use a range of startle sounds from 0 dB up to a sound level that will elicit a startle near the animal's maximum. 
3) If you want to study PPI, you need at least one non-baseline prepulse condition, and this too needs to be measured at a minimum of four different startle sound levels. These do *not* have to be the same startle sound levels at which you measured the baseline prepulse condition. Any prepulse conditions with less than four startle sound levels will be dropped from the model.
4) You need at least one control stimulus, where the startle sound level equals 0 dB above baseline. You do not need to measure a control stimulus for every prepulse condition, but doing so will make fitting the model easier.
5) For each stimulus--i.e. unique combination of prepulse condition and startle sound level--you will need an average startle response for each individual animal. Importantly, this average should only be done after taking the log of the animal's startle responses for individual trials (see below).

## Take the log of the startle responses before averaging
We have shown that the distribution of startle responses is non-Gaussian and closer to a lognormal distribution within individual animals. It is therefore not appropriate to take the average of the raw startle responses. While the lognormal is also not a perfect fit, it is unquestionably better than a normal distribution. Therefore, we take the log of the trial-by-trial startle responses before averaging across the trials within a stimulus for a given animal.


## Parameters of the model_fit() function

**input_data** : Nx4 Numpy array or list
The input data must be either a list or a Numpy array, and either way it must be shaped Nx4 (i.e. four columns and any number of rows). Each row is a single stimulus and the animal's average startle response to that stimulus. The rows can occur in any order, but the columns must be in the following order: 
1) prepulse sound level
2) delay
3) startle sound level
4) average startle response for this animal at this stimulus

**plot_model** : boolean (True or False)
If True, the function will plot the startle response vs. startle sound data for all prepulse conditions on a single figure, overlaid with the PPI model curves. We always recommend plotting every animal's model fit to ensure that the model accurately fit the data, as it is possible that some inputs could fail to converge on an appropriate model.

For more info, calling help() on any of the functions in this repo will provide information on the expected input/output behavior of the function.


## Output of the model_fit() function
The model_fit function will return a dictionary containing all of the parameters of the model and some other metrics. The structure of the dictionary is the following:

* 'baseline_parameters' : list - baseline saturation, baseline midpoint, baseline slope
* 'startle_scaling_parameters' : dict - mapping from (prepulse sound level, delay) -> startle scaling for that condition
* 'sound_scaling_parameters' : dict - mapping from (prepulse sound level, delay) -> sound scaling for that condition
* 'baseline_startle_threshold' : minimum sound required for baseline curve to reach 5% of saturation
* 'baseline_fraction_saturation' :  M / S, where M = startle response to the maximal startle sound level in the baseline prepulse condition, and S = baseline saturation
* 'model_fitting_error' : total RMSE error between the data and the model across all prepulse conditions

Note that a low fraction saturation suggests that you might not have measured loud enough startle sounds in the baseline condition to accurately estimate the animal's baseline saturation point.
