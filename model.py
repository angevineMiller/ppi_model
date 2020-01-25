import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def model_fit(data, plot_model=True):
    """
    Fit a sigmoid-based model of PPI to average startle data measured across a range of startle sounds
    and prepulse conditions. Returns the parameters of the model and several other key metrics. 
    Optionally plots the model overlaid on the data.

    Parameters
    ----------
    data : N x 4 Numpy array or list
        The 'data' array must have the following four columns, in this order:
            1. prepulse sound level (dB above baseline)
            2. delay between prepulse and startle sounds (ms)
            3. startle sound level (dB above baseline)
            4. average log(startle response) across all trials at this condition
        There can be any number of rows in any order, but each row must have the above
        form. Namely, each row represents a single stimulus defined by a unique prepulse sound level, 
        delay, and startle sound level with its associated average log(startle response). 
        
    plot_model : bool (default True) If True, plot the startle response vs. startle sound data for all prepulse
                 conditions on a single figure, overlaid with the PPI model curves. We recommend plotting every
                 animal's model fit to ensure that the model accurately fit the data, as it is possible that
                 some inputs could fail to converge on an appropriate model.

    Requirements
    ------------
    1. All startle responses should be logged, i.e. they represent the average of all the logged startle responses
       at given condition. This is because the acoustic startle responses is a long-tailed, non-Gaussian distribution.
    2. All prepulse and startle sound levels should be measured in dB above the background sound level.
    3. All delays (between prepulse and startle sounds) should be measured in milliseconds (ms).
    4. There must be exactly 1 baseline prepulse condition defined as a prepulse sound level of 0 dB above
       baseline, and the baseline prepulse condition must have been measured at a minimum of 4 startle sound
       levels (i.e. at least 4 rows of the data matrix must be the baseline prepulse condition, each with a
       different startle sound level).
    5. There must be at least one control stimulus defined as having startle sound level of 0 dB above baseline.
    6. All valid prepulse conditions will be measuresd at at least 4 different startle sound levels, 
       i.e. occupy at least 4 rows of the 'data' matrix with a prepulse sound level of 0 dB above baseline, 
       each with a different startle sound level. Any non-baseline prepulse conditions failing this test will
       not be included in the model.

    Returns
    -------
    model : dict
        A dictionary containing the model parameters and other important model metrics:
            'baseline_parameters' : [baseline saturation, baseline midpoint, baseline slope]
            'startle_scaling_parameters' : dict mapping (prepulse sound level, delay) -> startle scaling for that condition
            'sound_scaling_parameters' : dict mapping (prepulse sound level, delay) -> sound scaling for that condition
            'model_fitting_error' : total RMSE error between the data and the model across all prepulse conditions
            
    Examples
    --------
    See Readme for usage examples.
    """

    check_valid_type(data)  # check for correct datatype and shape
    df = pd.DataFrame(data, columns=['prepulse_sound', 'delay', 'startle_sound', 'avg_startle'])
    check_valid_content(df) # check that data meets all of the requirements
        
    # Subtract the average startle response to the control (0 dB startle sound) condition from all startle responses   
    df = subtract_control_startle(df)

    # Extract components of the data for ease/speed of processing
    prepulse_conditions, startle_sounds_each_condition = extract_prepulse_conditions(df)
    data_dict = convert_data_to_dict(df, prepulse_conditions, startle_sounds_each_condition)
    
    # Fit the model
    init_params = get_init_params(df, prepulse_conditions, startle_sounds_each_condition)  # Make initial parameter guesses for the model fit
    model_bounds = get_model_bounds_PPI(prepulse_conditions) # Generic bounds for PPI only, with scaling bounded between 0 and 1
    model_args = (data_dict, prepulse_conditions, startle_sounds_each_condition)
    model_fit = minimize(error, init_params, args=model_args, bounds=model_bounds)   
    model_error = error(model_fit.x, data_dict, prepulse_conditions, startle_sounds_each_condition)
    
    # Extract the fitted parameters and compute a few additional metrics
    baseline_params, startle_scaling_params, sound_scaling_params = extract_fitted_params(model_fit, prepulse_conditions)
    baseline_threshold = get_baseline_startle_threshold(baseline_params)
    fraction_saturation = get_fraction_saturation(baseline_params['saturation'], data_dict, 
                                                  prepulse_conditions, startle_sounds_each_condition)
    if fraction_saturation < 0.75:
        import warnings
        warnings.warn("Max baseline startle response < 75% of estimated baseline startle saturation for this animal. " + 
                      "Consider measuring the baseline prepulse condition at louder startle sound levels to better " +
                      "estimate the baseline startle response curve.")
        
    model_fit =  {'baseline_parameters': baseline_params, 
                  'startle_scaling_parameters': startle_scaling_params, 
                  'sound_scaling_parameters': sound_scaling_params,
                  'baseline_startle_threshold': baseline_threshold,
                  'baseline_fraction_saturation': fraction_saturation,
                  'model_fitting_error': model_error}
    
    if plot_model:
        plot_data_and_model(df, prepulse_conditions, startle_sounds_each_condition, model_fit)  
    return model_fit


def error(params, data_dict, prepulse_conditions, startle_sounds_each_condition):
    '''
    This is the error function that we try to minimize for the model. It returns the total
    RMSE error between the model and the actual data across all of the prepulse conditions, given the
    parameter set in the variable 'params'. 
    
    Parameters
    ----------
    params : list The set of parameters we are trying to optimize. Of note, this does not include a startle-scaling 
                  or sound-scaling parameter for the baseline prepulse condition, as they are defined to be 1.0 so 
                  are not free parameters. The ordering of the list of parameters is the following:
                  [baseline saturation, baseline slope, baseline midpoint, k1, k2, ..., kn, r1, r2, ..., rn], where
                  k1 - kn are the startle-scaling parameters for n non-baseline prepulse condition
                  r1 - rn are the sound-scaling parameters for n non-baseline prepulse conditions. 
                  The ordering from 1 - n of scaling parameters should correspond to the ordering of prepulse
                  conditions in the 'prepulse_conditions'variable after removing its first element, 
                  which is the baseline condition.
    data_dict : nested dictionary mapping: prepulse condition --> startle sound level --> average startle response,
                where prepulse_condition is a tuple of (prepulse sound level, delay)
    prepulse_conditions : list of tuples (prepulse sound level, delay) defining all of the prepulse conditions, 
                          where the first elment in the list should be the baseline prepulse condition, i.e. 
                          the condition with prepulse sound of 0 dB above baseline
    startle_sounds_each_condition : dictionary mapping prepulse condition to a list of startle sound levels at which
                                    that condition was measured: 
                                    {(prepulse sound level, delay)->[startle sound level 1, startle sound level 2, ...]}
    
    
    Returns
    -------
    RMSE error : float  The total RMSE between the model and the data across all of the prepulse conditions.
    '''
    # Extract the parameters
    num_baseline_params = 3
    num_conditions_not_baseline = len(prepulse_conditions) - 1 
    saturation, slope, midpoint = params[0:num_baseline_params]
    startle_scaling_end_idx = num_baseline_params + num_conditions_not_baseline
    sound_scaling_end_idx = startle_scaling_end_idx + num_conditions_not_baseline
    startle_scaling_not_control = params[num_baseline_params:startle_scaling_end_idx]
    sound_scaling_not_control = params[startle_scaling_end_idx:sound_scaling_end_idx]

    # Compute the total RMSE error using these parameters
    total_error = 0
    n = 0
    for i, condition in enumerate(prepulse_conditions):
        prepulse_sound, delay = condition
        startle_sounds = startle_sounds_each_condition[condition]
        for sound in startle_sounds:
            if i == 0:
                assert(prepulse_sound == 0)
                startle_scaling, sound_scaling = 1, 1  # by definition
            else:
                startle_scaling = startle_scaling_not_control[i-1]
                sound_scaling = sound_scaling_not_control[i-1]
            actual = data_dict[condition][sound]
            estimate = sigmoid(sound, saturation, slope, midpoint, startle_scaling, sound_scaling)
            total_error += (estimate - actual) ** 2
            n += 1
    return math.sqrt(total_error / n)



def sigmoid(x, saturation, slope, midpoint, startle_scaling, sound_scaling):
    '''
    Function for a single sigmoid curve (e.g. the startle response vs. startle sound curve for one prepulse condition).
    Returns the y-value (log startle response) given the x-value (startle sound) and the parameters defining the curve.
    
    Parameters
    ----------
    x : x-value to compute the function on (e.g. the startle sound level in dB above baseline)
    saturation : baseline saturation parameter
    slope : baseline slope parameter
    midpoint : baseline midpoint parameter
    startle_scaling : the PPI startle-scaling parameter for this prepulse condition. Note that percent startle-scaling
                      is defined as 100 * (1 - p), where p is this parameter.
    sound_scaling : the PPI sound-scaling parameter for this prepulse condition. Note that percent sound-scaling
                    is defined as 100 * (1 - p), where p is this parameter.
                      
    Returns
    -------
    y-value (log startle response) 
    
    '''
    return (startle_scaling * saturation) / (1 + math.exp(-slope * ((x * sound_scaling) - midpoint)))



def check_valid_type(input_data):
    '''
    Raises an error if the data is not 1) a list or numpy array, and 2) Nx2 shape.
    
    Parameters
    ----------
    input_data : this will be whatever the user passed into the model_fit() function
    
    Returns
    -------
    None
    
    Effects
    -------
    Raises errors if invalid input 
    '''
    if not isinstance(input_data, (list, np.ndarray)):
        raise ValueError('Data must be an N x 4 Numpy array or Python list. See the Readme ' +
                         'for more information.')
    if isinstance(input_data, list):
        input_data = np.array(input_data)
    
    if len(input_data) < 1:
        raise ValueError('Empty array. See the Readme for more information on the valid input requirements.')
    
    if input_data.shape[1] != 4:
        raise ValueError('Data must be an N x 4 array, where the four columns are 1) prepulse sound level, ' +
                         '2) delay, 3) startle sound level, and 4) average log(startle response). See the Readme ' +
                         'for more information.')


def check_valid_content(df):
    '''
    Raises an error if the data does not:
    1) contain a baseline prepulse condition (i.e. prepulse sound level = 0 dB above baseline) that is measured 
    at a minimum of 4 different startle sound levels (i.e. at least 4 different rows of the input)
    2) contain at least one control stimulus (i.e. startle sound level = 0 dB above baseline)
    
    Parameters
    ----------
    df : Nx4 Pandas DataFrame with columns 'prepulse_sound', 'delay', 'startle_sound', and 'avg_startle'
         Each of the N rows is a different stimulus and its associated avg(log startle response)
    
    Returns
    -------
    None
    
    Effects
    -------
    Raises an error if invalid content
    '''
    df_baseline = df[df['prepulse_sound']==0]
    
    
    # Check that there is exacltly one baseline prepulse condition
    baseline_delays = df_baseline['delay'].drop_duplicates().values
    if len(baseline_delays) > 1:
        raise ValueError("Found more than one baseline prepulse condition, i.e. condition with prepulse sound level " +
                         "of 0 dB above baseline. This model requires a single baseline prepulse condition--the delay " +
                         "can be whatever you want--and it must be measured at a minimum of 4 startle sound levels.")
    
    # Check that the baseline condition (prepulse = 0) was measured at at least 4 startle sound levels    
    if df_baseline.shape[0] < 4:
        raise ValueError('Data must include one baseline prepulse condition, i.e. a condition with prepulse sound ' +
                         '= 0 dB above background, and this condition must be measured at a miniumum of 4 different ' +
                         'startle sound levels. See the Readme for more information.')
                
    # Check for at least one control stimulus (startle sound level = 0 dB above baseline)
    df_control = df[df['startle_sound']==0]
    if df_control.shape[0] < 1:
        raise ValueError('Data must include at least one control stimulus, i.e. with a startle sound level of 0 dB ' +
                         'above background. See the Readme for more information.')
        
        
def convert_data_to_dict(df, prepulse_conditions, startle_sounds_each_condition):
    '''
    Converts Pandas DataFrame into a nested dictionary indexed in the following way: 
    prepulse condition --> startle sound level --> average startle response , where prepulse_condition 
    is a tuple of (prepulse sound level, delay)
    
    Parameters
    ----------
    df : Nx4 Pandas DataFrame with columns 'prepulse_sound', 'delay', 'startle_sound', and 'avg_startle'
         Each of the N rows is a different stimulus and its associated avg(log startle response)
    prepulse_conditions : list of tuples (prepulse sound level, delay) defining all of the prepulse conditions, 
                          where the first elment in the list should be the baseline prepulse condition, i.e. 
                          the condition with prepulse sound of 0 dB above baseline
    startle_sounds_each_condition : dictionary mapping prepulse condition to a list of startle sound levels at which
                                    that condition was measured: 
                                    {(prepulse sound level, delay)->[startle sound level 1, startle sound level 2, ...]}
                                    
    Returns
    -------
    data_dict : nested dictionary mapping: prepulse condition --> startle sound level --> average startle response,
                where prepulse_condition is a tuple of (prepulse sound level, delay)
    '''
    data_dict = {}
    for condition in prepulse_conditions:
        prepulse_sound, delay = condition
        data_dict[condition] = {}
        startle_sounds = startle_sounds_each_condition[condition]
        for sound in startle_sounds:
            avg_startle = df[(df['startle_sound']==sound) & 
                               (df['prepulse_sound']==prepulse_sound) & 
                               (df['delay']==delay)]['avg_startle'].values[0]
            data_dict[condition][sound] = avg_startle
    return data_dict

def get_fraction_saturation(saturation, data_dict, prepulse_conditions, startle_sounds_each_condition):
    '''
    Returns the fraction saturation = M / S, where 
    M = startle response to the maximal startle sound level in the baseline prepulse condition
    S = baseline saturation parameter of the model, which is the asymptotic maximum of the curve
    
    A high fraction saturation means that you have measured loud enough startle sounds to effectively 
    estimate the saturation point of the curve. Low fraction saturation suggests that you might not have
    measured loud enough startle sounds in the baseline condition to effectively estimate the animal's 
    underlying startle curve saturation point.
    
    Parameters
    ----------
    saturation : baseline saturation parameter of the model fit
    data_dict : nested dictionary mapping: prepulse condition --> startle sound level --> average startle response,
                where prepulse_condition is a tuple of (prepulse sound level, delay)
    prepulse_conditions : list of tuples (prepulse sound level, delay) defining all of the prepulse conditions, 
                          where the first elment in the list should be the baseline prepulse condition, i.e. 
                          the condition with prepulse sound of 0 dB above baseline
    startle_sounds_each_condition : dictionary mapping prepulse condition to a list of startle sound levels at which
                                    that condition was measured: 
                                    {(prepulse sound level, delay)->[startle sound level 1, startle sound level 2, ...]}
                                    
    Returns
    -------
    fractional sat : float   The fraction saturation = M / S, where 
                M = startle response to the maximal startle sound level in the baseline prepulse condition
                S = baseline saturation parameter of the model, which is the asymptotic maximum of the curve 
    '''
    baseline_condition = prepulse_conditions[0]
    max_baseline_startle_sound = np.max(startle_sounds_each_condition[baseline_condition])
    max_baseline_startle = data_dict[baseline_condition][max_baseline_startle_sound]
    return max_baseline_startle / saturation



def extract_fitted_params(model_fit, prepulse_conditions):
    '''
    Parses the output of the fitting function and returns dictionaries to easily access the model parameters.
    
    Parameters
    ----------
    model_fit : list  List of model parameters returned from the fitting function.
    prepulse_conditions : list of tuples (prepulse sound level, delay) defining all of the prepulse conditions, 
                          where the first elment in the list should be the baseline prepulse condition, i.e. 
                          the condition with prepulse sound of 0 dB above baseline
                          
    Returns
    -------
    baseline_params : dict  Indexed by name ('saturation', 'slope', 'midpoint')
    startle_scaling_params : dict  Indexed by prepulse condition tuple (prepulse sound level, delay)
    sound_scaling_params : dict  Indexed by prepulse condition tuple (prepulse sound level, delay)
    '''
    num_baseline_params = 3
    num_conditions_not_baseline = len(prepulse_conditions) - 1
    saturation, slope, midpoint = list(model_fit.x[0:num_baseline_params])
    startle_scaling_vals = list(model_fit.x[num_baseline_params:(num_baseline_params+num_conditions_not_baseline)])
    sound_scaling_vals = list(model_fit.x[(num_baseline_params+num_conditions_not_baseline):])

    baseline_params = {'saturation': saturation, 'slope': slope, 'midpoint': midpoint}

    startle_scaling_params = {}
    sound_scaling_params = {}
    for i, condition in enumerate(prepulse_conditions):
        if i == 0:
            assert(condition[0] == 0)
            startle_scaling_params[condition] = 1
            sound_scaling_params[condition] = 1
        else:
            startle_scaling_params[condition] = startle_scaling_vals[i-1]
            sound_scaling_params[condition] = sound_scaling_vals[i-1]
    return baseline_params, startle_scaling_params, sound_scaling_params


def get_baseline_startle_threshold(baseline_params, max_sound=140.0, threshold=0.05):
    '''
    Returns the baseline startle threshold, defined as the minimum startle sound level required
    for the baseline startle curve to reach 5% of its asymptotic max.
    
    Parameters
    ----------
    baseline_params : dictionary  The three parameters of the baseline startle curve indexed by name 
                      ('saturation', 'slope', and 'midpoint')
    max_sound : float  The maximum sound to search for the threshold.
    threshold : float  Change this if you want to re-define threshold to be something other than 5% of saturation.
    
    Returns
    -------
    threshold : float  The minimum startle sound level required for the baseline startle response curve to be
                       5% of its asymptotic maximal value.
    
    '''
    min_sound = 0
    saturation = baseline_params['saturation']
    slope = baseline_params['slope']
    midpoint = baseline_params['midpoint']
    base_startle_scaling = 1 # by definition
    base_sound_scaling = 1
    xs = np.linspace(min_sound, max_sound, 2000)    
    ys = [sigmoid(x, saturation, slope, midpoint, base_startle_scaling, base_sound_scaling) for x in xs]
    index_above_threshold = np.where(ys > threshold * saturation)[0][0]
    return xs[index_above_threshold]


def extract_prepulse_conditions(df):
    '''
    Returns all valid prepulse conditions, i.e. those measured at >= 4 startle sound levels. Also
    returns a mapping from prepulse condition to the startle sound levels at which that condition was measured.
    
    Parameters
    ---------
    df : Nx4 Pandas DataFrame with columns 'prepulse_sound', 'delay', 'startle_sound', and 'avg_startle'
         Each of the N rows is a different stimulus and its associated avg(log startle response)
    
    Returns
    -------
    valid_conditions : list of tuples (prepulse sound level, delay) defining all of the prepulse conditions, 
                          where the first elment in the list should be the baseline prepulse condition, i.e. 
                          the condition with prepulse sound of 0 dB above baseline
    startle_sounds_each_condition : dictionary mapping prepulse condition to a list of startle sound levels at which
                                    that condition was measured: 
                                    {(prepulse sound level, delay)->[startle sound level 1, startle sound level 2, ...]}

    '''
    all_conditions = df[['prepulse_sound', 'delay']].drop_duplicates().values
    valid_conditions = []
    baseline_condition = False
    for idx, condition in enumerate(all_conditions):
        if condition[0] == 0:
            entries = df[(df['prepulse_sound']==condition[0]) & (df['delay']==condition[1])]
            assert(baseline_condition == False) # we should only find one baseline condition
            assert(entries.shape[0] >= 4)
            baseline_condition = tuple(condition)
        else:
            entries = df[(df['prepulse_sound']==condition[0]) & (df['delay']==condition[1])]
            if entries.shape[0] >= 4:
                valid_conditions.append(tuple(condition))
    valid_conditions.insert(0, baseline_condition) # always insert baseline condition in the first position
    startle_sounds_each_condition = {}
    for condition in valid_conditions:
        entries = df[(df['prepulse_sound']==condition[0]) & (df['delay']==condition[1])]
        startle_sounds_each_condition[condition] = entries['startle_sound'].values
    return valid_conditions, startle_sounds_each_condition



def subtract_control_startle(df):
    '''
    From all startle responses, subtract the average startle response to the control stimulus 
    (0 dB startle sound) across all prepulse conditions in which there was a control startle sound.
    This effectively subtracts any y-offset in the "no startle" condition, so that we only model the
    actual startle responses.
    
    Parameters
    ----------
    df : Nx4 Pandas DataFrame with columns 'prepulse_sound', 'delay', 'startle_sound', and 'avg_startle'
         Each of the N rows is a different stimulus and its associated avg(log startle response)
    
    Returns
    -------
    df : Same as the input DataFrame, except for the column 'avg_startle', where for each entry we subtracted
         the average startle to the control stimulus (i.e. the no-startle condition)
    '''
    
    avg_control_startle = df[df['startle_sound']==0]['avg_startle'].mean()
    df['avg_startle'] = df['avg_startle'].apply(lambda startle: startle - avg_control_startle)
    return df


def get_init_params(df, prepulse_conditions, startle_sounds_each_condition):
    '''
    Returns initial parameter values for the model fit. We make some guesses on the baseline parameters based
    on the baseline data, and we initialize all startle-scaling and sound-scaling parameters to 1.0 (no scaling).
    
    Parameters
    ----------
    df : Nx4 Pandas DataFrame with columns 'prepulse_sound', 'delay', 'startle_sound', and 'avg_startle'
         Each of the N rows is a different stimulus and its associated avg(log startle response)
    prepulse_conditions : list of tuples (prepulse sound level, delay) defining all of the prepulse conditions, 
                          where the first elment in the list should be the baseline prepulse condition, i.e. 
                          the condition with prepulse sound of 0 dB above baseline
    startle_sounds_each_condition : dictionary mapping prepulse condition to a list of startle sound levels at which
                                    that condition was measured: 
                                    {(prepulse sound level, delay)->[startle sound level 1, startle sound level 2, ...]}
                                    
    Returns
    -------
    init_params : list of initial parameter values to pass into the model fitting routine
    '''
    baseline_condition = prepulse_conditions[0]
    num_conditions_not_baseline = len(prepulse_conditions) - 1

    
    max_baseline_startle_sound = np.max(startle_sounds_each_condition[baseline_condition])
    init_saturation = df[(df['startle_sound']==max_baseline_startle_sound) & 
                         (df['prepulse_sound']==baseline_condition[0]) & 
                         (df['delay']==baseline_condition[1])]['avg_startle'].values[0]
    
    all_baseline_startle_sounds = df[(df['prepulse_sound']==baseline_condition[0]) & 
                                     (df['delay']==baseline_condition[1])]['startle_sound'].values
    all_baseline_avg_startles = df[(df['prepulse_sound']==baseline_condition[0]) & 
                                    (df['delay']==baseline_condition[1])]['avg_startle'].values
    ydiffs = np.diff(all_baseline_avg_startles)
    xdiffs = np.diff(all_baseline_startle_sounds) 
    diff_slopes = ydiffs / xdiffs
    init_slope = np.max(diff_slopes)
    max_slope_idx = np.where(diff_slopes == init_slope)[0][0]
    init_x0 = all_baseline_startle_sounds[max_slope_idx]
    
    init_baseline_params = [init_saturation, init_slope, init_x0]
    init_startle_scaling = [1] * num_conditions_not_baseline
    init_sound_scaling = [1] * num_conditions_not_baseline
    
    init_params = [*init_baseline_params, *init_startle_scaling, *init_sound_scaling]
    return init_params

def get_model_bounds_PPI(prepulse_conditions):
    '''
    Returns default bounds on the parameters of the model. Note that these bounds
    require the startle-scaling and sound-scaling parameters to be between 0 and 1, so this will
    not effectively model prepulse facilitation.
    
    Parameters
    ---------
    prepulse_conditions : list of tuples (prepulse sound level, delay) defining all of the prepulse conditions, 
                          where the first elment in the list should be the baseline prepulse condition, i.e. 
                          the condition with prepulse sound of 0 dB above baseline
                          
    Returns
    -------
    model_bounds : list of tuples containing upper and lower bounds for each model parameter.
    
    '''
    num_conditions_not_baseline = len(prepulse_conditions) - 1
    min_height, max_height = (None, None)
    min_slope, max_slope = (0, 1)
    min_x0, max_x0 = (0, 70)
    startle_scaling_bounds = (0, 1)
    sound_scaling_bounds = (0, 1)
    model_bounds = [(min_height, max_height), (min_slope, max_slope), (min_x0, max_x0)]
    for i in range(num_conditions_not_baseline):
        model_bounds.append(startle_scaling_bounds) 
    for i in range(num_conditions_not_baseline):
        model_bounds.append(sound_scaling_bounds)
    return model_bounds


def plot_data_and_model(df, prepulse_conditions, startle_sounds_each_condition, model_fit):
    '''
    Plots the average log(startle responses) for each prepulse condition vs. startle sound level, and overlays
    the fitted model curves. Note this is an individual animal's startle data.
    
    Parameters
    ----------
    df : Nx4 Pandas DataFrame with columns 'prepulse_sound', 'delay', 'startle_sound', and 'avg_startle'
         Each of the N rows is a different stimulus and its associated avg(log startle response)
    prepulse_conditions : list of tuples (prepulse sound level, delay) defining all of the prepulse conditions, 
                          where the first elment in the list should be the baseline prepulse condition, i.e. 
                          the condition with prepulse sound of 0 dB above baseline
    startle_sounds_each_condition : dictionary mapping prepulse condition to a list of startle sound levels at which
                                    that condition was measured: 
                                    {(prepulse sound level, delay)->[startle sound level 1, startle sound level 2, ...]}
    model_fit : dictionary containing the model parameters for one rat. This is the return value of model_fit function
                defined above.

    Returns
    ------
    None
    
    Effects
    -------
    Generates a Matplotlib Figure
    '''
    size = plt.figaspect(0.65)
    fig, ax = plt.subplots(figsize=size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.title('PPI model fit for individual rat', fontsize=16)
    plt.xlabel('Startle sound (dB above baseline)', fontsize=16)
    plt.ylabel('Average startle response', fontsize=16)
    saturation = model_fit['baseline_parameters']['saturation']
    slope = model_fit['baseline_parameters']['slope']
    midpoint = model_fit['baseline_parameters']['midpoint']
    colors = plt.cm.rainbow(np.linspace(0, 1, len(prepulse_conditions)))
    for i, condition in enumerate(prepulse_conditions):
        prepulse_sound, delay = condition
        startle_sounds = startle_sounds_each_condition[condition]
        startle_responses_this_condition = []
        for sound in startle_sounds:
            avg_startle = df[(df['startle_sound']==sound) & 
                       (df['prepulse_sound']==prepulse_sound) & 
                       (df['delay']==delay)]['avg_startle'].values[0]
            startle_responses_this_condition.append(avg_startle)
        condition_label = "%s dB, %s ms" % (condition[0], condition[1])
        plt.plot(startle_sounds, startle_responses_this_condition, marker='o', linestyle='None', color=colors[i])
        startle_scaling = model_fit['startle_scaling_parameters'][condition]
        sound_scaling = model_fit['sound_scaling_parameters'][condition]
        xs = np.linspace(0, startle_sounds[-1], 100)
        ys = [sigmoid(x, saturation, slope, midpoint, startle_scaling, sound_scaling) for x in xs]
        plt.plot(xs, ys, color=colors[i], label=condition_label)
#     plt.xticks(range(4), np.arange(0, midpoint*2, 4))
    leg = plt.legend(frameon=False, handlelength=0)
    for i in range(len(prepulse_conditions)):    
        leg.texts[i].set_color(colors[i])
    fig.tight_layout()
    plt.show()