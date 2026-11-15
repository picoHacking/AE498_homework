"""This module contains functions for design of experiments analysis for HW 4 in AE 498 Data-Driven Modeling and Optimization."""
import numpy as np
import matplotlib.pyplot as plt
import scipy
def effects(responses, n_replicates):
    """Function to calculate main and interaction effects for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.

    Returns
    -------
    effects : array_like
        Calculated effects (3 x 1 dimensions) for a, b, and ab.

    Notes
    -----
    Standard order of experiment conditions (order of rows in responses) = (1), a, b, ab
    (1) = both low
    a = only a high
    b = only b high
    ab = both high
    """

    # effect of a
    A = ((np.mean(responses[1])-np.mean(responses[0])) + (np.mean(responses[3])-np.mean(responses[2])))/2
    
    # effect of b
    B = ((np.mean(responses[2])-np.mean(responses[0])) + (np.mean(responses[3])-np.mean(responses[1])))/2
    
    # effect of a on b
    AB = ((np.mean(responses[3])-np.mean(responses[2])) - (np.mean(responses[1])-np.mean(responses[0])))/2
    
    return np.array([A, B, AB])

def plot_effects(responses, n_replicates, datafile):
    print(type(datafile))
    """Function to plot effects for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.
    datafile: str
        Filename of dataset to plot
        
    Returns
    -------
    out : list
        List of artists added.

    Notes
    -----
    This function structure is based on matplotlib's suggested signature function for returning a plot:
    https://matplotlib.org/3.5.0/tutorials/introductory/usage.html#the-object-oriented-interface-and-the-pyplot-interface
    """
    # plot R vs A +, - for B +, -
    fig, ax = plt.subplots()
    B_minus = [np.mean(responses[0]), np.mean(responses[1])]
    B_plus = [np.mean(responses[2]), np.mean(responses[3])]
    A = [0,1]
    ax.set_title(datafile.split('.')[0])
    out1 = ax.plot(A, B_plus, color="#F56127",marker='o')
    out2 = ax.plot(A, B_minus, color="#2765F5", marker='o')
    ax.text(np.median(A)-0.1,np.median(B_plus)+0.5, 'B=high')
    ax.text(np.median(A)-0.1,np.median(B_minus)+0.5, 'B=low')
    ax.set_xlabel('A')
    ax.set_ylabel('Mean Response')
    ax.grid()
    ax.set_xticks(A)
    ax.set_xticklabels(['low','high'])
    return out1, out2

def plot_tornado(responses, n_replicates, datafile):
    
    """Function to make a tornado plot for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.
    datafile: str
        Filename of dataset to plot
        
    Returns
    -------
    out : list
        List of artists added.

    Notes
    -----
    This function structure is based on matplotlib's suggested signature function for returning a plot:
    https://matplotlib.org/3.5.0/tutorials/introductory/usage.html#the-object-oriented-interface-and-the-pyplot-interface
    """
    colors=["#AF1042","#1F7EB1"]
    labels = np.char.array(['A','B','AB'])
    eff = effects(responses, n_replicates)
    baseline = 0.0
    eff = np.array(eff)
    bars = eff - baseline
    #print(bars)
    fig, ax = plt.subplots()
    sort_i = np.argsort(bars)
    bars = bars[sort_i]
    labels = labels[sort_i]
    out = ax.barh(labels, bars, height=0.5,)
    ax.set_xlabel('Main Effect')
    ax.set_title(datafile.split('.')[0])
    plt.plot([0,0],[-0.5,2.5], color='black')
    i = 0
    for bar in out:
        if bars[i] <0:
            plt.text(bars[i]+0.1,i-0.01,f'{np.round(bars[i],2)}',color='white')
            bar.set_color(colors[0])
        else:
            plt.text(bars[i]-1,i-0.01,f'{np.round(bars[i],2)}',color='white')
            bar.set_color(colors[1])
        i+=1
    return out
print(scipy.__version__)