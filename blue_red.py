#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:34:18 2019

@author: jsoif
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import Markdown

np.random.seed(110) # for reproducible results

def weight_of_colour(colour_likelihood, total_likelihood):
    """ 
        Calcul du poids du point pour chaque couleur
    """
    return colour_likelihood / total_likelihood

def estimate_mean(data, weight):
    """
    Pour chaque point de données, multipliez le point par la probabilité qu'il
    a été tiré de la distribution de la couleur (son "poids").
    
    Diviser par le poids total: essentiellement, on trouve où 
    le poids est centré parmi nos points de données.
    """
    return np.sum(data * weight) / np.sum(weight)

def estimate_std(data, weight, mean):
    """
    Calul de la variance: formule de la loi normale
    Le carré positif pour trouver l’écart type.
    """
    variance = np.sum(weight * (data - mean)**2) / np.sum(weight)
    return np.sqrt(variance)

def plot_guesses(red_mean_guess, blue_mean_guess, red_std_guess, blue_std_guess, alpha=1):
    """
    Dessin des courbes de la distribution bleu et rouge
    alpha : transparence de la courbe
    """
    # set figure size and plot the purple dots
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.plot(both_colours, np.zeros_like(both_colours), '.', color='purple', markersize=10)
       
    # compute the size of the x axis
    lo = np.floor(both_colours.min()) - 1
    hi = np.ceil(both_colours.max()) + 1
    x = np.linspace(lo, hi, 500)
    
    # plot the bell curves
    plt.plot(x, stats.norm(red_mean_guess, red_std_guess).pdf(x), color='r', alpha=alpha)
    plt.plot(x, stats.norm(blue_mean_guess, blue_std_guess).pdf(x), color='b', alpha=alpha)
    
    # vertical dotted lines for the mean of each colour - find the height
    # first (i.e. the probability of the mean of the colour group)
    r_height = stats.norm(red_mean_guess, red_std_guess).pdf(red_mean_guess)
    b_height = stats.norm(blue_mean_guess, blue_std_guess).pdf(blue_mean_guess)
    
    plt.vlines(red_mean_guess, 0, r_height, 'r', '--', alpha=alpha)
    plt.vlines(blue_mean_guess, 0, b_height, 'b', '--', alpha=alpha);


# set parameters
red_mean = 3    # moyenne
red_std = 0.8   # ecart_type

blue_mean = 7   # moyenne
blue_std = 2    # ecart_type

# draw 20 samples from normal distributions with red/blue parameters
red = np.random.normal(red_mean, red_std, size=20)
blue = np.random.normal(blue_mean, blue_std, size=20)
both_colours = np.sort(np.concatenate((red, blue))) # array with every sample point (for later use)


plt.rcParams['figure.figsize'] = (15, 2)
"""
#Known color

lo = np.floor(both_colours.min()) - 1
hi = np.ceil(both_colours.max()) + 1
x = np.linspace(lo, hi, 500)
    
plt.plot(red, np.zeros_like(red), '.', color='red', markersize=10);
plt.plot(blue, np.zeros_like(blue), '.', color='blue', markersize=10);
plt.title(r'Distribution des points rouge et bleu', fontsize=17);
plt.yticks([]);

plt.plot(x, stats.norm(red_mean, red_std).pdf(x), color='r')
plt.plot(x, stats.norm(blue_mean, blue_std).pdf(x), color='b')
    
plt.vlines(red_mean, 0, stats.norm(red_mean, red_std).pdf(red_mean), 'r', '--')
plt.vlines(blue_mean, 0, stats.norm(blue_mean, blue_std).pdf(blue_mean), 'b', '--');
"""

# estimates for the meanand the standard deviation for blue and red
red_mean_guess = 1.1
blue_mean_guess = 9
red_std_guess = 2
blue_std_guess = 1.7

N_ITER = 20 # number of iterations of EM
alphas = np.linspace(0.2, 1, N_ITER) # transparency of curves to plot for each iteration

# plot initial estimates
plot_guesses(red_mean_guess, blue_mean_guess, red_std_guess, blue_std_guess, alpha=0.13)

for i in range(N_ITER):
    
    ## Expectation step
    ## ----------------
    
    #vraisemblance
    likelihood_of_red = stats.norm(red_mean_guess, red_std_guess).pdf(both_colours)
    likelihood_of_blue = stats.norm(blue_mean_guess, blue_std_guess).pdf(both_colours)
    
    likelihood_total = likelihood_of_red + likelihood_of_blue

    red_weight = weight_of_colour(likelihood_of_red, likelihood_total)
    blue_weight = weight_of_colour(likelihood_of_blue, likelihood_total)

    ## Maximisation step
    ## -----------------
    
    # N.B. it should not ultimately matter if compute the new standard deviation guess
    # before or after the new mean guess
    
    red_std_guess = estimate_std(both_colours, red_weight, red_mean_guess)
    blue_std_guess = estimate_std(both_colours, blue_weight, blue_mean_guess)

    red_mean_guess = estimate_mean(both_colours, red_weight)
    blue_mean_guess = estimate_mean(both_colours, blue_weight)

    plot_guesses(red_mean_guess, blue_mean_guess, red_std_guess, blue_std_guess, alpha=alphas[i])
    
plt.title(
    r'Estimations des groupes de distributions après {} itérations d’Expérance Maximisation'.format(
        N_ITER
    ), 
    fontsize=17);



md = """
|            | True Mean      | Estimated Mean | True Std.      | Estimated Std. | 
| :--------- |:--------------:| :------------: |:-------------: |:-------------: |
| Red        | {true_r_m:.5f} | {est_r_m:.5f}  | {true_r_s:.5f} | {est_r_s:.5f}  | 
| Blue       | {true_b_m:.5f} | {est_b_m:.5f}  | {true_b_s:.5f} | {est_b_s:.5f}  |
"""

Markdown(
    md.format(
        true_r_m=np.mean(red),
        true_b_m=np.mean(blue),
        
        est_r_m=red_mean_guess,
        est_b_m=blue_mean_guess,
        
        true_r_s=np.std(red),
        true_b_s=np.std(blue),
        
        est_r_s=red_std_guess,
        est_b_s=blue_std_guess,
    )
)