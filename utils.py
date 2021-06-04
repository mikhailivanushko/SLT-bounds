## FUNCTIONS FOR SLT COURSEWORK

# Numpy, Pandas
import numpy as np
import pandas as pd

import random as rd
import copy
import pickle

# Plotting lib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split

# Models and optimizers
from sklearn import svm
from scipy.optimize import differential_evolution, shgo, dual_annealing, basinhopping
from sklearn.dummy import DummyRegressor
import cvxpy as cp

# Saving images
import os
from datetime import datetime

pd.set_option('chained_assignment',None) # Supress annoying pandas warning


def _outlier(params, args_list):
    # TODO doc
    mloss1 = _margin_loss(
        params[1:],     # w
        params[0],      # intercept
        args_list[6],   # margin size
        args_list[0],   # X
        args_list[1])   # Y

    mloss2 = _margin_loss(
        params[1:],     # w
        params[0],      # intercept
        args_list[6],   # margin size
        args_list[2],   # masked X
        args_list[3])   # masked Y

    N = args_list[4] # set size
    M = args_list[5] # mask size

    return - ( (np.sum(mloss1) / N) - (np.sum(mloss2) / M) )


def solve_outlier(X, Y, masked_X, masked_Y, margin_size, bound, kwargs):

    st_bounds = [(-bound,bound)] + [(-1, 1) for x in range(X.shape[1]) ]
    result = dual_annealing(
        func=_outlier,
        bounds=st_bounds,
        args=([X,Y,masked_X,masked_Y,X.shape[0],masked_X.shape[0],margin_size],),
        maxiter=kwargs['maxiter'],
        initial_temp=kwargs['initial_temp'],
        accept=kwargs['accept']
    )

    return result

def _margin_loss_correlation(params, args_list):
    # input: params as [intercept, w_1, w_2, ... , w_n]; args list as [X, Y, rad_vector, margin_size]
    # output: correlation of margin loss function with the rademacher vector (times the number of samples)

    radvec = args_list[2]

    margin_loss = _margin_loss(
        params[1:],     # w
        params[0],      # intercept
        args_list[3],   # margin size
        args_list[0],   # X
        args_list[1])   # Y

    return -np.sum(margin_loss * radvec) # negate correlation to create minimization problem

def _margin_loss(w, intercept, margin_size, X, Y):
    # TODO doc
    if (margin_size == 0):
        margin_loss = 0.5 - ( Y*( np.sign( np.dot(X,w)+intercept ) ) ) / 2
    else:
        signed_distance = ( Y*(np.dot(X,w)+intercept) )
        margin_loss = 1 - ( signed_distance / ( margin_size * np.linalg.norm(w) ) )
        margin_loss[margin_loss<0] = 0
        margin_loss[margin_loss>1] = 1

    return margin_loss


def _confidence(w, intercept, X):
    return (np.dot(X,w)+intercept) / np.linalg.norm(w)

'''
def _confidence_correlation(params, args_list):
    # input: params as [intercept, w_1, w_2, ... , w_n]; args list as [X, rad_vector]
    # output: correlation of confidence with the rademacher vector (times the number of samples)

    radvec = args_list[1]

    confidence = _confidence(
        params[1:],     # w
        params[0],      # intercept
        args_list[0])   # X

    return -np.sum(margin_loss * radvec) # negate correlation to create minimization problem
'''

def _stochastic_solve(function, X, Y, radvec, margin, bounds, kwargs, verbose=False, method='de'):
    # use Stochastic method, TODO speed it up
    if (method == "de"):
        st_bounds = [(-bounds,bounds)] + [(-1, 1) for x in range(X.shape[1]) ]
        result = differential_evolution(
            func=function,
            bounds=st_bounds,
            args=([X,Y,radvec,margin],),
            popsize=kwargs['popsize'],
            disp=verbose
        )
        result.x = np.append(result.x, margin)
        return result.x
    elif (method == "da"):
        st_bounds = [(-bounds,bounds)] + [(-1, 1) for x in range(X.shape[1]) ]
        result = dual_annealing(
            func=function,
            bounds=st_bounds,
            args=([X,Y,radvec,margin],),
            maxiter=kwargs['maxiter'],
            initial_temp=kwargs['initial_temp'],
            accept=kwargs['accept']
        )
        result.x = np.append(result.x, margin)

        return result.x
    else:
        assert(False)
        return

def plot_predicts(
    model, X, Y, radvec=None,
    model_type="custom",

    verbose=False,
    p=1, fix_limits=False, lim_padding=0.5, x_lim=[-1.0, 1.0], y_lim=[-1.0, 1.0],    # Plotting
    dotsize=100, title='', color_bar=False, figsize=[10,10], save_image=False): #
    
    ''' TODO
    Plot function
    model: the model
    features: X
    targets: y
    fix_limits: use custom axis limits?
    lim_padding: auto axis limit padding
    x_lim, y_lim: custom axis limits
    dotsize: dot size on plot
    title: plot title
    figsize: plot size
    verbose: toggle verbose output

    model_type: type of model provided. Can be "svc", "dummy" or "custom"

    save_image: whether to save the plot as an image
    '''

    mask = np.random.choice([True, False], size=X.shape[0], 
                            p=[p, 1.0 - p])

    Y = Y[mask]
    X = X[mask]
    if radvec is not None: radvec = radvec[mask]

    features_0 = X[:, 0]
    features_1 = X[:, 1]
    
    if not fix_limits:
        x_lim = [features_0.min() - lim_padding, features_0.max() + lim_padding]
        y_lim = [features_1.min() - lim_padding, features_1.max() + lim_padding]
    
    
    # plot the hyperplane and margin (if not supplied a dummy model)
    if (model_type != "dummy"):

        # get the separating hyperplane from model
        if (model_type == "svc"):
            w = model.coef_[0]
            b = model.intercept_[0]
            
            a = -w[0] / w[1]
            xx = np.linspace(x_lim[0], x_lim[1])

            # main plane
            yy = a * xx - (b) / w[1]

            if (np.sqrt(np.sum(w[:2] ** 2)) == 0):
                margin = 0
            else:
                margin = 1 / np.sqrt(np.sum(w[:2] ** 2))

            # dashed lines that are 'margin'-away from the main plane
            yy_dashed = []
            yy_dashed.append( yy + np.sqrt(1 + a ** 2) * margin )
            yy_dashed.append( yy - np.sqrt(1 + a ** 2) * margin )
            
        elif(model_type=="margin_loss" or model_type=="confidence"):
            b = model[0]

            w_raw = model[1:] if (model_type=="confidence") else model[1:-1]
            margin = 1.0 if (model_type=="confidence") else model[-1]

            w = (w_raw/np.linalg.norm(w_raw))

            xx = np.linspace(x_lim[0], x_lim[1])
            if (margin >= 0):
                # main plane
                yy = (xx - b * w[0]) * (-w[0]/w[1]) + b * w[1]

                # dashed lines that are 'margin'-away from the main plane
                yy_dashed = []
                yy_dashed.append( (xx - (b + margin) * w[0]) * (-w[0]/w[1]) + (b + margin) * w[1] )
                yy_dashed.append( (xx - (b - margin) * w[0]) * (-w[0]/w[1]) + (b - margin) * w[1] )
            else:
                yy_dotted = []
                for l in range(-3,4):
                    yy_dotted.append( (xx - (b + margin * l * 0.2) * w[0]) * (-w[0]/w[1]) + (b + margin * l * 0.2) * w[1] )
        elif(model_type=="fixed_svm"):
            b = model[0]

            w_raw = model[1:-1]
            margin = model[-1]

            w = (w_raw/np.linalg.norm(w_raw))

            xx = np.linspace(x_lim[0], x_lim[1])
            if (margin >= 0):
                # main plane
                yy = (xx - b * w[0]) * (-w[0]/w[1]) + b * w[1]

                # dashed lines that are 'margin'-away from the main plane
                yy_dashed = []
                yy_dashed.append( (xx - (b + margin) * w[0]) * (-w[0]/w[1]) + (b + margin) * w[1] )
                yy_dashed.append( (xx - (b - margin) * w[0]) * (-w[0]/w[1]) + (b - margin) * w[1] )
            else:
                yy_dotted = []
                for l in range(-3,4):
                    yy_dotted.append( (xx - (b + margin * l * 0.2) * w[0]) * (-w[0]/w[1]) + (b + margin * l * 0.2) * w[1] )
    

    # calculated margin lines, w, b (intercept)

    if (model_type=="svc"):
        values = (Y[:] == 1).astype(np.float)
    elif(model_type=="margin_loss"):
        #values = (radvec[:] == 1).astype(np.float)
        values = _margin_loss(w_raw, b, margin, X, Y) * radvec
        print(np.min(_margin_loss(w_raw, b, margin, X, Y)))
    elif(model_type=="confidence"):
        values = (Y[:] == 1).astype(np.float)
    elif(model_type=="fixed_svm"):
        values = (Y[:] == 1).astype(np.float)

    plt.figure(figsize=figsize)
    
    XX, YY = np.mgrid[x_lim[0]:x_lim[1]:500j, y_lim[0]:y_lim[1]:500j]
    ravel = np.c_[XX.ravel(), YY.ravel()]
    if (model_type=="svc"):
        Z = model.predict(ravel)
    elif (model_type=="margin_loss"):
        # Z = np.array(_margin_loss( w[:2], b, margin, ravel, np.ones(ravel.shape[0])) ) #Y))
        if (verbose):
            correlations = _margin_loss(w_raw, b, margin, X, Y) * radvec
            print('correlation with the rademacher vector:', np.sum(correlations), np.sum(correlations) / len(correlations))
    elif (model_type=="confidence"):
        Z = np.array( _confidence( w[:2], b, ravel) )
        if (verbose):
            correlations = _confidence( w_raw, b, X ) * radvec
            print('correlation with the radvec:', np.sum(correlations), np.sum(correlations) / len(correlations))
    elif (model_type=="fixed_svm"):
        Z = np.sign(ravel.dot(w[:2]) - b)

    cmbg = LinearSegmentedColormap.from_list(
        'color_map_bg', [(16/255, 90/255, 173/255), (193/255, 110/255, 49/255)], N=100)
    cmdot = LinearSegmentedColormap.from_list(
        'color_map_dot', [(11/255, 165/255, 254/255), (255/255, 150/255, 0/255)], N=100)
    
    # Put the result into a color plot
    if (model_type=="svc" or model_type=="confidence" or model_type=="fixed_svm"):
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z, antialiased=True, cmap=cmbg)
        if (color_bar):
            plt.colorbar()
    
    # Draw the plane and margin (if not supplied a dummy model)
    plt.plot(xx, yy, 'k-',c='black')
    plt.plot(xx, yy_dashed[0], 'k--',c='black')
    plt.plot(xx, yy_dashed[1], 'k--',c='black')
        
    

    # Highlight Support vectors for svc model
    #if (model_type == "svc"):
    #    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=dotsize,
    #                facecolors='none', edgecolors='white', linewidths=2,cmap=cmdot)
    
    # Draw the rest of the features
    plt.scatter(features_0, features_1, c=values, zorder=10,
               edgecolors='white',s=dotsize, linewidths=1.5, cmap=cmdot)
    
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    #plt.xlabel(str(features.columns[0]))
    #plt.ylabel(str(features.columns[1]))
    plt.title(title)
    plt.grid()
    

    if not os.path.exists("images"):
        os.mkdir("images")

    if (save_image):
        now = datetime.now()
        date_time = now.strftime("%m%d%Y%H%M%S")
        plt.savefig("images/" + str(model_type) + "_" + str(margin) + "_" + date_time + ".png")

    plt.show()

def process_dataset(
    X, Y,
    model_type="svc",
    method="de", kwargs=None, radvec=None, # used if model_type = 'margin_loss'
    margin=1.0, bounds=1e-10,
    figsize=[7,7], lim_padding=1, dotsize=200, color_bar=False, # plot options
    verbose=False, save_image=False):
    
    '''
    TODO FIX
    Process the dataset with a certain model and plot the result

    Input passed to "prep_data":
    data: data to be processed
    class_col: name of class column
    classes: a pair of lists, containing the values of class_col belonging to classes A and B

    Input passed to plot function:
    lim_padding: auto axis limit padding
    dotsize: dot size on plot
    figsize: plot size
    verbose: toggle verbose output

    model_type: model type to use. Can be "svc" or "custom"
    
    margins: a list of margins to try. Used by SVC as the value of C; Used by the custom optimizer as the actual margin size
    (if supplied a negative margin, then no margin is used at all (no clamping on confidence)).

    save_image: whether to save the plots as images
    '''
    data_size = X.shape[0]

    if (model_type == "svc"):
        model = svm.SVC(kernel='linear', C=margin, probability=False)
        model.fit(X, Y)

    elif (model_type == "margin_loss"):
        if (margin >= 0):
            # use Stochastic method
            model = _stochastic_solve(
                function=_margin_loss_correlation,
                X=X,
                Y=Y,
                radvec=radvec,
                margin=margin,
                bounds=bounds,
                method=method,
                kwargs=kwargs,
                verbose=verbose
            )
        else:
            assert(False)

    elif (model_type == "confidence"):
        # Linear problem, use analytical solution
        X_signed = np.transpose(np.transpose(X)*radvec)
        solution = np.mean(X_signed, axis=0)
        if (np.linalg.norm(solution) == 0):
            model = [0] + [0 for x in range(X.shape[1])]
        else:
            check = np.sign(np.sum(radvec*Y))
            model = np.array([bounds * check] + list(solution))

    if (verbose): print('model:',model)

    # set plot title in accordance with model and margin
    if (model_type == 'svc'):
        plot_title = 'C = ' + str(margin)
    elif (model_type == 'margin_loss'):
            plot_title = 'margin size = ' + str(margin)
    elif (model_type == 'confidence'):
            plot_title = 'linear confidence (dashed line shows confidence 1)'

    plot_predicts(
        model, X, Y, radvec,
        fix_limits=False, lim_padding=lim_padding, dotsize=dotsize, title=plot_title, figsize=figsize,
        color_bar=color_bar, model_type=model_type, verbose=verbose, save_image=save_image
    )

    

class RDhistory():
    '''
    a class which records the calculation of rademacher complexity
    model_type: type of model being tested
    rademacher: list of rademacher vectors
    hypothesis: list of calculated hypothesis vectors
    complexity: history of complexity. list of pairs [sample_count, correlation]
    correlation: history of correlations. list of lists of calculated correlations
    model: list of models, relating to the hypotheses
    '''
    def __init__(self, model_type):
        self.model_type = model_type
        self.rademacher = list()
        self.hypothesis = list()
        self.complexity = list()
        self.correlation = list()
        self.model = list()

def pump_rademacher(
    X, Y, 
    history,
    radvec=None,
    method="de",
    kwargs=None,
    runs_per_sample=1,
    margin=0,
    bounds=0,
    verbose=True):
    
    # This function takes an instance of RDhistory class and generates additional samples for it
    #
    # samples: number of random vectors to be generated
    # runs_per_sample: number of runs to execute the solver for a single random vector. The best-corelating hypothesis will be appended
    #   to history.hypothesis, and a list of corelations for all the runs will be appended to history.correlation
    # tol: tolerance for the solver
    # flip: an option to flip the randomly generated vector on every other attempt at solving for it.
    # margin, bound: options for the plane
    # popsize: size of population for the stochastic solver
    # verbose: enable verbose output

    '''
    if (len(history.rademacher) > len(history.hypothesis)):
        print('size mismatch, fixing...\n', 'rd:', len(history.rademacher), 'hp:', len(history.hypothesis))
        history.rademacher = history.rademacher[:len(history.rademacher)]
    elif (len(history.hypothesis) > len(history.rademacher)):
        print('size mismatch, fixing...\n', 'rd:', len(history.rademacher), 'hp:', len(history.hypothesis))
        history.hypothesis = history.hypothesis[:len(history.rademacher)]
    '''

    
    data_size = X.shape[0]
        
    if radvec is None:
        radvec = [rd.randint(0, 1) * 2 - 1 for x in range(data_size)]

    sample_hypothesis = []
    correlations = []
    accuracies = []
    models = []

    for j in range(runs_per_sample):

        if (history.model_type=='svc'):
            check = np.sum(radvec)
            if (check == data_size and history.model_type=='svc'):
                model = DummyRegressor(strategy="constant",constant=1)
                model.fit(X, Y)
                pred = model.predict(X)
            elif (history.model_type=='svc' and check == -data_size):
                model = DummyRegressor(strategy="constant",constant=-1)
                model.fit(X, Y)
                pred = model.predict(X)
            else:
                model = svm.SVC(kernel='linear', C=C_exp, probability=False, tol=tol, verbose=verbose)
                model.fit(X, Y)
                pred = model.predict(X)
        elif (history.model_type=='margin_loss'):
            # use Stochastic method
            model = _stochastic_solve(
                function=_margin_loss_correlation,
                X=X,
                Y=Y,
                radvec=radvec,
                margin=margin,
                bounds=bounds,
                method=method,
                kwargs=kwargs,
                verbose=verbose
            )
            pred = _margin_loss(model[1:-1], model[0], model[-1], X, Y)
            models.append(model)

        elif (history.model_type=="confidence"):
            # Linear problem, use analytical solution
            X_signed = np.transpose(np.transpose(X)*radvec)
            solution = np.mean(X_signed, axis=0)
            if (np.linalg.norm(solution) == 0):
                model = [0] + [0 for x in range(X.shape[1])]
            else:
                check = np.sign(np.sum(radvec))
                model = np.array([bounds * check] + list(solution))

            pred = _confidence(model[1:], model[0], X)
            models.append(model)

        # got prediction
        sample_hypothesis.append(pred)
        correlations.append(np.sum(radvec * pred))

    best_run_index = correlations.index(max(correlations))
    if (verbose):
        print('best:', correlations[best_run_index], 'all:', correlations)
    
    best_pred = sample_hypothesis[best_run_index]
    
    history.rademacher.append(radvec)
    history.hypothesis.append(best_pred)
    history.correlation.append(correlations)
    history.model.append(models[best_run_index])
    # print('b: ', bounds, '\tm: ', margin, '\tc: ', correlations[best_run_index])

def calc_complexity(history, dist=False):

    # Calculate rademacher complexity given an instance of RDhistory class
    # return the complexity, and optionally also return the distribution of corelations with the random vectors.

    samples = len(history.hypothesis)
    data_size = len(history.hypothesis[0])
    
    if (len(history.rademacher) != samples):
        history.rademacher = history.rademacher[:samples]
    
    assert(len(history.rademacher) == len(history.hypothesis))
    
    complexity = []

    for s in range(samples):
        complexity.append(np.sum(history.rademacher[s] * history.hypothesis[s]))

    hypothesis_complexity = sum(complexity) * (1/samples) * (1/data_size)
    
    if dist:
        return hypothesis_complexity, np.array(complexity)
    else:
        return hypothesis_complexity
           
def plot_correlations(history, range_in=[0,1], bins=100):

    # Plot a histogram of corelations given an instance of RDhistory class

    samples = len(history.hypothesis)
    data_size = len(history.hypothesis[0])

    if (len(history.rademacher) != samples):
        history.rademacher = history.rademacher[:samples]

    print(len(history.rademacher), len(history.hypothesis))

    assert(len(history.rademacher) == len(history.hypothesis))

    hypothesis_complexity, dist_new = calc_complexity(history, dist=True)
    print('rademacher complexity:', hypothesis_complexity)
    plt.figure(figsize=[20,5])
    plt.hist( dist_new / data_size, bins=bins, range=range_in, label="new seeds data", alpha=1, density=True)
    plt.legend(loc='upper right')
    plt.show()
    
def best_worst(data, history, figsize=[7,7], color_bar=False):
    
    '''
    Display the best and worst correlation from a given RDhistory class

    data: data that the samples were based on
    history: rademacher calculation history

    figsize: size of plot
    '''

    print("Best and worst correlations")

    samples = len(history.hypothesis)
    data_size = len(history.hypothesis[0])

    if (len(history.rademacher) != samples):
        history.rademacher = history.rademacher[:samples]

    assert(len(history.rademacher) == len(history.hypothesis))

    correlations = []
    for s in range(samples):
        correlations.append(np.sum(history.rademacher[s] * history.hypothesis[s]))        

    complexity_df = pd.DataFrame(correlations)
    max_correlation = complexity_df.idxmax()
    min_correlation = complexity_df.idxmin()

    rd_max = np.array(history.rademacher[int(max_correlation)])
    rd_min = np.array(history.rademacher[int(min_correlation)])

    print('best correlation', correlations[max_correlation[0]], 'occured in sample #', max_correlation[0])
    plot_predicts(history.model[int(max_correlation)], data, rd_max, model_type=history.model_type, fix_limits=False, lim_padding=0.2, dotsize=150, figsize=[10,10], color_bar=color_bar)
    
    print('worst correlation',  correlations[min_correlation[0]], 'occured in sample #', min_correlation[0])
    plot_predicts(history.model[int(min_correlation)], data, rd_min, model_type=history.model_type, fix_limits=False, lim_padding=0.2, dotsize=150, figsize=[10,10], color_bar=color_bar)

def save_file(name,history):
    with open(name, "wb") as fp:
        pickle.dump(history, fp)

def load_file(name):
    with open(name, "rb") as fp:
        hist = pickle.load(fp)
    return hist
