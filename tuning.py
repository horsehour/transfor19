import os
import sys
import math
import random

import glob

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from itertools import cycle

import re
import time
import pytz
from datetime import datetime

from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import *

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from hyperopt import tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt import fmin as hyfmin

# from sklearn.cluster import KMeans
# from sklearn.neighbors.kde import KernelDensity

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.serif'] = 'Times'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 25

#plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.labelweight'] = 'bold'
    
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markeredgewidth'] = 2

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1

plt.rcParams['savefig.dpi'] = 500


def aligned_observations():
    '''
    Generate timestamp from string time format and align the time stamp w.r.t 00:00:00AM on the same day
    '''
    speed_north = pd.read_csv('Predictions_north.csv')
    speed_south = pd.read_csv('Predictions_south.csv')

    timestr = '20161201 {0}'

    timestamps_north = []
    for i in range(speed_north.shape[0]):
        timeline = speed_north.iloc[i].time
        stamp = datetime.strptime(timestr.format(timeline), '%Y%m%d %H:%M:%S %p').timestamp()
        if timeline.startswith('12') and timeline.endswith('AM'):
            timeline = '0' + timeline[2:]
            stamp = datetime.strptime(timestr.format(timeline), '%Y%m%d %H:%M:%S %p').timestamp()
        elif timeline.endswith('PM'):
            stamp += 12 * 3600

        if i == 0:
            ref = stamp

        timestamps_north.append(1480521605 + stamp - ref)

    timestamps_south = []
    for i in range(speed_south.shape[0]):
        timeline = speed_south.iloc[i].time
        stamp = datetime.strptime(timestr.format(timeline), '%Y%m%d %H:%M:%S').timestamp()
        if i == 0:
            ref = stamp
        timestamps_south.append(1480521604 + stamp - ref)

    timestamps_north = np.array(timestamps_north, dtype='int')
    timestamps_south = np.array(timestamps_south, dtype='int')

    speed_north['timestamp'] = timestamps_north
    speed_south['timestamp'] = timestamps_south
    
    ## reference timestamp
    ref_timestamp = datetime.strptime('20161201 00:00:00+0800',"%Y%m%d %H:%M:%S%z").timestamp()
    ## add timeoffset column for alignment and comparison
    speed_north['timeoffset'] = speed_north['timestamp'] - int(ref_timestamp)
    speed_south['timeoffset'] = speed_south['timestamp'] - int(ref_timestamp)
    return speed_north, speed_south
    

def dataset_split(train_ratio=0.8,direction='N'):
    '''
    Split dataset into training set and testing set
    '''
    bounding = 'north' if direction == 'N' else 'south'
    dataset = pd.read_csv('dataset_{}.csv'.format(bounding))
    selected = dataset[dataset.truth > 0].reset_index(drop=True)

    means = selected.mean()
    stdvar = selected.std()
    zscored = (selected - means)/stdvar

    ninstances, nfeatures = selected.shape[0], dataset.shape[1] - 2
    ntrain = int(train_ratio * ninstances)
    ntest = ninstances - ntrain

    train_indices = np.random.choice(range(ninstances), ntrain)
    test_indices = np.where(~np.isin(np.array(range(ninstances)), train_indices))[0]

    xtrain = zscored.iloc[train_indices,:-2].reset_index(drop=True)
    ytrain = np.array(selected.iloc[train_indices,-2])

    xtest = zscored.iloc[test_indices,:-2].reset_index(drop=True)
    ytest = np.array(selected.iloc[test_indices,-2])
    return dataset, means, stdvar, xtrain, ytrain, xtest, ytest

def training(model,xtrain,ytrain,xtest,ytest):
    '''
    Training model with training and testing data sets
    '''
    model.fit(xtrain, ytrain)

    rmse_train = np.sqrt(mean_squared_error(ytrain, model.predict(xtrain)))
    rmse_test = np.sqrt(mean_squared_error(ytest, model.predict(xtest)))

    print('rmse train/test: {0:.4f}/{1:.4f}'.format(rmse_train,rmse_test))
    return model

def predict_speeds(dataset,means,stdvar,model,direction,regressor='gb'):
    '''
    Prediction performance (mixture of truths in blue and predictions in red)
    '''
    plt.figure(figsize=(20,5))

    selected = dataset[dataset.truth > 0]
    sc1 = plt.scatter(selected.timeoffset,selected.truth,c='b')

    normalized = (dataset - means)/stdvar
    xdata = normalized.iloc[:,:-2]
    ypred = model.predict(xdata)
    sc2 = plt.scatter(dataset.timeoffset,ypred,c='r')
    
    bounding = 'south' if direction == 'S' else 'north'
    plt.title('{} bounding'.format(bounding),fontsize=25)
    plt.legend([sc1,sc2],['truths', 'preds'], fontsize=25)
    plt.xlabel('time offset (s)')
    plt.ylabel('speed')
    
    plt.savefig('{0}_{1}.png'.format(bounding, regressor))
    
    return ypred

def evaluate_model(regressor, df_params, dataset, means, stdvar, xtrain, ytrain, xtest, ytest, sign):
    '''
    Evaluation the selected model on all dataset
    '''
    best = df_params[df_params.rmse == min(df_params.rmse)]

    params = {}
    if regressor == 'gb':
        params['n_estimators'] = best['n_estimators'].values[0]
        params['max_depth'] = best['max_depth'].values[0]
        params['learning_rate'] = best['learning_rate'].values[0]
        params['random_state'] = 0
        params['loss'] = 'ls'
        
        model = GradientBoostingRegressor(**params)
        
    elif regressor == 'rf':
        params['n_estimators'] = best['n_estimators'].values[0]
        params['max_depth'] = best['max_depth'].values[0]
        params['random_state'] = 0
        
        model = RandomForestRegressor(**params)
        
    elif regressor == 'sv':
        params['kernel'] = 'rbf'
        params['gamma'] = best['gamma'].values[0]
        params['C'] = best['C'].values[0]
        params['epsilon'] = best['epsilon'].values[0]

        model = SVR(**params)

    if sign == 'N':
        bounding = 'north'
        speed_truth = speed_north
    else:
        bounding = 'south'
        speed_truth = speed_south

    pred_fname = 'preds_{0}_{1}.csv'.format(bounding,regressor)
    
    model = training(model,xtrain,ytrain,xtest,ytest)
    # make predictions
    speed_truth['preds']= predict_speeds(dataset,means,stdvar,model,direction=sign,regressor='gb')
    
    xdata = pd.concat([xtrain,xtest])
    ydata = np.concatenate([ytrain,ytest])
    model.predict(xdata)
    rmse = np.sqrt(mean_squared_error(ydata, model.predict(xdata)))
    print('rmse all: {0:.4f}'.format(rmse))

    # save prediction to local file
    speed_truth.to_csv(pred_fname, columns=['time','speed','preds'], index=False)
    
def hyp_search(xtrain, ytrain, regressor='gb', numsamples=1000, direction='north'):
    '''
    Hyper optimization
    '''
    fname = 'params_{0}_{1}.csv'.format(regressor, direction)

    params_names = {'gb': ['n_estimators','max_depth','learning_rate'],
                   'rf': ['n_estimators','max_depth'],
                   'sv': ['gamma','C', 'epsilon']}
    
    if not os.path.exists(fname):
        ostream = open(fname, 'a')
        ostream.write(','.join(params_names[regressor]) + ',rmse\n')
        ostream.flush()
        os.fsync(ostream.fileno())
    else:
        ostream = open(fname, 'a')

    space = {}
    if regressor == 'gb':
        space['n_estimators'] = 60 + 20 * hp.randint('n_estimators', 12)
        space['max_depth'] = hp.choice('max_depth', [2, 3, 4])
        space['learning_rate'] = hp.uniform('learning_rate', 0.05,0.2)
    elif regressor == 'rf':
        space['n_estimators'] = 60 + 20 * hp.randint('n_estimators', 12)
        space['max_depth'] = hp.choice('max_depth', [2, 3, 4])
    elif regressor == 'sv':
        space['gamma'] = hp.uniform('gamma',0.1,100)
        space['C'] = hp.uniform('C', 1e1, 1e3)
        space['epsilon'] = hp.uniform('epsilon', 0.1, 0.5)

    num_params = len(space)
    re = ','.join(['{' + str(i) + '}' for i in range(num_params)])

    def query(params):
        if regressor in ['gb','rf']:
            n_estimators, max_depth = params['n_estimators'], params['max_depth']
            params['random_state'] = 0
            if regressor == 'gb':
                params['loss'] = 'ls'
                learning_rate = params['learning_rate']
                model = GradientBoostingRegressor(**params)
                mnm = re.format(n_estimators, max_depth, learning_rate)
            else:
                model = RandomForestRegressor(**params)
                mnm = re.format(n_estimators, max_depth)
        else:
            gamma, C, epsilon = params['gamma'], params['C'], params['epsilon']
            params['kernel'] = 'rbf'
            model = SVR(**params)
            mnm = re.format(gamma,C,epsilon)
        
        ## training and evaluate the prediction error
        # model.fit(xtrain, ytrain)
        # rmse_train = np.sqrt(mean_squared_error(ytrain, model.predict(xtrain)))

        # 10 fold cross validation
        rmse_train = np.sqrt(-cross_val_score(model,xtrain,ytrain,scoring='mean_squared_error',cv=10,n_jobs=5).mean())

        ostream.write(mnm + ',' + str(rmse_train) + '\n')
        ostream.flush()
        os.fsync(ostream.fileno())

        return {'loss': rmse_train, 'status': STATUS_OK}
    
    hyfmin(query, space, algo=tpe.suggest, max_evals=numsamples, trials=Trials())
    ostream.close()

def hyper_search_evaluation(dataset, means, stdvar, xtrain, ytrain, xtest, ytest, sign='N'):
    '''
    Hyper-searching and evaluation altogether
    '''
    direction = 'south' if sign == 'S' else 'north'
    numsamples = 1000
    for regressor in ['gb','rf','sv']:
        # if regressor == 'sv':
        #     numsamples = 1000
        # else:
        #     numsamples = 1000

        hyp_search(xtrain, ytrain, regressor, numsamples, direction=direction)
        parameters = pd.read_csv('params_{0}_{1}.csv'.format(regressor,direction))

       ################### parameters #################################### 
        if regressor == 'gb':
            fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,3),sharey='all')
            ax1.scatter(parameters['n_estimators'],parameters['rmse'],c='purple')
            ax1.set_xlabel('num estimators')
            ax1.set_ylabel('rmse')

            ax2.scatter(parameters['max_depth'],parameters['rmse'],c='purple')
            ax2.set_xlabel('max depth')

            ax3.scatter(parameters['learning_rate'],parameters['rmse'],c='purple')
            ax3.set_xlabel('learning rate')
        elif regressor == 'rf':
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,3),sharey='all')
            ax1.scatter(parameters['n_estimators'],parameters['rmse'],c='purple')
            ax1.set_xlabel('num estimators')
            ax1.set_ylabel('rmse')

            ax2.scatter(parameters['max_depth'],parameters['rmse'],c='purple')
            ax2.set_xlabel('max depth')
        elif regressor == 'sv':
            fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,3),sharey='all')
            ax1.scatter(parameters['gamma'],parameters['rmse'],c='purple')
            ax1.set_xlabel('gamma')
            ax1.set_ylabel('rmse')

            ax2.scatter(parameters['C'],parameters['rmse'],c='purple')
            ax2.set_xlabel('C')

            ax3.scatter(parameters['epsilon'],parameters['rmse'],c='purple')
            ax3.set_xlabel('epsilon')
        ###################################################################
        plt.savefig('param_tuning_{}.png'.format(regressor))
        plt.close(fig)

        df_params = parameters[parameters.rmse == min(parameters.rmse)]
        evaluate_model(regressor, df_params, dataset, means, stdvar, xtrain, ytrain, xtest, ytest, sign)

speed_north, speed_south = aligned_observations()

sign = 'N'
dataset, means, stdvar, xtrain, ytrain, xtest, ytest = dataset_split(train_ratio=0.8,direction=sign)
hyper_search_evaluation(dataset, means, stdvar, xtrain, ytrain, xtest, ytest, sign)

sign = 'S'
dataset, means, stdvar, xtrain, ytrain, xtest, ytest = dataset_split(train_ratio=0.8,direction=sign)
hyper_search_evaluation(dataset, means, stdvar, xtrain, ytrain, xtest, ytest, sign)
