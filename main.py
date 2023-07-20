# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.spatial import ConvexHull
from flask import Flask, jsonify
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim

from utils import read_config
from utils.data_processing import data_processing
from autoformer_model import Autoformer
from train import Train
from predict import Predict

import threading

import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

app = Flask(__name__, template_folder="template")

def filter_features(args, x):
    numb_features = x.shape[1]
    #print("numb_features", numb_features)
    filter_features = []
    for i in range(numb_features):
        
        result = False
        numb_unique_points = len(np.unique(x[:,i]))
        #print("numb_unique_points", numb_unique_points)
 
        if numb_unique_points >= args["min_numb_unique_points"]:
            #tuple of (timestamp,value)
            
            x_tuple = [(j, x[j,i]) for j in range(len(x[:,i]))]
            hull = ConvexHull(x_tuple)
            #print("hull", len(hull.vertices))   
            if len(hull.vertices) >= args["min_numb_convex_hull"]:
                result = True
            
        filter_features.append(result) 
                
    idx = range(numb_features)
    idx_filtered = np.array(idx)[filter_features]
    return(idx_filtered)

def filter_features_small_std(x):
    numb_features = x.shape[1]
    #print("numb_features", numb_features)
    filter_features = []
    for i in range(numb_features):   
        result = False   
        if np.std(x[:,i]) >= 1:
            result = True
            
        filter_features.append(result) 
                
    idx = range(numb_features)
    idx_filtered = np.array(idx)[filter_features]
    return(idx_filtered)

def plot_features(n, timeseries):
    
    max_width = 2 ##images per row
    height, width = n//max_width +1, max_width
    fig, axs = plt.subplots(height, width, sharex=True)
    
    for i in range(n):
        ax = axs.flat[i]
        print("std", np.std(timeseries))           
        ax.plot(timeseries[:,i], "o-", c= "k", markersize=0.5, label=i)
          
     ## access each axes object via axs.flat
    for ax in axs.flat:
        ## check if something was plotted 
        if not bool(ax.has_data()):
            fig.delaxes(ax) ## delete if nothing is plotted in the axes obj

    plt.legend("best")
    plt.show()
    
    
def plot_results(n, results, targets):
    
    max_width = 2 ##images per row
    height, width = n//max_width +1, max_width
    fig, axs = plt.subplots(height, width, sharex=True)
    
    for i in range(n):
        ax = axs.flat[i]
        print("std", np.std(results))           
        ax.plot(results[:,0,i], "o-", c= "k", markersize=0.5, label=str(i) + " predict")
        ax.plot(targets[:,0,i], "o-", c= "r", markersize=0.5, label=str(i) + " target")
     ## access each axes object via axs.flat
    for ax in axs.flat:
        ## check if something was plotted 
        if not bool(ax.has_data()):
            fig.delaxes(ax) ## delete if nothing is plotted in the axes obj

    plt.legend("best")
    plt.show()

def start_trainin():
    args = read_config.get_args()
    data_proc = data_processing()
    

    traing_set, training_loader = data_proc.data_provider(args,"train")
    valid_set, valid_loader = data_proc.data_provider(args,"valid")
    test_set, test_loader = data_proc.data_provider(args,"test")
    
    print("traing_set", traing_set.data_x.shape, traing_set.data_y.shape)
    print("valid_set", valid_set.data_x.shape, valid_set.data_y.shape)
    print("test_set", test_set.data_x.shape, test_set.data_y.shape)

    # create a subplot with rows and columns
    n = traing_set.data_x.shape[1]
    
    #plot_features(n, traing_set.data_x)
    #plot_features(n, valid_set.data_x)
    #plot_features(n, test_set.data_x)
    
    args["enc_in"] = n
    args["dec_in"] = n
    args["c_out"] = n
    
    model = Autoformer(args) 
    
    #model_optim = optim.Adam(model.parameters(), lr=args["learning_rate"])
    model_optim = torch.optim.SGD(model.parameters(), lr=args["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(model_optim,base_lr=3e-4, max_lr=3e-2,
                                                      step_size_up=5,mode="exp_range",gamma=0.85)
    criterion = nn.MSELoss()    

    training = Train(model, model_optim, criterion, scheduler, args, device)
    #training_thread  = threading.Thread(target=training.start_training, name='train_autoformer', daemon = True, \
    #                                     args=((args), training_loader, valid_set, valid_loader, test_set, test_loader)) 
    training.start_training(args,training_loader, valid_set, valid_loader, test_set, test_loader )
    
    path = args["checkpoints_path"]
    best_model_path = path + 'checkpoint.pth'
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
    
    predict = Predict(model, args, device)
    error, results, targets = predict.make_prediction(valid_loader)
    

    plot_results(n, results, targets)
    """
    results = np.reshape(results, (results.shape[0]*results.shape[1], results.shape[2]))
    targets = np.reshape(targets, (targets.shape[0]*targets.shape[1], targets.shape[2]))
    print(results.shape)
    """
    #print("mean test error",predict.make_prediction(test_loader))
    
    
    
    resp = jsonify({})
    resp.status_code = 200
    return resp
    
@app.route('/run', methods=['GET'])    
def run():
    resp = start_trainin()
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=5000,
            debug=True)
    
    path = "logs/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    
    