# -*- coding: utf-8 -*-

import os
import configparser
import numpy as np


config = configparser.RawConfigParser()
config.optionxform = str
if os.path.isfile('config'):
    config.read("config")
else:
    print("config file not found")



args = {}
args["model_name"] = config.get('model', 'model_name')
args["data_root_path"] = config.get('data_loader', 'data_root_path')
args["data_filename"] = config.get('data_loader', 'data_filename')
args["features"] = config.get('data_loader', 'features')
args["target"] = config.get('data_loader', 'target')
args["scaler_on"] = bool(config.get('data_loader', 'scaler_on'))
args["freq"] = config.get('data_loader', 'freq')
args["checkpoints_path"] = config.get('data_loader', 'checkpoints_path')
args["min_numb_unique_points"] = int(config.get('data_loader', 'min_numb_unique_points'))
args["min_numb_convex_hull"] = int(config.get('data_loader', 'min_numb_convex_hull'))
args["filter_strategy"] = config.get('data_loader', 'filter_strategy')
args["min_std_value"] = float(config.get('data_loader', 'min_std_value'))


args["training_set_size_percent"] = float(config.get('forecasting_task', 'training_set_size_percent'))
args["validation_set_size_percent"] = float(config.get('forecasting_task', 'validation_set_size_percent'))
args["test_set_size_percent"] = float(config.get('forecasting_task', 'test_set_size_percent'))
args["seq_len"] = int(config.get('forecasting_task', 'seq_len'))
args["label_len"] = int(config.get('forecasting_task', 'label_len'))
args["pred_len"] = int(config.get('forecasting_task', 'pred_len'))

args["enc_in"] = int(config.get('model_define', 'enc_in'))
args["dec_in"] = int(config.get('model_define', 'dec_in'))
args["c_out"] = int(config.get('model_define', 'c_out'))
args["d_model"] = int(config.get('model_define', 'd_model'))
args["n_heads"] = int(config.get('model_define', 'n_heads'))
args["e_layers"] = int(config.get('model_define', 'e_layers'))
args["d_layers"] = int(config.get('model_define', 'd_layers'))
args["d_ff"] = int(config.get('model_define', 'd_ff'))
args["moving_avg"] = int(config.get('model_define', 'moving_avg'))
args["factor"] = int(config.get('model_define', 'factor'))
args["distil"] = config.get('model_define', 'distil')
args["dropout"] = float(config.get('model_define', 'dropout'))
args["embed"] = config.get('model_define', 'embed')
args["activation"] = config.get('model_define', 'activation')
args["output_attention"] = bool(config.get('model_define', 'output_attention'))
args["do_predict"] = config.get('model_define', 'do_predict')

args["num_workers"] = int(config.get('optimization', 'num_workers'))
args["train_epochs"] = int(config.get('optimization', 'train_epochs'))
args["batch_size"] = int(config.get('optimization', 'batch_size'))
args["patience"] = int(config.get('optimization', 'patience'))
args["learning_rate"] = float(config.get('optimization', 'learning_rate'))
args["des"] = config.get('optimization', 'des')
args["loss"] = config.get('optimization', 'loss')
args["lradj"] = config.get('optimization', 'lradj')
args["use_amp"] = config.get('optimization', 'use_amp')

args["database_name"] = config.get('mogodb', 'database_name')  
args["dbHost"] = config.get('mogodb', 'dbHost') 
args["dbPort"] = config.get('mogodb', 'dbPort')
args["dbUsername"] = config.get('mogodb', 'dbUsername')
args["dbPassword"] = config.get('mogodb', 'dbPassword')

args["debug"] = bool(config.get('debug', 'plot_timeseries'))

def get_args():
    return args
    
    
    



