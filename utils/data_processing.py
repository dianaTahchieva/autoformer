# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
import numpy as np
from scipy.spatial import ConvexHull

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

filter_features = []

class Dataset_Custom(Dataset):
    
    def __init__(self, args, timeenc, flag):
        self.args = args
        self.scaler = StandardScaler()
        self.df_stamp = {}
        self.timeenc = timeenc
        self.flag = flag
        self.load_data()
        
        
        
        
    def load_data(self):
        
        root_path = self.args["data_root_path"]    
        filename = self.args["data_filename"]
        

        df_raw = pd.read_csv(root_path+filename, index_col=False)

        ###########
        if 'Unnamed: 0' in df_raw.keys():
            df_raw = df_raw.drop(['Unnamed: 0'], axis=1)
        
        
        if "timestamp" in df_raw.keys():
            dates = []
            #convert the timestamp to a datetime object in the local timezone
            for timestamp in df_raw["timestamp"]:
                #convert from timestamp in milisec 
                dt_object = datetime.fromtimestamp(int(timestamp)/1000)
                dates.append(str(dt_object))

            df_raw["timestamp"] = dates
            df_raw = df_raw.rename(columns={"timestamp":"date"})
        ##############


        if self.args["features"] == 'M' or self.args["features"] == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            if self.flag == "train":
                if self.args["filter_strategy"] == "convex hull":
                    df_data =  self.filter_features_convex_hull(df_data)
                else:
                    df_data =  self.run_filter_features(df_data)
            else:
                print("filter_features", filter_features)
                df_data = df_data[filter_features]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            if self.flag == "train":
                if self.args["filter_strategy"] == "convex hull":
                    df_data =  self.filter_features_convex_hull(df_data)
                else:
                    df_data =  self.filter_features(df_data)
            else:
                df_data = df_data[filter_features]
                
        filter_features_copy = filter_features.copy()
        filter_features_copy.insert(0, 'date')    
        #print(filter_features_copy)
        df_raw = df_raw[filter_features_copy]   
    
        num_train = round(len(df_data.values)*self.args["training_set_size_percent"])
        num_test = round(len(df_data.values)*self.args["test_set_size_percent"])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.args["seq_len"], len(df_raw) - num_test - self.args["seq_len"]]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[0]
        border2 = border2s[0]
        
        
        if self.args["scaler_on"]:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        if self.args["debug"]:
            self.plot_timeseries(df_data)
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.args["freq"])
            data_stamp = data_stamp.transpose(1, 0)

        if self.flag == "train":
            self.data_x = data[border1s[0]:border2s[0]]
            self.data_y = data[border1s[0]:border2s[0]]
        elif self.flag == "valid":
            self.data_x = data[border1s[1]:border2s[1]]
            self.data_y = data[border1s[1]:border2s[1]]
        elif self.flag == "test":
            self.data_x = data[border1s[2]:border2s[2]]
            self.data_y = data[border1s[2]:border2s[2]]
            
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.args["seq_len"]
        r_begin = s_end - self.args["label_len"]
        r_end = r_begin + self.args["label_len"] + self.args["pred_len"]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.args["seq_len"] - self.args["pred_len"] + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
     
    
    def run_filter_features(self, df_data):
        print("data shape", df_data.shape)
        keys = list(df_data.keys())
        #print("numb_features", numb_features)
        
        for key in keys:
            if np.std(df_data[key]) >= self.args["min_std_value"]:
                filter_features.append(key) 
                        
        #print("filter_features ", filter_features)
        df_data = df_data[filter_features]
        print("data shape after filtering", df_data.shape)
        return(df_data)

    
    def filter_features_convex_hull(self, df_data):
        print("data shape", df_data.shape)
        keys = list(df_data.keys())
        #print("numb_features", numb_features)
        
        for key in keys:
            numb_unique_points = len(np.unique(df_data[key]))
            #print("numb_unique_points", numb_unique_points)
     
            if numb_unique_points >= self.args["min_numb_unique_points"]:
                #tuple of (timestamp,value)
                
                x_tuple = [(j, df_data[key].iloc[j]) for j in range(len(df_data[key]))]
                hull = ConvexHull(x_tuple)
                #print("hull", len(hull.vertices))   
                if len(hull.vertices) >= self.args["min_numb_convex_hull"]:
                    filter_features.append(key) 
            
                        
        #print("filter_features ", filter_features)
        df_data = df_data[filter_features]
        print("data shape after filtering", df_data.shape)
        return(df_data)

    def plot_timeseries(self, df_data):
        data = df_data.values
        print("data shape",data.shape)
        # create a subplot with rows and columns
        rows = int(data.shape[1]/2)
        cols = 2
        fig, axs = plt.subplots(rows, 2)
        count = 0
        keys = df_data.keys()
        for i in range(rows):
            for j in range(cols):
                print("std", np.std(data[:,count]))
                       
                axs[i,j].plot(data[:,count], "o-", c= "k", markersize=0.5, label=keys[count])
                count+=1
        plt.legend()
        plt.show()

class data_processing(object):  
      
    def data_provider(self, args, flag):
        
        timeenc = 0 if args["embed"] != 'timeF' else 1
    
        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = args["batch_size"]
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args["batch_size"]
    
    
        data_set = Dataset_Custom(
            args,
            timeenc=timeenc, 
            flag = flag
        )
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args["num_workers"],
            drop_last=drop_last)
        
        return data_set, data_loader