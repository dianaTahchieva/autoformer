import os
import torch
import torch.nn 
import time

import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate

import warnings

warnings.filterwarnings('ignore')

class Predict(object):
    
    def __init__(self,model,  args, device):
        self.model = model
        self.args = args      
        self.device = device
    
    
    def make_prediction(self, data_loader):
        
        self.model.to(self.device)
        
        total_loss = []
        self.model.eval()
        results = []
        targets = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args["pred_len"]:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args["label_len"], :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.args["output_attention"]:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args["output_attention"]:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args["features"] == 'MS' else 0
                outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                
                print("outputs.shape",outputs.shape)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                #input = batch_x.detach().cpu().numpy()
                #gt = np.concatenate((input[0, :, -1], targets[0, :, -1]), axis=0)
                #pd = np.concatenate((input[0, :, -1], results[0, :, -1]), axis=0)
                #print("gt.shape",gt.shape)
                            
                results.append(pred)
                targets.append(true)
                loss = np.mean(pred - true)
                                
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        
        results = np.concatenate(results, axis=0)
        targets = np.concatenate(targets, axis=0)
        print('test shape:', results.shape, targets.shape)
        results = results.reshape(-1, results.shape[-2], results.shape[-1])
        targets = targets.reshape(-1, targets.shape[-2], targets.shape[-1])
        print('test shape:', results.shape, targets.shape)
        return total_loss, np.array(results), np.array(targets)
        
       
