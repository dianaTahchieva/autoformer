import os
import torch
import torch.nn 
import time

import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate

import warnings

warnings.filterwarnings('ignore')

class Train(object):
    
    def __init__(self,model, model_optim, criterion,scheduler,  args, device):
        self.model = model
        self.model_optim = model_optim
        self.criterion = criterion
        self.scheduler = scheduler
        self.args = args      
        self.device = device
    
    def start_training(self, settings, train_loader, valid_data ,\
               valid_loader, test_data, test_loader  ):
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args["patience"], verbose=True)
        
        path = self.args["checkpoints_path"]
        best_model_path = path + 'checkpoint.pth'
        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device(self.device)))

        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.args["use_amp"]:
            scaler = torch.cuda.amp.GradScaler()
            
        self.model.to(self.device)

        for epoch in range(self.args["train_epochs"]):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
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

                        f_dim = -1 if self.args["features"] == 'MS' else 0
                        outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                        batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                        loss = self.criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args["output_attention"]:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args["features"] == 'MS' else 0
                    outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                    batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                    loss = self.criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.)
                """
                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args["train_epochs"] - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                """
                if self.args["use_amp"]:
                    scaler.scale(loss).backward()
                    scaler.step(self.model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    self.model_optim.step()
                    
                self.scheduler.step()
                
            #print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(valid_data, valid_loader, self.criterion)
            test_loss = self.vali(test_data, test_loader, self.criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            #early_stopping(vali_loss, self.model, path)
            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    break

            adjust_learning_rate(self.model_optim, epoch + 1, self.args)
            torch.save(self.model.state_dict(), best_model_path)
        

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
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

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
       
