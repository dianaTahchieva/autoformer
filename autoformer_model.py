import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    
    """
    Original code from: https://github.com/thuml/Autoformer/blob/main/models/Autoformer.py
    """
    
    def __init__(self, args ): 
        
        super().__init__() 

        self.args = args
        
        
        # Decomp
        kernel_size = self.args["moving_avg"]
        self.decomp = series_decomp(kernel_size)

        print("enc_in: {0}, d_model: {1}, embed: {2}, freq: {3}, dropout: {4} ".format( self.args["enc_in"],\
                                                             self.args["d_model"], self.args["embed"], self.args["freq"], self.args["dropout"]))
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.args["enc_in"], self.args["d_model"], self.args["embed"], self.args["freq"],
                                                  self.args["dropout"])
        self.dec_embedding = DataEmbedding_wo_pos(self.args["dec_in"], self.args["d_model"], self.args["embed"], self.args["freq"],
                                                  self.args["dropout"])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.args["factor"], attention_dropout=self.args["dropout"],
                                        output_attention=self.args["output_attention"]),
                        self.args["d_model"], self.args["n_heads"]),
                    self.args["d_model"],
                    self.args["d_ff"],
                    moving_avg=self.args["moving_avg"],
                    dropout=self.args["dropout"],
                    activation=self.args["activation"]
                ) for l in range(self.args["e_layers"])
            ],
            norm_layer=my_Layernorm(self.args["d_model"])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, self.args["factor"], attention_dropout=self.args["dropout"],
                                        output_attention=False),
                        self.args["d_model"], self.args["n_heads"]),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.args["factor"], attention_dropout=self.args["dropout"],
                                        output_attention=False),
                        self.args["d_model"], self.args["n_heads"]),
                    self.args["d_model"],
                    self.args["c_out"],
                    self.args["d_ff"],
                    moving_avg=self.args["moving_avg"],
                    dropout=self.args["dropout"],
                    activation=self.args["activation"],
                )
                for l in range(self.args["d_layers"])
            ],
            norm_layer=my_Layernorm(self.args["d_model"]),
            projection=nn.Linear(self.args["d_model"], self.args["c_out"], bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.args["pred_len"], 1)
        zeros = torch.zeros([x_dec.shape[0],self.args["pred_len"], x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        #print("x_enc: {0}, x_mark_enc: {1}, x_dec: {2}, x_mark_dec: {3}, mean: {4}, zeros: {5},seasonal_init: {6},\
        # trend_init: {6}".format(x_enc.shape,x_mark_enc.shape, x_dec.shape, x_mark_dec.shape, mean.shape, zeros.shape, seasonal_init.shape ))
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.args["label_len"]:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.args["label_len"]:, :], zeros], dim=1)
        #print("trend_init: {0}, seasonal_init: {1}".format(trend_init, seasonal_init))
        
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        #print("enc_out: {0}, dec_out: {1}, seasonal_part: {2}, trend_part: {3}".format(enc_out, dec_out, seasonal_part, trend_part))
        
        # final
        dec_out = trend_part + seasonal_part

        if self.args["output_attention"]:
            return dec_out[:, -self.args["pred_len"]:, :], attns
        else:
            #print("dec_out:",dec_out[:, -self.pred_len:, :].shape)
            return dec_out[:, -self.args[".pred_len"]:, :]  # [B, L, D]