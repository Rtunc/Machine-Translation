import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import math
import logging

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=200, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._max_seq_length = max_seq_length
        
        pe = torch.zeros(max_seq_length, d_model)
        
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)

        @torch.jit.script
        def splice_by_size(source, target):
            """Custom function to splice the source by target's second dimension. Required due to torch.Size not a torchTensor. Why? hell if I know."""
            length = target.size(1);
            return source[:, :length]

        self.splice_by_size = splice_by_size
    
    def forward(self, x):
        if(x.shape[1] > self._max_seq_length):
            print(x.shape)
            logging.warn("Input longer than maximum supported length for PE detected. Build a model with a larger input_max_length limit if you want to keep the input; or ignore if you want the input trimmed")
            x = x[:, :self._max_seq_length]
        
        x = x * math.sqrt(self.d_model)
        
        spliced_pe = self.splice_by_size(self.pe, x) # self.pe[:, :x.shape[1]]
#        pe = Variable(spliced_pe, requires_grad=False)
        pe = spliced_pe.requires_grad_(False)
        
#        if x.is_cuda: # remove since it is a sub nn.Module
#            pe.cuda()
#        assert all([xs == ys for xs, ys in zip(x.shape[1:], pe.shape[1:])]), "{} - {}".format(x.shape, pe.shape)

        x = x + pe
        x = self.dropout(x)
        
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    """A two-hidden-linear feedforward layer that can activate and dropout its transition state"""
    def __init__(self, d_model, d_ff=2048, internal_activation=functional.relu, dropout=0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        self.internal_activation = internal_activation
    
    def forward(self, x):
        x = self.dropout(self.internal_activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x
