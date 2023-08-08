# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Parameter

class TA_LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TA_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.W_decomp = Parameter(torch.randn(hidden_size, hidden_size))
        self.b_decomp = Parameter(torch.randn(hidden_size))
        
    def g(self,t):
        T = torch.zeros_like(t)
        T[t.nonzero(as_tuple=True)] = 1 / t[t.nonzero(as_tuple=True)]
        
        Ones = torch.ones([1, self.hidden_size], dtype=torch.float32).to(t.device)

        T = torch.mm(T, Ones)
        return T      

    def forward(self, input, t, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        T = self.g(t)
        
        C_ST = torch.tanh(torch.mm(cx, self.W_decomp) + self.b_decomp)
        C_ST_dis = T * C_ST
        # if T is 0, then the weight is one
        cx = cx - C_ST + C_ST_dis
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return (hy, cy)

class TA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TA_LSTM, self).__init__()
        self.TA_lstm = TA_LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
    def forward(self, X, time):
        c = torch.zeros([self.hidden_size])
        h = torch.zeros([self.hidden_size])
        state = (h, c)
        AllStates = []
        for x,t in zip(X,time):
            state = self.TA_lstm(x, t, state)
            AllStates.append(state[0])
        