import numpy as np
import torch
import pywt
import pandas as pd
import math

def imputeVals(X, imputation='mean'):
    Xs = []
    for i in range(X.shape[-1]):
        df = pd.DataFrame(X[:, :, i].transpose())
        if imputation == 'mean':
            df = df.fillna(df.mean())
        elif imputation == 'forward':
            df = df.ffill()
        elif imputation == 'zero':
            df = df.fillna(0)
        elif imputation == 'backward':
            df = df.bfill()
        Xs.append(df.to_numpy().transpose())
    return np.stack(Xs, -1)

def Regularize(X, times, imputation='mean'):
    size = X[0].shape[0]
    series = []
    for x,t in zip(X, times):
        A = pd.DataFrame(x.T, index=t)
        series.append(A)
    df = pd.concat(series, axis=1)
    AllTimes = torch.tensor(df.index.values).float()
    X = df.to_numpy() # shape: Len, size*feats
    X = X.reshape([X.shape[0], -1, size]) # shape: Len, feats, size
    X = X.transpose([2, 0, 1])
    X = imputeVals(X, imputation=imputation)
    return torch.tensor(X), AllTimes

def getdeltaTimes(times):
    Times = times.clone()
    for i in reversed(range(1, len(times))):
        Times[i] = times[i] - times[i-1]
    return Times
                      
def getRNNFreqGroups_mr(data, times, device = torch.device("cuda:0"), maxlevels=4, waveletType='haar', imputation='mean', fulldata=None, regularize=True, return_times=False):
    WL = pywt.Wavelet(waveletType)
    MLs = []
    Outs = [[] for _ in range(maxlevels + 1)]
    Ts = [[] for _ in range(maxlevels + 1)]
    for d in data:
        ML = pywt.dwt_max_level(d.shape[1], WL)
        MLs.append(ML)
    dL = max(MLs) - maxlevels
    MaxT = max([max(t) for t in times])
    for i, d in enumerate(data):
        out = pywt.wavedec(d, WL, level=MLs[i] - dL, axis=1, mode='periodization')
        for j, o in enumerate(out):
            Outs[j].append(o)
            TSubSamp = math.ceil(times[i].shape[0] / o.shape[1])
            Ts[j].append(times[i][::TSubSamp])
    if fulldata is None:
        Outs.append([d.numpy() for d in data])
        Ts.append(times)
    if regularize:
        Times = []
        Outs_ls = []
        for x,t in zip(Outs, Ts):
            o, time = Regularize(x, t, imputation)
            time /= MaxT
            time = getdeltaTimes(time)
            Outs_ls.append(o)
            Times.append(time)
        Outs = Outs_ls
    else:
        Outs = [[torch.tensor(x) for x in x_arr] for x_arr in Outs]
    if fulldata is not None:
        Outs.append(fulldata)
    if return_times:
        return Outs, Times
    return Outs

def getRNNFreqGroups(data, device = torch.device("cuda:0"), maxlevels=4, waveletType='haar'):
    WL = pywt.Wavelet(waveletType)
    ML = pywt.dwt_max_level(data.shape[1], WL)
    out = pywt.wavedec(data, WL, level=min(maxlevels, ML), axis=1)
    out.append(data)
    out = [torch.tensor(o) for o in out] # , device=device
    return out