import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from Models.TorchModels import CVE, Attention, Transformer

import wandb

class switch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        return mask * x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class gradswitch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.mask = mask
        return x
    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.mask
        return grad_output * mask, None

class MaskedGradLinear(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True, masks=[0., 1., 1., 0., 0., 0., 0., 0.]):
        super(MaskedGradLinear, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        self.masks = masks
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        outmasked = []
        for i in range(len(out)):
            outmasked.append(out[i])
        cont_emb = torch.cat(outmasked, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](gradswitch.apply(out[i], 0.0))
            Outs.append(o.squeeze(-1))
        return op, Outs
    
class MaskedFusionSwitch(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True, masks=[1., 1., 1., 1., 1., 1., 1., 1.]):
        super(MaskedFusionSwitch, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        self.masks = masks
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        outmasked = []
        for i in range(len(out)):
            outmasked.append(switch.apply(out[i], self.masks[i]))
        cont_emb = torch.cat(outmasked, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        return op, Outs
    
class MaskedFusionGradSwitch(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True, masks=[1., 1., 1., 1., 1., 1., 1., 1.], GradMasks=[1., 1., 1., 1., 1., 1., 1., 1.]):
        super(MaskedFusionGradSwitch, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        self.masks = masks
        self.GradMasks = GradMasks
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        outmasked = []
        for i in range(len(out)):
            outmasked.append(switch.apply(out[i], self.masks[i]))
        cont_emb = torch.cat(outmasked, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](gradswitch.apply(out[i], self.GradMasks[i]))
            Outs.append(o.squeeze(-1))
        return op, Outs
    
class MaskedFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True, masks=[0., 1., 1., 0., 0., 0., 0., 0.]):
        super(MaskedFusion, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        self.masks = masks
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        outmasked = []
        for i in range(len(out)):
            outmasked.append(self.masks[i] * out[i])
        cont_emb = torch.cat(outmasked, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        return op, Outs

class SigWeightedFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(SigWeightedFusion, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.weights = nn.Parameter(torch.zeros(len(input_size_all)))
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        outmasked = []
        weights = self.sig(self.weights * 10.)
        for i in range(len(out)):
            outmasked.append(weights[i] * out[i])
        cont_emb = torch.cat(outmasked, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        self.wandblog(weights)
        return op, Outs
    def wandblog(self, weights):
        weightdict = {}
        for i, w in enumerate(weights):
            weightdict['weights_Model' + str(i)] = w
        wandb.log(weightdict, commit=False)

class GumbelWeightedFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(GumbelWeightedFusion, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        self.tau = 100.
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.weights = nn.Parameter(torch.ones(len(input_size_all)) / len(input_size_all))
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        outmasked = []
        weights = F.gumbel_softmax(self.weights, tau=self.tau, hard=False)
        for i in range(len(out)):
            outmasked.append(weights[i] * out[i])
        cont_emb = torch.cat(outmasked, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        self.wandblog(weights)
        return op, Outs
    def wandblog(self, weights):
        weightdict = {}
        for i, w in enumerate(weights):
            weightdict['weights_Model' + str(i)] = w
        weightdict['Tau'] = self.tau
        wandb.log(weightdict, commit=False)
    
class AttentionFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(AttentionFusion, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.attn =  Attention(hidden_size, multiplier*max(input_size_all))
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier*max(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier*max(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        Ms = []
        for i in range(len(input_size_all)):
            ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
            Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        cont_emb = torch.stack(out, -2)
        masks = torch.ones([cont_emb.shape[0], cont_emb.shape[1]], device=cont_emb.device)
        attn_weights = self.attn(cont_emb, masks)
        op = torch.sum(cont_emb * attn_weights, dim=-2)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        return op, Outs

class AvgFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(AvgFusion, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier*max(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier*max(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        Ms = []
        for i in range(len(input_size_all)):
            ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
            Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        cont_emb = torch.stack(out, -2)
        cont_emb = torch.mean(cont_emb, dim=-2)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        return op, Outs

class WeightedAvgFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(WeightedAvgFusion, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier*max(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier*max(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        self.weights = nn.Parameter(torch.ones([len(input_size_all)]) / len(input_size_all))
        Ms = []
        for i in range(len(input_size_all)):
            ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
            Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        weights = self.weights / torch.sum(self.weights)
        cont_emb = torch.stack(out, -1)
        cont_emb = torch.matmul(cont_emb, weights)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        self.wandblog(weights)
        return op, Outs
    def wandblog(self, weights):
        weightdict = {}
        for i, w in enumerate(weights):
            weightdict['weights_Model' + str(i)] = w
        wandb.log(weightdict, commit=False)

class WeightedAvgEnsemble(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(WeightedAvgEnsemble, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.weights = nn.Parameter(torch.ones([len(input_size_all)]) / len(input_size_all))
        Ms = []
        for i in range(len(input_size_all)):
            ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
            Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        weights = self.weights / torch.sum(self.weights)
        op = 0.0
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
            op += weights[i] * o.squeeze(-1)
        self.wandblog(weights)
        return op, Outs
    def wandblog(self, weights):
        weightdict = {}
        for i, w in enumerate(weights):
            weightdict['weights_Model' + str(i)] = w
        wandb.log(weightdict, commit=False)

class LinearFusionSameFC(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(LinearFusionSameFC, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier*sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier*sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        Ms = []
        self.ModelLinear = nn.Linear(multiplier * max(input_size_all), out_size)
#         self.finalAct = nn.Sigmoid()
#         if out_size > 1:
#             self.finalAct = nn.Softmax()
    def forward(self, out):
        cont_emb = torch.cat(out, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
#         op = self.finalAct(op)
        Outs = []
        for i in range(len(out)):
            o = self.ModelLinear(out[i])
#             o = self.finalAct(o)
            Outs.append(o.squeeze(-1))
        return op, Outs

class LinearFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(LinearFusion, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        print('input_size_all ', sum(input_size_all))
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier*sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier*sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], out_size)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
#         self.finalAct = nn.Sigmoid()
#         if out_size > 1:
#             self.finalAct = nn.Softmax()
    def forward(self, out):
        cont_emb = torch.cat(out, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
#         op = self.finalAct(op)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
#             o = self.finalAct(o)
            Outs.append(o.squeeze(-1))
        return op, Outs

class HieLinFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(HieLinFusion, self).__init__()
        Ms = []
        Mouts = []
        multiplier = 2 if bidirectional else 1
        for i in range(1, len(input_size_all)):
            if i < len(input_size_all) - 1:
                model = nn.Sequential(nn.Linear(multiplier * (input_size_all[i-1] + input_size_all[i]), multiplier * input_size_all[i]),
                                      nn.ReLU())
            else:
                model = nn.Sequential(nn.Linear(2 * (input_size_all[i-1] + input_size_all[i]), out_size),
                                      nn.ReLU())
            Ms.append(model)
            linearout = nn.Linear(multiplier * input_size_all[i-1], 1)
            Mouts.append(linearout)
        self.linears = nn.ModuleList(Ms)
        self.CompOuts = nn.ModuleList(Mouts)
    def forward(self, out):
        Outs = []
        o = out[0]
        for i in range(len(self.linears)):
            op = self.CompOuts[i](o)
            Outs.append(op.squeeze(-1))
            o = self.linears[i](torch.cat((o, out[i+1]), -1))
        return o, Outs

class TransformerFusion(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True, dropout=0.0):
        super(TransformerFusion, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.varEmebedding = nn.Embedding(len(input_size_all), multiplier * input_size_all[0]) # Assuming all models have same size
        d, N, hes = multiplier*input_size_all[0], 1, 1
        self.transformer = Transformer(d, N, hes, dk=None, dv=None, dff=None, dropout=dropout)
        self.attn = Attention(2*d, d)
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier*sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier*sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        Ms = []
        for i in range(len(input_size_all)):
            ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
            Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        comb_emb = torch.stack(out, -2)
        masks = torch.ones([comb_emb.shape[0], comb_emb.shape[1]], device=comb_emb.device)
        varis = torch.arange(len(out), device=comb_emb.device)
        varis_emb = self.varEmebedding(varis)
        comb_emb = varis_emb + comb_emb
        cont_emb = self.transformer(comb_emb, mask=masks)
        attn_weights = self.attn(cont_emb, mask=masks)
        op = torch.sum(cont_emb * attn_weights, dim=-2)
        if self.useExtralin:
            op = self.relu(self.foreLinear(op))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(op).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        return op, Outs

class trainableswitch(nn.Module):
    def __init__(self):
        super(trainableswitch, self).__init__()
        self.W = nn.Parameter(torch.tensor(0.6))
        self.activation = nn.ReLU()
        self.Mask = 0.6
    def forward(self, x):
        self.Mask = self.activation(self.W - 0.5)
        return x * self.Mask
class TrainableFusionSwitch(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(TrainableFusionSwitch, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.useExtralin = useExtralin
        if not useExtralin:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), out_size)
        else:
            self.foreLinear = nn.Linear(multiplier * sum(input_size_all), hidden_size)
            self.LastLinear = nn.Linear(hidden_size, out_size)
            self.relu = nn.ReLU()
        switches = []
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
                Ms.append(ModelLinear)
#                 switches.append(trainableswitch())
        self.CompOuts = nn.ModuleList(Ms)
        self.switchweights = nn.Parameter(torch.tensor([0.1 for _ in range(len(input_size_all))]))
#         self.switches = nn.ModuleList(switches)
        self.activation = nn.ReLU()
    def forward(self, out):
        self.switchMasks = self.activation(self.switchweights)
#         self.switchMasks = self.switchweights
#         switchMasks = self.switchMasks / self.switchMasks.sum()
        outmasked = []
        for i in range(len(out)):
            outmasked.append(self.switchMasks[i] * out[i])
        cont_emb = torch.cat(outmasked, -1)
        if self.useExtralin:
            op = self.relu(self.foreLinear(cont_emb))
            op = self.LastLinear(op).squeeze(-1)
        else:
            op = self.foreLinear(cont_emb).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        self.wandblog()
        return op, Outs
    def l1_norm(self):
        L1_norm = torch.norm(self.switchMasks, 1)
#         L1_norm = 0.0
#         for i, sw in enumerate(self.switchMasks):
#             L1_norm += torch.norm(sw.Mask, 1)
        return L1_norm
    def wandblog(self):
        weightdict = {}
        for i, sw in enumerate(self.switchMasks):
            weightdict['weights_Model' + str(i)] = self.switchweights[i].cpu().detach()
            weightdict['Mask_Model' + str(i)] = sw.cpu().detach()
        wandb.log(weightdict, commit=False)

class BranchLayer(nn.Module):
    def __init__(self, num_ins):
        super(BranchLayer, self).__init__()
        self.prob = nn.Parameter(torch.ones(num_ins))
    def forward(self, outs, temp):
        self.weights = F.gumbel_softmax(self.prob, tau=temp, hard=False)
        outcomb = 0.0
        for i, o in enumerate(outs):
            outcomb += o * self.weights[i]
        return outcomb
class SimpleWaveBranch(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size, useExtralin=False, bidirectional=True):
        super(SimpleWaveBranch, self).__init__()
        multiplier = 2 if bidirectional else 1
        self.WaveLinear = nn.Linear(multiplier*sum(input_size_all[:-1]), hidden_size)
        self.SimpleModelLinear = nn.Linear(multiplier*input_size_all[-1], hidden_size)
        self.Branching = BranchLayer(2)
        self.useExtralin = useExtralin
        self.relu = nn.ReLU()
        self.LastLinear = nn.Linear(hidden_size, out_size)
        self.temp = 10
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(multiplier * input_size_all[i], 1)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)
    def forward(self, out):
        wave_out = torch.cat(out[:-1], -1)
        wave_out = self.WaveLinear(wave_out)
        Simp_out = self.SimpleModelLinear(out[-1])
        op = self.Branching([wave_out, Simp_out], self.temp)
        op = self.LastLinear(op).squeeze(-1)
        Outs = []
        for i in range(len(out)):
            o = self.CompOuts[i](out[i])
            Outs.append(o.squeeze(-1))
        self.wandblog()
        return op, Outs
    def wandblog(self):
        weightdict = {}
        weightdict['weights_Wave'] = self.Branching.weights[0].cpu().detach()
        weightdict['weights_Simple'] = self.Branching.weights[1].cpu().detach()
        weightdict['Tau'] = self.temp
        wandb.log(weightdict, commit=False)
def getFusion(fusionstr):
    if fusionstr == 'LinearFusion':
        fusion = LinearFusion
    elif fusionstr == 'AttentionFusion':
        fusion = AttentionFusion
    elif fusionstr == 'TransformerFusion':
        fusion = TransformerFusion
    elif fusionstr == 'HieLinFusion':
        fusion = HieLinFusion
    elif fusionstr == 'AvgFusion':
        fusion = AvgFusion
    elif fusionstr == 'WeightedAvgFusion':
        fusion = WeightedAvgFusion
    elif fusionstr == 'MaskedFusion':
        fusion = MaskedFusion
    elif fusionstr == 'MaskedFusionSwitch':
        fusion = MaskedFusionSwitch
    elif fusionstr == 'LinearFusionSameFC':
        fusion = LinearFusionSameFC
    elif fusionstr == 'MaskedGradLinear':
        fusion = MaskedGradLinear
    elif fusionstr == 'WeightedAvgEnsemble':
        fusion = WeightedAvgEnsemble
    elif fusionstr == 'SigWeightedFusion':
        fusion = SigWeightedFusion
    elif fusionstr == 'GumbelWeightedFusion':
        fusion = GumbelWeightedFusion
    elif fusionstr == 'MaskedFusionGradSwitch':
        fusion = MaskedFusionGradSwitch
    elif fusionstr == 'TrainableFusionSwitch':
        fusion = TrainableFusionSwitch
    elif fusionstr == 'SimpleWaveBranch':
        fusion = SimpleWaveBranch
    else:
        raise ValueError('Fusion value provided not found: ' + routinestr)
    return fusion