import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import wandb, copy

from abc import ABC, abstractmethod

import collections
from utils.ModelUtils import set_seed
from Models.TorchModels import getFreqModel
 
class RoutineClass(ABC):
    def __init__(self):
        self.startingepochs = 0
        self.hascustombackward = False
        self.iswrapper = False
        self.stopatES = True
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
    def setConfig(self, config):
        self.config = config
    def SetSubRoutine(self, LossRoutine):
        pass
    def PreTrainStep(self, epoch): #Will run before the the training step so you can change the model here
        pass
    @abstractmethod
    def getLoss(self, loss, CompLosses, epoch):
        pass
    def saveLosses(self, TrainLosses, ValLosses, epoch):
        if self.iswrapper:
            self.LossRoutine.saveLosses(TrainLosses, ValLosses, epoch)
        pass

def getEntropy(seq):
    _, counts = seq.unique(return_counts=True)
    probs = counts.float() / len(seq)
    ent = torch.distributions.Categorical(probs).entropy()
    return ent

class LowToHighFreq(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(LowToHighFreq, self).__init__()
        self.epochstrain = epochstrain
        self.startingepochs = NumComps*epochstrain
    def getLoss(self, loss, CompLosses, epoch):
        indx = epoch // self.epochstrain
        if indx < len(CompLosses):
            outloss = CompLosses[indx]
        else:
            outloss = loss
        return outloss

class AllLosses(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(AllLosses, self).__init__()
        self.startingepochs = 0
    def getLoss(self, loss, CompLosses, epoch):
        wLoss = 1/(len(CompLosses) + 1)
        outloss = wLoss * loss
        for loss in CompLosses:
            outloss += wLoss * loss
        return outloss
    
class OnlyLastLossWithWarming(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLossWithWarming, self).__init__()
        self.burninepochs = epochstrain
        self.startingepochs = epochstrain
    def getLoss(self, loss, CompLosses, epoch):
        wLoss = 1/(len(CompLosses) + 1)
        outloss = loss
        if epoch < self.burninepochs:
            outloss = wLoss * loss
            for loss in CompLosses:
                outloss += wLoss * loss
        return outloss

class OnlyLastLossPretraining(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLossPretraining, self).__init__()
        self.burninepochs = epochstrain
        self.startingepochs = epochstrain
    def getLoss(self, loss, CompLosses, epoch):
#         wLoss = 1/(len(CompLosses) + 1)
        wLoss = 1
        outloss = loss
        if epoch < self.burninepochs:
            outloss = 0.0
            for loss in CompLosses:
                outloss += wLoss * loss
        return outloss

class OnlyLastLossPretrainingBest(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLossPretrainingBest, self).__init__()
        self.NumComps = NumComps
        self.burninepochs = epochstrain
        self.startingepochs = epochstrain
        self.bestLoss = np.ones(NumComps) * float('inf')
        self.loaded = False
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.bestModels = [copy.deepcopy(self.model.freqmodels[i].state_dict()) for i in range(self.NumComps)]
        self.bestCompOuts = [copy.deepcopy(self.model.fusion.CompOuts[i].state_dict()) for i in range(self.NumComps)]
    def getLoss(self, loss, CompLosses, epoch):
#         wLoss = 1/(len(CompLosses) + 1)
        wLoss = 1
        outloss = loss
        if epoch < self.burninepochs:
            outloss = 0.0
            for loss in CompLosses:
                outloss += wLoss * loss
        elif not self.loaded:
            self.loadmodels()
            self.loaded = True
        return outloss
    def loadmodels(self):
        for i, model in enumerate(self.model.freqmodels):
            model.load_state_dict(self.bestModels[i])
            self.model.fusion.CompOuts[i].load_state_dict(self.bestCompOuts[i])
#             self.freezemodel(model)
#             self.freezemodel(self.model.fusion.CompOuts[i])
    def freezemodel(self, m):
        for param in m.parameters():
            param.requires_grad = False
    def saveLosses(self, TrainLosses, ValLosses, epoch):
        loss, comploss = TrainLosses
        valloss, valcomploss = ValLosses
        if epoch < self.burninepochs:
            for i, closs in enumerate(valcomploss):
                if closs < self.bestLoss[i]:
                    self.bestLoss[i] = closs
                    self.bestModels[i] = copy.deepcopy(self.model.freqmodels[i].state_dict())
                    self.bestCompOuts[i] = copy.deepcopy(self.model.fusion.CompOuts[i].state_dict())
class OnlyLastLoss(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLoss, self).__init__()
        self.startingepochs = epochstrain
    def getLoss(self, loss, CompLosses, epoch):
        return loss

class OnlyLastLossGumbel(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLossGumbel, self).__init__()
        self.startingepochs = 0
    def getLoss(self, loss, CompLosses, epoch):
        self.model.fusion.tau = 100 / (epoch + 1)
        return loss

class TwoThreeLoss(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(TwoThreeLoss, self).__init__()
        self.startingepochs = 0
    def getLoss(self, loss, CompLosses, epoch):
        wLoss = 1/4
        outloss = loss * 0.5
        outloss += CompLosses[1] * wLoss
        outloss += CompLosses[2] * wLoss
        return outloss

class TwoThreeLossEq(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(TwoThreeLossEq, self).__init__()
        self.startingepochs = 0
    def getLoss(self, loss, CompLosses, epoch):
        outloss = loss
        outloss += CompLosses[1]
        outloss += CompLosses[2]
        return outloss

class TrainAndFreeze(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(TrainAndFreeze, self).__init__()
        self.epochstrain = epochstrain
        self.startingepochs = NumComps*epochstrain
    def setModelOptimizer(self):
        self.model = model
        for freqmodel in self.model.freqmodels:
            self.freezemodel(freqmodel)
    def freezemodel(self, m):
        for param in m.parameters():
            param.requires_grad = False
    def unfreezemodel(self, m):
        for param in m.parameters():
            param.requires_grad = True
    def getLoss(self, loss, CompLosses, epoch):
        indx = epoch // self.epochstrain
        for freqmodel in self.model.freqmodels: # freeze all models
            self.freezemodel(freqmodel)
        if indx < len(CompLosses):
            self.unfreezemodel(self.model.freqmodels[indx]) # unfreeze the current model
            outloss = CompLosses[indx]
        else:
            outloss = loss
        return outloss
    
class CosineLosses(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(CosineLosses, self).__init__()
        self.startingepochs = 0
        self.CosineSigns = []
        self.burninepochs = epochstrain
        self.hascustombackward = True
    def wandblog(self, CosineSimilarities, CosEntropies, CosEntropiesAvg, epoch):
        cosdict = {}
        for i, cossim in enumerate(CosineSimilarities):
            for j, cs in enumerate(cossim):
                cosdict['CosSim_' + self.names[j] + '_Sim_' + str(i)] = cs
                cosdict['CosSim_' + self.names[j] + '_Entropy_' + str(i)] = CosEntropies[i][j]
            cosdict['CosSimAvg_' + str(i)] = CosEntropiesAvg[i]
        cosdict['epoch'] = epoch
        wandb.log(cosdict, commit=False)
    def MaskGrads(self):
        for i, freqmodel in enumerate(self.model.freqmodels):
            for j, (name, param) in enumerate(freqmodel.named_parameters()):
                param.grad *= self.Masks[i][j]
    def Backward(self, loss, Complosses):
        for i, comploss in enumerate(Complosses):
            comploss.backward(retain_graph=True)
            for j, (name, param) in enumerate(self.model.freqmodels[i].named_parameters()):
                param.grad *= self.Masks[i][j]
        loss.backward()
    def getSigns(self, CosineSimilarities, epoch):
        if len(self.CosineSigns) == 0 or epoch == self.burninepochs:
            self.CosineSigns = [[torch.tensor([], device=CosineSimilarities[0][0].device) for _ in range(len(CosineSimilarities[i]))] for i in range(len(CosineSimilarities))]
        CosEntropies = [[0 for _ in range(len(CosineSimilarities[i]))] for i in range(len(CosineSimilarities))]
        self.Masks = [[0 for _ in range(len(CosineSimilarities[i]))] for i in range(len(CosineSimilarities))]
        for i, cossim in enumerate(CosineSimilarities):
            for j, cs in enumerate(cossim):
                self.CosineSigns[i][j] = torch.cat([self.CosineSigns[i][j], torch.sign(cs).reshape(1)])
                CosEntropies[i][j] = getEntropy(self.CosineSigns[i][j])
                self.Masks[i][j] = torch.clamp(torch.sign(cs), min=0)
        return CosEntropies
    def getLoss(self, loss, CompLosses, epoch):
        CosineSimilarities = self.getCosineLosses(loss, CompLosses, epoch)
        CosEntropies = self.getSigns(CosineSimilarities, epoch)
        CosEntropiesAvg = np.array(CosEntropies)
        CosEntropiesAvg = CosEntropiesAvg.mean(axis=0)
        self.wandblog(CosineSimilarities, CosEntropies, CosEntropiesAvg, epoch)
#         outloss = loss
#         wLoss = 1/(len(CompLosses) + 1)
#         outloss = wLoss * loss
#         for L in CompLosses:
#             outloss += wLoss * L
        outloss = loss
        return outloss
    def getCosineLosses(self, loss, CompLosses, epoch):
        CosineSimilarities = []
        cos = nn.CosineSimilarity(dim=0)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        freqModels_loss = []
        self.names = []
        for freqmodel in self.model.freqmodels:
            paramgrads = []
            for name, param in freqmodel.named_parameters():
                paramgrads.append(param.grad)
                self.names.append(name)
            freqModels_loss.append(paramgrads)
        for i, comploss in enumerate(CompLosses):
            self.optimizer.zero_grad()
            comploss.backward(retain_graph=True)
            CosineSim = []
            for j, (name, param) in enumerate(self.model.freqmodels[i].named_parameters()):
                CosineSim.append(cos(param.flatten(), freqModels_loss[i][j].flatten()))
            CosineSimilarities.append(CosineSim)
        self.optimizer.zero_grad()
        return CosineSimilarities
    
class CosineLossesWrapper(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(CosineLossesWrapper, self).__init__()
        self.CosineSigns = []
        self.burninepochs = epochstrain
        self.initialized = False
        self.iswrapper = True
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.LossRoutine = LossRoutine
        self.startingepochs = LossRoutine.startingepochs
    def wandblog(self, CosineSimilarities, CosEntropies, CosEntropiesAvg, epoch):
        cosdict = {}
        for i, cossim in enumerate(CosineSimilarities):
            for j, cs in enumerate(cossim):
                cosdict['CosSim_' + self.names[j] + '_Sim_' + str(i)] = cs
                cosdict['CosSim_' + self.names[j] + '_Entropy_' + str(i)] = CosEntropies[i][j]
            cosdict['CosSimAvg_' + str(i)] = CosEntropiesAvg[i]
        cosdict['epoch'] = epoch
        wandb.log(cosdict)
    def getSigns(self, CosineSimilarities, epoch):
        if len(self.CosineSigns) == 0 or (epoch == self.burninepochs and not self.initialized):
            self.CosineSigns = [[torch.tensor([], device=CosineSimilarities[0][0].device) for _ in range(len(CosineSimilarities[i]))] for i in range(len(CosineSimilarities))]
            if epoch == self.burninepochs:
                self.initialized = True
        CosEntropies = [[0 for _ in range(len(CosineSimilarities[i]))] for i in range(len(CosineSimilarities))]
        for i, cossim in enumerate(CosineSimilarities):
            for j, cs in enumerate(cossim):
                self.CosineSigns[i][j] = torch.cat([self.CosineSigns[i][j], torch.sign(cs).reshape(1)])
                CosEntropies[i][j] = getEntropy(self.CosineSigns[i][j])
        return CosEntropies
    def getLoss(self, loss, CompLosses, epoch):
        CosineSimilarities = self.getCosineLosses(loss, CompLosses, epoch)
        CosEntropies = self.getSigns(CosineSimilarities, epoch)
        CosEntropiesAvg = np.array(CosEntropies)
        CosEntropiesAvg = CosEntropiesAvg.mean(axis=1)
        self.wandblog(CosineSimilarities, CosEntropies, CosEntropiesAvg, epoch)
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        return outloss
    def getCosineLosses(self, loss, CompLosses, epoch):
        CosineSimilarities = []
        cos = nn.CosineSimilarity(dim=0)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        freqModels_loss = []
        self.names = []
        for freqmodel in self.model.freqmodels:
            paramgrads = []
            for name, param in freqmodel.named_parameters():
                paramgrads.append(param.grad)
                self.names.append(name)
            freqModels_loss.append(paramgrads)
        for i, comploss in enumerate(CompLosses):
            self.optimizer.zero_grad()
            comploss.backward(retain_graph=True)
            CosineSim = []
            for j, (name, param) in enumerate(self.model.freqmodels[i].named_parameters()):
                CosineSim.append(cos(param.flatten(), freqModels_loss[i][j].flatten()))
            CosineSimilarities.append(CosineSim)
        self.optimizer.zero_grad()
        return CosineSimilarities

class CosineLossesWrapperSwitch(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(CosineLossesWrapperSwitch, self).__init__()
        self.CosineSigns = []
        self.burninepochs = epochstrain
        self.entepochs = 5
        self.initialized = False
        self.threshold = 0.5
        self.iswrapper = True
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.LossRoutine = LossRoutine
        self.startingepochs = LossRoutine.startingepochs
    def wandblog(self, CosineSimilarities, CosEntropies, CosEntropiesAvg, Masks, epoch):
        cosdict = {}
        for i, cossim in enumerate(CosineSimilarities):
            for j, cs in enumerate(cossim):
                cosdict['CosSim_' + self.names[j] + '_Sim_' + str(i)] = cs
                cosdict['CosSim_' + self.names[j] + '_Entropy_' + str(i)] = CosEntropies[i][j]
            cosdict['CosSimAvg_' + str(i)] = CosEntropiesAvg[i]
            cosdict['ModelMask_' + str(i)] = Masks[i]
        cosdict['epoch'] = epoch
        wandb.log(cosdict)
    def getSigns(self, CosineSimilarities, epoch):
        if len(self.CosineSigns) == 0 or (epoch == self.burninepochs and not self.initialized):
            self.CosineSigns = [[torch.tensor([], device=CosineSimilarities[0][0].device) for _ in range(len(CosineSimilarities[i]))] for i in range(len(CosineSimilarities))]
            if epoch == self.burninepochs:
                self.initialized = True
        CosEntropies = [[0 for _ in range(len(CosineSimilarities[i]))] for i in range(len(CosineSimilarities))]
        for i, cossim in enumerate(CosineSimilarities):
            for j, cs in enumerate(cossim):
                self.CosineSigns[i][j] = torch.cat([self.CosineSigns[i][j], torch.sign(cs).reshape(1)])
                CosEntropies[i][j] = getEntropy(self.CosineSigns[i][j])
        return CosEntropies
    def getLoss(self, loss, CompLosses, epoch):
        CosineSimilarities = self.getCosineLosses(loss, CompLosses, epoch)
        CosEntropies = self.getSigns(CosineSimilarities, epoch)
        CosEntropiesAvg = np.array(CosEntropies)
        CosEntropiesAvg = CosEntropiesAvg.mean(axis=1)
        if epoch == self.startingepochs + self.entepochs:
            for i in range(len(CompLosses)):
                if CosEntropiesAvg[i] > self.threshold:
                    self.model.fusion.masks[i] = 0.0
                else:
                    self.model.fusion.masks[i] = 1.0
        self.wandblog(CosineSimilarities, CosEntropies, CosEntropiesAvg, self.model.fusion.masks, epoch)
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        return outloss
    def getCosineLosses(self, loss, CompLosses, epoch):
        CosineSimilarities = []
        cos = nn.CosineSimilarity(dim=0)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        freqModels_loss = []
        self.names = []
        for freqmodel in self.model.freqmodels:
            paramgrads = []
            for name, param in freqmodel.named_parameters():
                paramgrads.append(param.grad)
                self.names.append(name)
            freqModels_loss.append(paramgrads)
        for i, comploss in enumerate(CompLosses):
            self.optimizer.zero_grad()
            comploss.backward(retain_graph=True)
            CosineSim = []
            for j, (name, param) in enumerate(self.model.freqmodels[i].named_parameters()):
                CosineSim.append(cos(param.flatten(), freqModels_loss[i][j].flatten()))
            CosineSimilarities.append(CosineSim)
        self.optimizer.zero_grad()
        return CosineSimilarities

class FreezeFinalFC(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(FreezeFinalFC, self).__init__()
        self.burninepochs = epochstrain
        self.SetSubRoutine()
    def SetSubRoutine(self, LossRoutine=OnlyLastLossWithWarming()):
        LossRoutine=OnlyLastLossWithWarming(epochstrain=self.burninepochs)
        self.LossRoutine = LossRoutine
        self.startingepochs = LossRoutine.startingepochs
    def PreTrainStep(self, epoch):
        if epoch >= self.burninepochs:
            for param in self.model.fusion.foreLinear.parameters():
                param.requires_grad = False
            if self.model.fusion.useExtralin:
                for param in self.model.fusion.LastLinear.parameters():
                    param.requires_grad = False
    def getLoss(self, loss, CompLosses, epoch):
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        self.wandblog(epoch)
        return outloss
    def wandblog(self, epoch):
        weightdict = {}
        foreLinear_Param_avgs = []
        LastLinear_Param_avgs = []
        for param in self.model.fusion.foreLinear.parameters():
            foreLinear_Param_avgs.append(param.mean())
        if self.model.fusion.useExtralin:
            for param in self.model.fusion.LastLinear.parameters():
                LastLinear_Param_avgs.append(param.mean())
        weightdict['foreLinear_Param_avgs'] = torch.mean(torch.stack(foreLinear_Param_avgs))
        weightdict['LastLinear_Param_avgs'] = torch.mean(torch.stack(LastLinear_Param_avgs))
        for i, cfc in enumerate(self.model.fusion.CompOuts):
            weightdict['ModelFC_avgs_' + str(i)] = cfc.weight.mean()
        weightdict['epoch'] = epoch
        wandb.log(weightdict)

class OGR(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OGR, self).__init__()
        self.startingepochs = epochstrain
        self.burninepochs = epochstrain
        self.NumComps = NumComps
        self.losshistory = {}
        self.lossWs = np.ones(NumComps + 1)
        self.loaded = False
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.modeldict = copy.deepcopy(model.state_dict())
    def getLoss(self, loss, CompLosses, epoch):
        if epoch < self.burninepochs:
            outloss = loss
            for closs in CompLosses:
                outloss += closs
        else:
            if not self.loaded:
                self.model.load_state_dict(self.modeldict)
                self.loaded = True
            outloss = self.lossWs[0] * loss
            for i, closs in enumerate(CompLosses):
                outloss += self.lossWs[i+1] * closs
        self.wandblog(self.lossWs, epoch)
        return outloss
    def wandblog(self, lossWs, epoch):
        weightdict = {}
        for i, W in enumerate(lossWs):
            if i == 0:
                weightdict['ModelW_Full'] = W
            else:
                weightdict['ModelW_' + str(i-1)] = W
        weightdict['epoch'] = epoch
        wandb.log(weightdict)
    def compute_overfit(self, val_loss, train_loss, prev_val_loss, prev_train_loss):
        new_O = val_loss - train_loss
        prev_O = prev_val_loss - prev_train_loss
        return new_O - prev_O
    def compute_gen(self, val_loss, prev_val_loss):
        return val_loss - prev_val_loss
    def compute_coef(self, val_loss, train_loss, prev_val_loss, prev_train_loss):
        overfit = self.compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss)
        gen = self.compute_gen(val_loss, prev_val_loss)
        return gen / (overfit * overfit)
    def compute_loss_coef(self, epoch):
        num_models = self.num_models
        coef = np.zeros(num_models)
        prev_train_loss, prev_val_loss = self.losshistory[0]
        train_loss, val_loss = self.losshistory[epoch]
        for i in range(num_models):
            coef[i] = self.compute_coef(val_loss[i], train_loss[i], prev_val_loss[i], prev_train_loss[i])
        coef_sum = np.sum(coef)
        return coef / coef_sum
    def saveLosses(self, TrainLosses, ValLosses, epoch):
        loss, comploss = TrainLosses
        TrainLosses_arr = [loss] + comploss
        self.num_models = len(TrainLosses_arr)
        loss, comploss = ValLosses
        ValLosses_arr = [loss] + comploss
        self.losshistory[epoch] = (TrainLosses_arr, ValLosses_arr)
        if epoch < self.burninepochs:
            self.lossWs = self.compute_loss_coef(epoch)

class LossSwitches(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(LossSwitches, self).__init__()
        self.startingepochs = epochstrain
        self.burninepochs = epochstrain
        self.NumComps = NumComps
        self.losshistory = {}
        self.loaded = False
        self.currepoch = 0
        self.bestLosses = torch.ones(NumComps) * 1000
        self.LastLoss = 1000
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.initmodeldict = copy.deepcopy(model.state_dict())
    def getLoss(self, loss, CompLosses, epoch):
        outloss = loss
        for closs in CompLosses:
            outloss += closs
        return outloss
    def wandblog(self, Masks, epoch):
        weightdict = {}
        for i, W in enumerate(Masks):
            weightdict['ModelMask_' + str(i)] = W
        weightdict['epoch'] = epoch
        wandb.log(weightdict)
    def saveLosses(self, TrainLosses, ValLosses, epoch):
        self.currepoch += 1
        loss, comploss = TrainLosses
        vloss, vcomploss = ValLosses
        self.losshistory[epoch] = (comploss, vcomploss)
        self.bestLosses = torch.min(self.bestLosses, torch.tensor(vcomploss))
        if self.currepoch == self.burninepochs:
            if vloss < self.LastLoss:
                self.currepoch = 0
                for i in range(len(self.model.fusion.masks)):
                    if self.model.fusion.masks[i] == 0.0:
                        self.bestLosses[i] = 0.0
                WorstModel = torch.argmax(self.bestLosses)
                self.currWorstModel = WorstModel
                self.model.fusion.masks[WorstModel] = 0.0
                print('Removing model', str(WorstModel))
                self.LastLoss = vloss
                self.LastmodelDict = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(self.initmodeldict)
                self.bestLosses = torch.ones(self.NumComps) * 1000
            else:
                self.model.fusion.masks[self.currWorstModel] = 1.0
                print('Using current model', self.model.fusion.masks)
                self.model.load_state_dict(self.LastmodelDict)
        self.wandblog(self.model.fusion.masks, epoch)

class LossSwitchesPretrain(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(LossSwitchesPretrain, self).__init__()
        self.pretrainEpochs = epochstrain
        self.startingepochs = epochstrain + epochstrain + epochstrain
        self.burninepochs = epochstrain + epochstrain
        self.NumComps = NumComps
        self.losshistory = {}
        self.loaded = False
        self.currepoch = 0
        self.bestLosses = torch.ones(NumComps) * 1000
        self.LastLoss = 1000
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.initmodeldict = copy.deepcopy(model.state_dict())
    def getLoss(self, loss, CompLosses, epoch):
        outloss = loss
        if epoch <= self.pretrainEpochs:
            outloss = 0
        for closs in CompLosses:
            outloss += closs
        return outloss
    def wandblog(self, Masks, epoch):
        weightdict = {}
        for i, W in enumerate(Masks):
            weightdict['ModelMask_' + str(i)] = W
        weightdict['epoch'] = epoch
        wandb.log(weightdict)
    def TurnGradsOnOff(self, On=1.0):
        for i in range(len(self.model.fusion.GradMasks)):
            self.model.fusion.GradMasks[i] = On
    def saveLosses(self, TrainLosses, ValLosses, epoch):
        self.currepoch += 1
        loss, comploss = TrainLosses
        vloss, vcomploss = ValLosses
        self.losshistory[epoch] = (comploss, vcomploss)
        self.bestLosses = torch.min(self.bestLosses, torch.tensor(vcomploss))
        if epoch == self.pretrainEpochs:
            print("saving model at epoch:", epoch)
            self.TurnGradsOnOff(On=0.0) # Turn off grads
            self.initmodeldict = copy.deepcopy(self.model.state_dict())
        if self.currepoch == self.burninepochs:
            if vloss < self.LastLoss:
                self.currepoch = self.pretrainEpochs
                for i in range(len(self.model.fusion.masks)):
                    if self.model.fusion.masks[i] == 0.0:
                        self.bestLosses[i] = 0.0
                WorstModel = torch.argmax(self.bestLosses)
                self.currWorstModel = WorstModel
                self.model.fusion.masks[WorstModel] = 0.0
                print('Removing model', str(WorstModel))
                self.LastLoss = vloss
                self.LastmodelDict = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(self.initmodeldict)
                self.bestLosses = torch.ones(self.NumComps) * 1000
            else:
                self.model.fusion.masks[self.currWorstModel] = 1.0
                print('Using current model', self.model.fusion.masks)
                self.model.load_state_dict(self.LastmodelDict)
        self.wandblog(self.model.fusion.masks, epoch)
class NormLossWrapperSign(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(NormLossWrapperSign, self).__init__()
        self.startingepochs = epochstrain
        self.NumComps = NumComps
        self.OtherParams = OtherParams
        self.LW = OtherParams['LW']
        self.loaded = False
        self.iswrapper = True
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.model.fusion.switchweights.data = self.model.fusion.switchweights.data * self.OtherParams['InitWs']
        self.initmodeldict = copy.deepcopy(model.state_dict())
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.LossRoutine = LossRoutine
    def PreTrainStep(self, epoch):
        if epoch == self.startingepochs:
            self.LW = 0
            if not self.loaded:
#                 self.model.fusion.switchweights = self.model.fusion.switchweights / self.model.fusion.switchweights.sum()
                BestMs = self.model.fusion.switchMasks.data
#                 self.model.load_state_dict(self.initmodeldict)
                self.model.fusion.switchweights.data = toch.sign(BestMs)
                self.model.fusion.switchweights.requires_grad = False
#                 print('optim before:', self.optimizer.state_dict())
#                 print('optim before state:', self.optimizer.state)
#                 self.optimizer.load_state_dict(self.OptimInit)
#                 self.optimizer.state = collections.defaultdict(dict)
#                 print('optim:', self.optimizer.state_dict())
#                 print('optim after state:', self.optimizer.state)
                self.loaded = True
    def getLoss(self, loss, CompLosses, epoch):
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        norm = self.model.fusion.l1_norm()
        return outloss + norm * self.LW
class NormLossWrapper(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(NormLossWrapper, self).__init__()
        self.startingepochs = epochstrain
        self.NumComps = NumComps
        self.OtherParams = OtherParams
        self.LW = OtherParams['LW']
        self.loaded = False
        self.iswrapper = True
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.model.fusion.switchweights.data = self.model.fusion.switchweights.data * self.OtherParams['InitWs']
#         print('init optim:', self.OptimInit)
#         print('init optim state:', self.optimizer.state)
        self.initmodeldict = copy.deepcopy(model.state_dict())
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.LossRoutine = LossRoutine
#         self.startingepochs = LossRoutine.startingepochs
    def PreTrainStep(self, epoch):
        if epoch == self.startingepochs:
            self.LW = 0
            if not self.loaded:
#                 self.model.fusion.switchweights = self.model.fusion.switchweights / self.model.fusion.switchweights.sum()
                BestMs = self.model.fusion.switchMasks.data
#                 self.model.load_state_dict(self.initmodeldict)
                self.model.fusion.switchweights.data = BestMs / BestMs.sum()
                self.model.fusion.switchweights.requires_grad = False
#                 print('optim before:', self.optimizer.state_dict())
#                 print('optim before state:', self.optimizer.state)
#                 self.optimizer.load_state_dict(self.OptimInit)
#                 self.optimizer.state = collections.defaultdict(dict)
#                 print('optim:', self.optimizer.state_dict())
#                 print('optim after state:', self.optimizer.state)
                self.loaded = True
    def getLoss(self, loss, CompLosses, epoch):
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        norm = self.model.fusion.l1_norm()
        return outloss + norm * self.LW
class FeatNormLossWrapper(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(FeatNormLossWrapper, self).__init__()
        self.startingepochs = epochstrain
        self.NumComps = NumComps
        self.OtherParams = OtherParams
        self.LW = OtherParams['LW']
        self.loaded = False
        self.ModelHasFeatNorm = True
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
#         self.model.FeatMaskWeights.data = self.model.FeatMaskWeights.data * self.OtherParams['InitWs']
        self.initmodeldict = copy.deepcopy(model.state_dict())
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.LossRoutine = LossRoutine
        self.LossRoutine.startingepochs = self.startingepochs
    def PreTrainStep(self, epoch):
        if epoch == self.startingepochs and self.ModelHasFeatNorm:
            self.LW = 0
            if not self.loaded:
#                 BestMs = self.model.fusion.switchMasks.data
#                 self.model.fusion.switchweights.data = BestMs / BestMs.sum()
                for FMW in self.model.FeatMaskWeights:
                    FMW.requires_grad = False
                self.loaded = True
    def getLoss(self, loss, CompLosses, epoch):
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        norm = 0.0
        if self.ModelHasFeatNorm:
            norm = self.model.l1_norm()
        self.wandblog(outloss, norm, outloss + norm * self.LW, epoch)
        return outloss + norm * self.LW
    def wandblog(self, loss, norm, totalloss, epoch):
        lossdict = {}
        lossdict['Onlyloss'] = loss
        lossdict['MaskNorm'] = norm
        lossdict['totalloss'] = totalloss
        lossdict['epoch'] = epoch
        wandb.log(lossdict, commit=False)

class ResetModuleWrapper(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(ResetModuleWrapper, self).__init__()
        self.stopatES = False
        self.SubWrapper = FeatNormLossWrapper(NumComps=NumComps, epochstrain=-1, OtherParams=OtherParams)
    def setModelOptimizer(self, optimizer, model):
        self.SubWrapper.setModelOptimizer(optimizer, model)
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.initmodeldict = copy.deepcopy(model.state_dict())
    def setConfig(self, config):
        self.config = config
        self.LastModelSize = config['hs'][-1]
        config['hs'][-1] = 0
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.SubWrapper.SetSubRoutine(LossRoutine)
    def PreTrainStep(self, epoch):
        self.SubWrapper.PreTrainStep(epoch)
    def getLoss(self, loss, CompLosses, epoch):
        outloss = self.SubWrapper.getLoss(loss, CompLosses, epoch)
        return outloss
    def ResetModel(self, es, checkpointPath, device):
        print('Reseting the model .... ')
        self.stopatES = True
        es.restart()
#         self.model.load_state_dict(torch.load(checkpointPath))
        for i, FMW in enumerate(self.model.FeatMasks):
            norm = torch.norm(FMW, 1)
            print(i, norm)
            if norm == 0:
                print('Removing comp ' + str(i))
                self.config['hs'][i] = 0
        self.config['hs'][-1] = self.LastModelSize
        set_seed(self.config["seed"])
        self.config['model'] = 'Modelfreq'
        self.SubWrapper.ModelHasFeatNorm = False
        newModel = getFreqModel(self.config).to(device)
        optimizer = torch.optim.Adam(newModel.parameters(), lr=self.config['lr'])
        return newModel, optimizer
    def wandblog(self, loss, norm, totalloss, epoch):
        pass

class ResetFeatWrapper(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(ResetFeatWrapper, self).__init__()
        self.stopatES = False
        self.SubWrapper = FeatNormLossWrapper(NumComps=NumComps, epochstrain=-1, OtherParams=OtherParams)
    def setModelOptimizer(self, optimizer, model):
        self.SubWrapper.setModelOptimizer(optimizer, model)
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.initmodeldict = copy.deepcopy(model.state_dict())
    def setConfig(self, config):
        self.config = config
        self.LastModelSize = config['hs'][-1]
        config['hs'][-1] = 0
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.SubWrapper.SetSubRoutine(LossRoutine)
    def PreTrainStep(self, epoch):
        self.SubWrapper.PreTrainStep(epoch)
    def getLoss(self, loss, CompLosses, epoch):
        outloss = self.SubWrapper.getLoss(loss, CompLosses, epoch)
        return outloss
    def ResetModel(self, es, checkpointPath, device):
        print('Reseting the model .... ')
        self.stopatES = True
        es.restart()
        FeatIdxs = self.model.FeatIdxs
        self.model.load_state_dict(torch.load(checkpointPath))
        for i, FMW in enumerate(self.model.FeatMasks):
            norm = torch.norm(FMW, 1)
            print(i, norm)
            if norm == 0:
                print('Removing comp ' + str(i))
                self.config['hs'][i] = 0
            else:
                for j, FeatureMask in enumerate(FMW):
                    if FeatureMask == 0:
                        print('Removing feature ', j, ' In component ', i)
                        FeatIdxs[i][j] = False
                        self.config['NumFeats'][i] -= 1
        print('Routine', FeatIdxs)
        self.config['hs'][-1] = self.LastModelSize
        set_seed(self.config["seed"])
        self.config['model'] = 'Modelfreq'
        self.SubWrapper.ModelHasFeatNorm = False
        newModel = getFreqModel(self.config).to(device)
        newModel.FeatIdxs = FeatIdxs
        print('NewModel featIdx', newModel.FeatIdxs)
        optimizer = torch.optim.Adam(newModel.parameters(), lr=self.config['lr'])
        self.setModelOptimizer(optimizer, newModel)
        return newModel, optimizer
    def wandblog(self, loss, norm, totalloss, epoch):
        pass
    
class NormLossWrapperNoStop(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(NormLossWrapperNoStop, self).__init__()
        self.startingepochs = epochstrain
        self.NumComps = NumComps
        self.OtherParams = OtherParams
        self.LW = OtherParams['LW']
        self.loaded = False
        self.iswrapper = True
    def setModelOptimizer(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
#         self.OptimInit = copy.deepcopy(optimizer.state_dict())
#         self.model.fusion.switchweights.data = self.model.fusion.switchweights.data * self.OtherParams['InitWs']
#         print('init optim:', self.OptimInit)
#         print('init optim state:', self.optimizer.state)
#         self.initmodeldict = copy.deepcopy(model.state_dict())
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.LossRoutine = LossRoutine
        self.LossRoutine.startingepochs = self.startingepochs
#         self.startingepochs = LossRoutine.startingepochs
    def getLoss(self, loss, CompLosses, epoch):
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        norm = self.model.l1_norm()
        self.wandblog(outloss, norm, outloss + norm * self.LW, epoch)
        return outloss + norm * self.LW
    def wandblog(self, loss, norm, totalloss, epoch):
        lossdict = {}
        lossdict['Onlyloss'] = loss
        lossdict['MaskNorm'] = norm
        lossdict['totalloss'] = totalloss
        lossdict['epoch'] = epoch
        wandb.log(lossdict, commit=False)
class TempLossWrapper(RoutineClass):
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(TempLossWrapper, self).__init__()
        self.startingepochs = epochstrain
        self.Inittemp = OtherParams['InitTemp']
        self.iswrapper = True
    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        self.LossRoutine = LossRoutine
    def getLoss(self, loss, CompLosses, epoch):
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        self.model.fusion.temp = self.Inittemp / (epoch + 1)
        return loss
def getRoutine(routinestr, NumComps=6, epochstrain=10, OtherParams=None):
    if routinestr == 'OnlyLastLoss':
        routine = OnlyLastLoss(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'AllLosses':
        routine = AllLosses(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'LowToHighFreq':
        routine = LowToHighFreq(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'CosineLosses':
        routine = CosineLosses(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'CosineLossesWrapper':
        routine = CosineLossesWrapper(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'TwoThreeLoss':
        routine = TwoThreeLoss(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'TwoThreeLossEq':
        routine = TwoThreeLossEq(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'TrainAndFreeze':
        routine = TrainAndFreeze(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'OnlyLastLossWithWarming':
        routine = OnlyLastLossWithWarming(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'OnlyLastLossPretraining':
        routine = OnlyLastLossPretraining(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'OnlyLastLossPretrainingBest':
        routine = OnlyLastLossPretrainingBest(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'FreezeFinalFC':
        routine = FreezeFinalFC(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'OGR':
        routine = OGR(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'OnlyLastLossGumbel':
        routine = OnlyLastLossGumbel(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'LossSwitches':
        routine = LossSwitches(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'LossSwitchesPretrain':
        routine = LossSwitchesPretrain(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'CosineLossesWrapperSwitch':
        routine = CosineLossesWrapperSwitch(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'NormLossWrapper':
        routine = NormLossWrapper(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'TempLossWrapper':
        routine = TempLossWrapper(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'NormLossWrapperNoStop':
        routine = NormLossWrapperNoStop(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'FeatNormLossWrapper':
        routine = FeatNormLossWrapper(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'ResetModuleWrapper':
        routine = ResetModuleWrapper(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    elif routinestr == 'ResetFeatWrapper':
        routine = ResetFeatWrapper(NumComps=NumComps, epochstrain=epochstrain, OtherParams=OtherParams)
    else:
        raise ValueError('Routine value provided not found: ' + routinestr)
    return routine