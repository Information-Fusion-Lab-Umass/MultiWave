import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb, copy
from abc import ABC, abstractmethod
import collections
from utils.ModelUtils import set_seed
from Models.TorchModels import getFreqModel

class RoutineClass(ABC):
    """
    Abstract base class for defining training routines.
    """
    def __init__(self):
        self.startingepochs = 0
        self.hascustombackward = False
        self.iswrapper = False
        self.stopatES = True

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model

    def setConfig(self, config):
        """
        Sets the configuration for the routine.
        
        Parameters:
        config (dict): Configuration dictionary.
        """
        self.config = config

    def SetSubRoutine(self, LossRoutine):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        pass

    def PreTrainStep(self, epoch):
        """
        Runs before the training step to allow modifications to the model.
        
        Parameters:
        epoch (int): Current epoch number.
        """
        pass

    @abstractmethod
    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        pass

    def saveLosses(self, TrainLosses, ValLosses, epoch):
        """
        Saves the training and validation losses.
        
        Parameters:
        TrainLosses (tuple): Training losses.
        ValLosses (tuple): Validation losses.
        epoch (int): Current epoch number.
        """
        if self.iswrapper:
            self.LossRoutine.saveLosses(TrainLosses, ValLosses, epoch)
        pass

def getEntropy(seq):
    """
    Computes the entropy of a sequence.
    
    Parameters:
    seq (torch.Tensor): Input sequence.
    
    Returns:
    torch.Tensor: Entropy of the sequence.
    """
    _, counts = seq.unique(return_counts=True)
    probs = counts.float() / len(seq)
    ent = torch.distributions.Categorical(probs).entropy()
    return ent

class LowToHighFreq(RoutineClass):
    """
    Training routine that trains components from low to high frequency.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(LowToHighFreq, self).__init__()
        self.epochstrain = epochstrain
        self.startingepochs = NumComps * epochstrain

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        indx = epoch // self.epochstrain
        if indx < len(CompLosses):
            outloss = CompLosses[indx]
        else:
            outloss = loss
        return outloss

class AllLosses(RoutineClass):
    """
    Training routine that combines all component losses.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(AllLosses, self).__init__()
        self.startingepochs = 0

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        wLoss = 1 / (len(CompLosses) + 1)
        outloss = wLoss * loss
        for loss in CompLosses:
            outloss += wLoss * loss
        return outloss

class OnlyLastLossWithWarming(RoutineClass):
    """
    Training routine that uses only the last loss with a warming period.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLossWithWarming, self).__init__()
        self.burninepochs = epochstrain
        self.startingepochs = epochstrain

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        wLoss = 1 / (len(CompLosses) + 1)
        outloss = loss
        if epoch < self.burninepochs:
            outloss = wLoss * loss
            for loss in CompLosses:
                outloss += wLoss * loss
        return outloss

class OnlyLastLossPretraining(RoutineClass):
    """
    Training routine that uses only the last loss with pretraining.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLossPretraining, self).__init__()
        self.burninepochs = epochstrain
        self.startingepochs = epochstrain

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        wLoss = 1
        outloss = loss
        if epoch < self.burninepochs:
            outloss = 0.0
            for loss in CompLosses:
                outloss += wLoss * loss
        return outloss

class OnlyLastLossPretrainingBest(RoutineClass):
    """
    Training routine that uses only the last loss with pretraining and saves the best models.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLossPretrainingBest, self).__init__()
        self.NumComps = NumComps
        self.burninepochs = epochstrain
        self.startingepochs = epochstrain
        self.bestLoss = np.ones(NumComps) * float('inf')
        self.loaded = False

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model
        self.bestModels = [copy.deepcopy(self.model.freqmodels[i].state_dict()) for i in range(self.NumComps)]
        self.bestCompOuts = [copy.deepcopy(self.model.fusion.CompOuts[i].state_dict()) for i in range(self.NumComps)]

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
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
        """
        Loads the best models and component outputs.
        """
        for i, model in enumerate(self.model.freqmodels):
            model.load_state_dict(self.bestModels[i])
            self.model.fusion.CompOuts[i].load_state_dict(self.bestCompOuts[i])

    def freezemodel(self, m):
        """
        Freezes the parameters of a model.
        
        Parameters:
        m (torch.nn.Module): The model to freeze.
        """
        for param in m.parameters():
            param.requires_grad = False

    def saveLosses(self, TrainLosses, ValLosses, epoch):
        """
        Saves the training and validation losses.
        
        Parameters:
        TrainLosses (tuple): Training losses.
        ValLosses (tuple): Validation losses.
        epoch (int): Current epoch number.
        """
        loss, comploss = TrainLosses
        valloss, valcomploss = ValLosses
        if epoch < self.burninepochs:
            for i, closs in enumerate(valcomploss):
                if closs < self.bestLoss[i]:
                    self.bestLoss[i] = closs
                    self.bestModels[i] = copy.deepcopy(self.model.freqmodels[i].state_dict())
                    self.bestCompOuts[i] = copy.deepcopy(self.model.fusion.CompOuts[i].state_dict())

class OnlyLastLoss(RoutineClass):
    """
    Training routine that uses only the last loss.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLoss, self).__init__()
        self.startingepochs = epochstrain

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        return loss

class OnlyLastLossGumbel(RoutineClass):
    """
    Training routine that uses only the last loss with Gumbel-softmax.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OnlyLastLossGumbel, self).__init__()
        self.startingepochs = 0

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        self.model.fusion.tau = 100 / (epoch + 1)
        return loss

class TwoThreeLoss(RoutineClass):
    """
    Training routine that combines the main loss with the second and third component losses.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(TwoThreeLoss, self).__init__()
        self.startingepochs = 0

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        wLoss = 1 / 4
        outloss = loss * 0.5
        outloss += CompLosses[1] * wLoss
        outloss += CompLosses[2] * wLoss
        return outloss

class TwoThreeLossEq(RoutineClass):
    """
    Training routine that combines the main loss with the second and third component losses equally.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(TwoThreeLossEq, self).__init__()
        self.startingepochs = 0

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = loss
        outloss += CompLosses[1]
        outloss += CompLosses[2]
        return outloss

class TrainAndFreeze(RoutineClass):
    """
    Training routine that trains and freezes components sequentially.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(TrainAndFreeze, self).__init__()
        self.epochstrain = epochstrain
        self.startingepochs = NumComps * epochstrain

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model
        for freqmodel in self.model.freqmodels:
            self.freezemodel(freqmodel)

    def freezemodel(self, m):
        """
        Freezes the parameters of a model.
        
        Parameters:
        m (torch.nn.Module): The model to freeze.
        """
        for param in m.parameters():
            param.requires_grad = False

    def unfreezemodel(self, m):
        """
        Unfreezes the parameters of a model.
        
        Parameters:
        m (torch.nn.Module): The model to unfreeze.
        """
        for param in m.parameters():
            param.requires_grad = True

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        indx = epoch // self.epochstrain
        for freqmodel in self.model.freqmodels:  # freeze all models
            self.freezemodel(freqmodel)
        if indx < len(CompLosses):
            self.unfreezemodel(self.model.freqmodels[indx])  # unfreeze the current model
            outloss = CompLosses[indx]
        else:
            outloss = loss
        return outloss
    
class CosineLosses(RoutineClass):
    """
    CosineLosses is a training routine that uses cosine similarity between gradients to adjust the training process.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(CosineLosses, self).__init__()
        self.startingepochs = 0
        self.CosineSigns = []
        self.burninepochs = epochstrain
        self.hascustombackward = True

    def wandblog(self, CosineSimilarities, CosEntropies, CosEntropiesAvg, epoch):
        """
        Logs cosine similarities and entropies to Weights & Biases.
        
        Parameters:
        CosineSimilarities (list): List of cosine similarities.
        CosEntropies (list): List of cosine entropies.
        CosEntropiesAvg (list): List of average cosine entropies.
        epoch (int): Current epoch number.
        """
        cosdict = {}
        for i, cossim in enumerate(CosineSimilarities):
            for j, cs in enumerate(cossim):
                cosdict['CosSim_' + self.names[j] + '_Sim_' + str(i)] = cs
                cosdict['CosSim_' + self.names[j] + '_Entropy_' + str(i)] = CosEntropies[i][j]
            cosdict['CosSimAvg_' + str(i)] = CosEntropiesAvg[i]
        cosdict['epoch'] = epoch
        wandb.log(cosdict, commit=False)

    def MaskGrads(self):
        """
        Masks gradients based on cosine similarities.
        """
        for i, freqmodel in enumerate(self.model.freqmodels):
            for j, (name, param) in enumerate(freqmodel.named_parameters()):
                param.grad *= self.Masks[i][j]

    def Backward(self, loss, Complosses):
        """
        Custom backward pass that applies gradient masking.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        Complosses (list): List of component losses.
        """
        for i, comploss in enumerate(Complosses):
            comploss.backward(retain_graph=True)
            for j, (name, param) in enumerate(self.model.freqmodels[i].named_parameters()):
                param.grad *= self.Masks[i][j]
        loss.backward()

    def getSigns(self, CosineSimilarities, epoch):
        """
        Computes the signs of cosine similarities and their entropies.
        
        Parameters:
        CosineSimilarities (list): List of cosine similarities.
        epoch (int): Current epoch number.
        
        Returns:
        list: List of cosine entropies.
        """
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
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        CosineSimilarities = self.getCosineLosses(loss, CompLosses, epoch)
        CosEntropies = self.getSigns(CosineSimilarities, epoch)
        CosEntropiesAvg = np.array(CosEntropies)
        CosEntropiesAvg = CosEntropiesAvg.mean(axis=0)
        self.wandblog(CosineSimilarities, CosEntropies, CosEntropiesAvg, epoch)
        outloss = loss
        return outloss

    def getCosineLosses(self, loss, CompLosses, epoch):
        """
        Computes cosine similarities between gradients of the main loss and component losses.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        list: List of cosine similarities.
        """
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
    """
    CosineLossesWrapper is a training routine that wraps another routine and uses cosine similarity between gradients.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(CosineLossesWrapper, self).__init__()
        self.CosineSigns = []
        self.burninepochs = epochstrain
        self.initialized = False
        self.iswrapper = True

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.LossRoutine = LossRoutine
        self.startingepochs = LossRoutine.startingepochs

    def wandblog(self, CosineSimilarities, CosEntropies, CosEntropiesAvg, epoch):
        """
        Logs cosine similarities and entropies to Weights & Biases.
        
        Parameters:
        CosineSimilarities (list): List of cosine similarities.
        CosEntropies (list): List of cosine entropies.
        CosEntropiesAvg (list): List of average cosine entropies.
        epoch (int): Current epoch number.
        """
        cosdict = {}
        for i, cossim in enumerate(CosineSimilarities):
            for j, cs in enumerate(cossim):
                cosdict['CosSim_' + self.names[j] + '_Sim_' + str(i)] = cs
                cosdict['CosSim_' + self.names[j] + '_Entropy_' + str(i)] = CosEntropies[i][j]
            cosdict['CosSimAvg_' + str(i)] = CosEntropiesAvg[i]
        cosdict['epoch'] = epoch
        wandb.log(cosdict)

    def getSigns(self, CosineSimilarities, epoch):
        """
        Computes the signs of cosine similarities and their entropies.
        
        Parameters:
        CosineSimilarities (list): List of cosine similarities.
        epoch (int): Current epoch number.
        
        Returns:
        list: List of cosine entropies.
        """
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
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        CosineSimilarities = self.getCosineLosses(loss, CompLosses, epoch)
        CosEntropies = self.getSigns(CosineSimilarities, epoch)
        CosEntropiesAvg = np.array(CosEntropies)
        CosEntropiesAvg = CosEntropiesAvg.mean(axis=1)
        self.wandblog(CosineSimilarities, CosEntropies, CosEntropiesAvg, epoch)
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        return outloss

    def getCosineLosses(self, loss, CompLosses, epoch):
        """
        Computes cosine similarities between gradients of the main loss and component losses.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        list: List of cosine similarities.
        """
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
    """
    CosineLossesWrapperSwitch is a training routine that wraps another routine and uses cosine similarity between gradients.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(CosineLossesWrapperSwitch, self).__init__()
        self.CosineSigns = []
        self.burninepochs = epochstrain
        self.entepochs = 5
        self.initialized = False
        self.threshold = 0.5
        self.iswrapper = True

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.LossRoutine = LossRoutine
        self.startingepochs = LossRoutine.startingepochs

    def wandblog(self, CosineSimilarities, CosEntropies, CosEntropiesAvg, Masks, epoch):
        """
        Logs cosine similarities, entropies, and masks to Weights & Biases.
        
        Parameters:
        CosineSimilarities (list): List of cosine similarities.
        CosEntropies (list): List of cosine entropies.
        CosEntropiesAvg (list): List of average cosine entropies.
        Masks (list): List of masks.
        epoch (int): Current epoch number.
        """
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
        """
        Computes the signs of cosine similarities and their entropies.
        
        Parameters:
        CosineSimilarities (list): List of cosine similarities.
        epoch (int): Current epoch number.
        
        Returns:
        list: List of cosine entropies.
        """
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
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
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
        """
        Computes cosine similarities between gradients of the main loss and component losses.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        list: List of cosine similarities.
        """
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
    """
    FreezeFinalFC is a training routine that freezes the final fully connected layer after a certain number of epochs.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(FreezeFinalFC, self).__init__()
        self.burninepochs = epochstrain
        self.SetSubRoutine()

    def SetSubRoutine(self, LossRoutine=OnlyLastLossWithWarming()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        LossRoutine = OnlyLastLossWithWarming(epochstrain=self.burninepochs)
        self.LossRoutine = LossRoutine
        self.startingepochs = LossRoutine.startingepochs

    def PreTrainStep(self, epoch):
        """
        Runs before the training step to freeze the final fully connected layer.
        
        Parameters:
        epoch (int): Current epoch number.
        """
        if epoch >= self.burninepochs:
            for param in self.model.fusion.foreLinear.parameters():
                param.requires_grad = False
            if self.model.fusion.useExtralin:
                for param in self.model.fusion.LastLinear.parameters():
                    param.requires_grad = False

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        self.wandblog(epoch)
        return outloss

    def wandblog(self, epoch):
        """
        Logs the parameters of the final fully connected layer to Weights & Biases.
        
        Parameters:
        epoch (int): Current epoch number.
        """
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
    """
    OGR is a training routine that adjusts the weights of the losses based on overfitting and generalization.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(OGR, self).__init__()
        self.startingepochs = epochstrain
        self.burninepochs = epochstrain
        self.NumComps = NumComps
        self.losshistory = {}
        self.lossWs = np.ones(NumComps + 1)
        self.loaded = False

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model
        self.modeldict = copy.deepcopy(model.state_dict())

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
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
        """
        Logs the weights of the losses to Weights & Biases.
        
        Parameters:
        lossWs (list): List of loss weights.
        epoch (int): Current epoch number.
        """
        weightdict = {}
        for i, W in enumerate(lossWs):
            if i == 0:
                weightdict['ModelW_Full'] = W
            else:
                weightdict['ModelW_' + str(i-1)] = W
        weightdict['epoch'] = epoch
        wandb.log(weightdict)

    def compute_overfit(self, val_loss, train_loss, prev_val_loss, prev_train_loss):
        """
        Computes the overfitting measure.
        
        Parameters:
        val_loss (float): Current validation loss.
        train_loss (float): Current training loss.
        prev_val_loss (float): Previous validation loss.
        prev_train_loss (float): Previous training loss.
        
        Returns:
        float: Overfitting measure.
        """
        new_O = val_loss - train_loss
        prev_O = prev_val_loss - prev_train_loss
        return new_O - prev_O

    def compute_gen(self, val_loss, prev_val_loss):
        """
        Computes the generalization measure.
        
        Parameters:
        val_loss (float): Current validation loss.
        prev_val_loss (float): Previous validation loss.
        
        Returns:
        float: Generalization measure.
        """
        return val_loss - prev_val_loss

    def compute_coef(self, val_loss, train_loss, prev_val_loss, prev_train_loss):
        """
        Computes the coefficient for adjusting the loss weights.
        
        Parameters:
        val_loss (float): Current validation loss.
        train_loss (float): Current training loss.
        prev_val_loss (float): Previous validation loss.
        prev_train_loss (float): Previous training loss.
        
        Returns:
        float: Coefficient for adjusting the loss weights.
        """
        overfit = self.compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss)
        gen = self.compute_gen(val_loss, prev_val_loss)
        return gen / (overfit * overfit)

    def compute_loss_coef(self, epoch):
        """
        Computes the coefficients for adjusting the loss weights.
        
        Parameters:
        epoch (int): Current epoch number.
        
        Returns:
        np.array: Coefficients for adjusting the loss weights.
        """
        num_models = self.num_models
        coef = np.zeros(num_models)
        prev_train_loss, prev_val_loss = self.losshistory[0]
        train_loss, val_loss = self.losshistory[epoch]
        for i in range(num_models):
            coef[i] = self.compute_coef(val_loss[i], train_loss[i], prev_val_loss[i], prev_train_loss[i])
        coef_sum = np.sum(coef)
        return coef / coef_sum

    def saveLosses(self, TrainLosses, ValLosses, epoch):
        """
        Saves the training and validation losses.
        
        Parameters:
        TrainLosses (tuple): Training losses.
        ValLosses (tuple): Validation losses.
        epoch (int): Current epoch number.
        """
        loss, comploss = TrainLosses
        TrainLosses_arr = [loss] + comploss
        self.num_models = len(TrainLosses_arr)
        loss, comploss = ValLosses
        ValLosses_arr = [loss] + comploss
        self.losshistory[epoch] = (TrainLosses_arr, ValLosses_arr)
        if epoch < self.burninepochs:
            self.lossWs = self.compute_loss_coef(epoch)

class LossSwitches(RoutineClass):
    """
    LossSwitches is a training routine that switches between different losses based on their performance.
    """
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
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model
        self.initmodeldict = copy.deepcopy(model.state_dict())

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = loss
        for closs in CompLosses:
            outloss += closs
        return outloss

    def wandblog(self, Masks, epoch):
        """
        Logs the masks to Weights & Biases.
        
        Parameters:
        Masks (list): List of masks.
        epoch (int): Current epoch number.
        """
        weightdict = {}
        for i, W in enumerate(Masks):
            weightdict['ModelMask_' + str(i)] = W
        weightdict['epoch'] = epoch
        wandb.log(weightdict)

    def saveLosses(self, TrainLosses, ValLosses, epoch):
        """
        Saves the training and validation losses. Also switches the the models based on their performance.
        
        Parameters:
        TrainLosses (tuple): Training losses.
        ValLosses (tuple): Validation losses.
        epoch (int): Current epoch number.
        """
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
    """
    LossSwitchesPretrain is a training routine that pretrains the model and then switches between different losses based on their performance.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(LossSwitchesPretrain, self).__init__()
        self.pretrainEpochs = epochstrain
        self.startingepochs = epochstrain * 3
        self.burninepochs = epochstrain * 2
        self.NumComps = NumComps
        self.losshistory = {}
        self.loaded = False
        self.currepoch = 0
        self.bestLosses = torch.ones(NumComps) * 1000
        self.LastLoss = 1000

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model
        self.initmodeldict = copy.deepcopy(model.state_dict())

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = loss
        if epoch <= self.pretrainEpochs:
            outloss = 0
        for closs in CompLosses:
            outloss += closs
        return outloss

    def wandblog(self, Masks, epoch):
        """
        Logs the masks to Weights & Biases.
        
        Parameters:
        Masks (list): List of masks.
        epoch (int): Current epoch number.
        """
        weightdict = {}
        for i, W in enumerate(Masks):
            weightdict['ModelMask_' + str(i)] = W
        weightdict['epoch'] = epoch
        wandb.log(weightdict)

    def TurnGradsOnOff(self, On=1.0):
        """
        Turns gradients on or off.
        
        Parameters:
        On (float): Value to set the gradients to (1.0 for on, 0.0 for off).
        """
        for i in range(len(self.model.fusion.GradMasks)):
            self.model.fusion.GradMasks[i] = On

    def saveLosses(self, TrainLosses, ValLosses, epoch):
        """
        Saves the training and validation losses.
        
        Parameters:
        TrainLosses (tuple): Training losses.
        ValLosses (tuple): Validation losses.
        epoch (int): Current epoch number.
        """
        self.currepoch += 1
        loss, comploss = TrainLosses
        vloss, vcomploss = ValLosses
        self.losshistory[epoch] = (comploss, vcomploss)
        self.bestLosses = torch.min(self.bestLosses, torch.tensor(vcomploss))
        if epoch == self.pretrainEpochs:
            print("saving model at epoch:", epoch)
            self.TurnGradsOnOff(On=0.0)  # Turn off grads
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
    """
    NormLossWrapperSign is a training routine that adds the sign of switch weights (uses sign of masks).
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(NormLossWrapperSign, self).__init__()
        self.startingepochs = epochstrain
        self.NumComps = NumComps
        self.OtherParams = OtherParams
        self.LW = OtherParams['LW']
        self.loaded = False
        self.iswrapper = True

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.model.fusion.switchweights.data = self.model.fusion.switchweights.data * self.OtherParams['InitWs']
        self.initmodeldict = copy.deepcopy(model.state_dict())

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.LossRoutine = LossRoutine

    def PreTrainStep(self, epoch):
        """
        Runs before the training step to adjust the switch weights.
        
        Parameters:
        epoch (int): Current epoch number.
        """
        if epoch == self.startingepochs:
            self.LW = 0
            if not self.loaded:
                BestMs = self.model.fusion.switchMasks.data
                self.model.fusion.switchweights.data = torch.sign(BestMs)
                self.model.fusion.switchweights.requires_grad = False
                self.loaded = True

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        norm = self.model.fusion.l1_norm()
        return outloss + norm * self.LW

class NormLossWrapper(RoutineClass):
    """
    NormLossWrapper is a training routine that adds a regularization term based on the L1 norm of the switch weights.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(NormLossWrapper, self).__init__()
        self.startingepochs = epochstrain
        self.NumComps = NumComps
        self.OtherParams = OtherParams
        self.LW = OtherParams['LW']
        self.loaded = False
        self.iswrapper = True

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.model.fusion.switchweights.data = self.model.fusion.switchweights.data * self.OtherParams['InitWs']
        self.initmodeldict = copy.deepcopy(model.state_dict())

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.LossRoutine = LossRoutine

    def PreTrainStep(self, epoch):
        """
        Runs before the training step to adjust the switch weights.
        
        Parameters:
        epoch (int): Current epoch number.
        """
        if epoch == self.startingepochs:
            self.LW = 0
            if not self.loaded:
                BestMs = self.model.fusion.switchMasks.data
                self.model.fusion.switchweights.data = BestMs / BestMs.sum()
                self.model.fusion.switchweights.requires_grad = False
                self.loaded = True

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        norm = self.model.fusion.l1_norm()
        return outloss + norm * self.LW

class FeatNormLossWrapper(RoutineClass):
    """
    FeatNormLossWrapper is a training routine that adds a regularization term based on the L1 norm of the feature mask weights.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(FeatNormLossWrapper, self).__init__()
        self.startingepochs = epochstrain
        self.NumComps = NumComps
        self.OtherParams = OtherParams
        self.LW = OtherParams['LW']
        self.loaded = False
        self.ModelHasFeatNorm = True

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.initmodeldict = copy.deepcopy(model.state_dict())

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.LossRoutine = LossRoutine
        self.LossRoutine.startingepochs = self.startingepochs

    def PreTrainStep(self, epoch):
        """
        Runs before the training step to adjust the feature mask weights.
        
        Parameters:
        epoch (int): Current epoch number.
        """
        if epoch == self.startingepochs and self.ModelHasFeatNorm:
            self.LW = 0
            if not self.loaded:
                for FMW in self.model.FeatMaskWeights:
                    FMW.requires_grad = False
                self.loaded = True

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        norm = 0.0
        if self.ModelHasFeatNorm:
            norm = self.model.l1_norm()
        self.wandblog(outloss, norm, outloss + norm * self.LW, epoch)
        return outloss + norm * self.LW

    def wandblog(self, loss, norm, totalloss, epoch):
        """
        Logs the losses to Weights & Biases.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        norm (torch.Tensor): L1 norm of the feature mask weights.
        totalloss (torch.Tensor): Total loss.
        epoch (int): Current epoch number.
        """
        lossdict = {}
        lossdict['Onlyloss'] = loss
        lossdict['MaskNorm'] = norm
        lossdict['totalloss'] = totalloss
        lossdict['epoch'] = epoch
        wandb.log(lossdict, commit=False)

class ResetModuleWrapper(RoutineClass):
    """
    ResetModuleWrapper is a training routine that resets the model based on feature mask norms.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(ResetModuleWrapper, self).__init__()
        self.stopatES = False
        self.SubWrapper = FeatNormLossWrapper(NumComps=NumComps, epochstrain=-1, OtherParams=OtherParams)

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.SubWrapper.setModelOptimizer(optimizer, model)
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.initmodeldict = copy.deepcopy(model.state_dict())

    def setConfig(self, config):
        """
        Sets the configuration for the routine.
        
        Parameters:
        config (dict): Configuration dictionary.
        """
        self.config = config
        self.LastModelSize = config['hs'][-1]
        config['hs'][-1] = 0

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.SubWrapper.SetSubRoutine(LossRoutine)

    def PreTrainStep(self, epoch):
        """
        Runs before the training step to adjust the feature mask weights.
        
        Parameters:
        epoch (int): Current epoch number.
        """
        self.SubWrapper.PreTrainStep(epoch)

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = self.SubWrapper.getLoss(loss, CompLosses, epoch)
        return outloss

    def ResetModel(self, es, checkpointPath, device):
        """
        Resets the model based on feature mask norms.
        
        Parameters:
        es (EarlyStopping): Early stopping object.
        checkpointPath (str): Path to the checkpoint file.
        device (torch.device): Device to load the model on.
        
        Returns:
        tuple: New model and optimizer.
        """
        print('Reseting the model .... ')
        self.stopatES = True
        es.restart()
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
        """
        Logs the losses to Weights & Biases.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        norm (torch.Tensor): L1 norm of the feature mask weights.
        totalloss (torch.Tensor): Total loss.
        epoch (int): Current epoch number.
        """
        pass

class ResetFeatWrapper(RoutineClass):
    """
    ResetFeatWrapper is a training routine that resets the model based on feature mask norms.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(ResetFeatWrapper, self).__init__()
        self.stopatES = False
        self.SubWrapper = FeatNormLossWrapper(NumComps=NumComps, epochstrain=-1, OtherParams=OtherParams)

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.SubWrapper.setModelOptimizer(optimizer, model)
        self.optimizer = optimizer
        self.model = model
        self.OptimInit = copy.deepcopy(optimizer.state_dict())
        self.initmodeldict = copy.deepcopy(model.state_dict())

    def setConfig(self, config):
        """
        Sets the configuration for the routine.
        
        Parameters:
        config (dict): Configuration dictionary.
        """
        self.config = config
        self.LastModelSize = config['hs'][-1]
        config['hs'][-1] = 0

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.SubWrapper.SetSubRoutine(LossRoutine)

    def PreTrainStep(self, epoch):
        """
        Runs before the training step to adjust the feature mask weights.
        
        Parameters:
        epoch (int): Current epoch number.
        """
        self.SubWrapper.PreTrainStep(epoch)

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = self.SubWrapper.getLoss(loss, CompLosses, epoch)
        return outloss

    def ResetModel(self, es, checkpointPath, device):
        """
        Resets the model based on feature mask norms.
        
        Parameters:
        es (EarlyStopping): Early stopping object.
        checkpointPath (str): Path to the checkpoint file.
        device (torch.device): Device to load the model on.
        
        Returns:
        tuple: New model and optimizer.
        """
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
        """
        Logs the losses to Weights & Biases.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        norm (torch.Tensor): L1 norm of the feature mask weights.
        totalloss (torch.Tensor): Total loss.
        epoch (int): Current epoch number.
        """
        pass

class NormLossWrapperNoStop(RoutineClass):
    """
    NormLossWrapperNoStop is a training routine that adds a regularization term based on the L1 norm of the switch weights without stopping.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(NormLossWrapperNoStop, self).__init__()
        self.startingepochs = epochstrain
        self.NumComps = NumComps
        self.OtherParams = OtherParams
        self.LW = OtherParams['LW']
        self.loaded = False
        self.iswrapper = True

    def setModelOptimizer(self, optimizer, model):
        """
        Sets the optimizer and model for the routine.
        
        Parameters:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        model (torch.nn.Module): The model to train.
        """
        self.optimizer = optimizer
        self.model = model

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.LossRoutine = LossRoutine
        self.LossRoutine.startingepochs = self.startingepochs

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        norm = self.model.l1_norm()
        self.wandblog(outloss, norm, outloss + norm * self.LW, epoch)
        return outloss + norm * self.LW

    def wandblog(self, loss, norm, totalloss, epoch):
        """
        Logs the losses to Weights & Biases.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        norm (torch.Tensor): L1 norm of the switch weights.
        totalloss (torch.Tensor): Total loss.
        epoch (int): Current epoch number.
        """
        lossdict = {}
        lossdict['Onlyloss'] = loss
        lossdict['MaskNorm'] = norm
        lossdict['totalloss'] = totalloss
        lossdict['epoch'] = epoch
        wandb.log(lossdict, commit=False)

class TempLossWrapper(RoutineClass):
    """
    TempLossWrapper is a training routine that adjusts the temperature parameter during training.
    """
    def __init__(self, NumComps=6, epochstrain=10, OtherParams=None):
        super(TempLossWrapper, self).__init__()
        self.startingepochs = epochstrain
        self.Inittemp = OtherParams['InitTemp']
        self.iswrapper = True

    def SetSubRoutine(self, LossRoutine=OnlyLastLoss()):
        """
        Sets a sub-routine for the training routine.
        
        Parameters:
        LossRoutine (RoutineClass): Sub-routine for the training routine.
        """
        self.LossRoutine = LossRoutine

    def getLoss(self, loss, CompLosses, epoch):
        """
        Computes the loss for the current training step and adjusts the temperature parameter.
        
        Parameters:
        loss (torch.Tensor): Main loss.
        CompLosses (list): List of component losses.
        epoch (int): Current epoch number.
        
        Returns:
        torch.Tensor: Computed loss.
        """
        outloss = self.LossRoutine.getLoss(loss, CompLosses, epoch)
        self.model.fusion.temp = self.Inittemp / (epoch + 1)
        return loss

def getRoutine(routinestr, NumComps=6, epochstrain=10, OtherParams=None):
    """
    Returns the appropriate training routine based on the specified routine string.
    
    Parameters:
    routinestr (str): Routine type.
    NumComps (int, optional): Number of components. Defaults to 6.
    epochstrain (int, optional): Number of epochs for training. Defaults to 10.
    OtherParams (dict, optional): Additional parameters for the routine. Defaults to None.
    
    Returns:
    RoutineClass: The appropriate training routine.
    
    Raises:
    ValueError: If the routine string is not recognized.
    """
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