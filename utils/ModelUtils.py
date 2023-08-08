from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn as nn
from utils.pytorchtools import EarlyStopping
import numpy as np
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

def converttoTensor(data):
    alldata = []
    for d in data:
        alldata.append(torch.tensor(d).float())
    return alldata
def set_seed(seed):
    print('setting seed to', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def wandblog(prfs, epoch, classification=True, postfix=''):
    if classification:
        for prf_k in prfs:
            wandb.log({'auc_' + prf_k + postfix: prfs[prf_k]['roc_macro'],
                   'auc_weighted_' + prf_k + postfix: prfs[prf_k]['roc_weighted'],
                   'fscore_' + prf_k + postfix: prfs[prf_k]['fscore_macro'],
                   'confMat_' + prf_k + postfix: prfs[prf_k]['confusionMatrix'],
                   'minrp_' + prf_k + postfix: prfs[prf_k]['minrp'],
                   'pr_auc_' + prf_k + postfix: prfs[prf_k]['pr_auc'],
                   'accuracy_' + prf_k + postfix: prfs[prf_k]['accuracy'],
                   'loss_' + prf_k + postfix: prfs[prf_k]['loss'],
                   'epoch': epoch})
    else:
        for prf_k in prfs:
            wandb.log({'mse_' + prf_k + postfix: prfs[prf_k]['mse'],
                   'mae_' + prf_k + postfix: prfs[prf_k]['mae'],
                   'r2_' + prf_k + postfix: prfs[prf_k]['r2'],
                   'epoch': epoch})

def wandbLossLogs(complosses, epoch):
    LossDict = {}
    for K in complosses:
        for i, L in enumerate(complosses[K]):
            LossDict['CompLoss_' + K + '_Model_' + str(i)] = L
    LossDict['epoch'] = epoch
    wandb.log(LossDict)
    
def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        output = torch.clamp(output,min=1e-7,max=1-1e-7)
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))
def weighted_cross_entropy(output, target, weights):
    return nn.CrossEntropyLoss(weight = weights)(output, target)

def getLoss(output, labels, class_weights=None, classification=True, NumClasses=1):
    if classification:
        if NumClasses == 1:
            loss = weighted_binary_cross_entropy(output, labels, class_weights)
        else:
            loss = weighted_cross_entropy(output, labels, class_weights)
    else:
        loss = nn.MSELoss()(output, labels)
    return loss

def train_step(epoch, model, device, train_loader, optimizer, class_weights, NumClasses=1, LossRoutine=None, classification=True, convertdirectly=False, scaler=None, Yscaled=False):
    model.train()
    correct = 0
    for batch_idx, (batch_in, labels) in enumerate(train_loader):
        labels = labels.to(device)
        if convertdirectly:
            batch_in = batch_in.to(device)
        else:
            for i in range(len(batch_in)):
                if type(batch_in[i]) == list:
                    for j in range(len(batch_in[i])):
                        batch_in[i][j] = batch_in[i][j].to(device)
                else:
                    batch_in[i] = batch_in[i].to(device)
        optimizer.zero_grad()
        output, compouts = model(batch_in)
        CompLosses = []
        for compout in compouts:
            L = getLoss(compout, labels, class_weights=class_weights, classification=classification, NumClasses=NumClasses)
            CompLosses.append(L)
        loss = getLoss(output, labels, class_weights=class_weights, classification=classification, NumClasses=NumClasses)
        if LossRoutine:
            loss = LossRoutine.getLoss(loss, CompLosses, epoch)
        if LossRoutine.hascustombackward:
            LossRoutine.Backward(loss, CompLosses)
        else:
            loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('batch_idx: {}\tLoss: {:.6f}'.format(
                batch_idx, loss.item()))
    train_prf, CompLosses = test(model, device, train_loader, class_weights=class_weights, classification=classification, convertdirectly=convertdirectly, scaler=scaler, Yscaled=Yscaled, NumClasses=NumClasses)
    return train_prf, CompLosses

def train(model, device, train_loader, val_loader, test_loader, optimizer, epochs, NumClasses=1, LossRoutine=None, class_weights = [1.0,1.0], classification=True, patience=5, checkpointPath='Checkpoints/CurrentChck', usewandb=True, convertdirectly=False, scaler=None, Yscaled=False, closewandb=True, wandbpostfix=''):
    initEpochs = 0
    if LossRoutine:
        initEpochs = LossRoutine.startingepochs
    es = EarlyStopping(initEpochs=initEpochs, patience= patience, path=checkpointPath, verbose=True)
    model.train()
    for epoch in range(epochs):
        LossRoutine.PreTrainStep(epoch)
#         if epoch == LossRoutine.startingepochs:
#             optimizer = optimizer.__class__(model.parameters(), **optimizer.defaults)
        train_prf, train_CompLosses = train_step(epoch, model, device, train_loader, optimizer, class_weights, classification=classification, convertdirectly=convertdirectly, scaler=scaler, Yscaled=Yscaled, LossRoutine=LossRoutine, NumClasses=NumClasses)
        if classification:
            print('\nEpoch: {}, Train set Accuracy: {:.2f}, Train set AUC macro: {:.4f}, Train set AUC weighted: {:.4f}\n'.format(epoch, train_prf['accuracy'], train_prf['roc_macro'], train_prf['roc_weighted']))
        else:
            print('\nEpoch: {}, Train set MSE: {:.4f}, Train set MAE: {:.4f}, Train set R2: {:.4f}\n'.format(epoch, train_prf['mse'], train_prf['mae'], train_prf['r2']))
        val_prf, val_CompLosses = test(model, device, val_loader, class_weights=class_weights, classification=classification, convertdirectly=convertdirectly, scaler=scaler, NumClasses=NumClasses)
        if classification:
            print('\nEpoch: {}, Val set Accuracy: {:.2f}, Val set AUC macro: {:.4f}, Val set AUC weighted: {:.4f}\n'.format(epoch, val_prf['accuracy'], val_prf['roc_macro'], val_prf['roc_weighted']))
            es(-(val_prf['roc_macro'] + val_prf['pr_auc']), model)
#             es(val_prf['loss'], model)
            LossRoutine.saveLosses((train_prf['loss'], train_CompLosses), (val_prf['loss'], val_CompLosses), epoch)
        else:
            print('\nEpoch: {}, Val set MSE: {:.4f}, Val set MAE: {:.4f}, Val set R2: {:.4f}\n'.format(epoch, val_prf['mse'], val_prf['mae'], val_prf['r2']))
            es(val_prf['mse'], model)
            LossRoutine.saveLosses((train_prf['mse'], train_CompLosses), (val_prf['mse'], val_CompLosses), epoch)
        if usewandb:
            wandblog({'train': train_prf, 'val': val_prf}, epoch, classification=classification, postfix=wandbpostfix)
            wandbLossLogs({'train': train_CompLosses, 'val': val_CompLosses}, epoch)
        if es.early_stop and LossRoutine.stopatES:
            print("Early stopping at epoch " + str(epoch))
            break
        elif es.early_stop:
            model, optimizer = LossRoutine.ResetModel(es, checkpointPath, device)
    model.load_state_dict(torch.load(checkpointPath))
    val_prf_final, val_CompLosses_final = test(model, device, val_loader, class_weights=class_weights, classification=classification, convertdirectly=convertdirectly, scaler=scaler, NumClasses=NumClasses)
    test_prf, test_CompLosses = test(model, device, test_loader, class_weights=class_weights, classification=classification, convertdirectly=convertdirectly, scaler=scaler, NumClasses=NumClasses)
    if usewandb:
        wandblog({'test': test_prf, 'val_final': val_prf_final}, epoch, classification=classification, postfix=wandbpostfix)
        wandbLossLogs({'test': test_CompLosses, 'val_final': val_CompLosses_final}, epoch)
        wandb.save(checkpointPath)
        if closewandb:
            wandb.finish()
    if classification:
        print('\nTest set Accuracy: {:.4f}, Test set f1: {:.4f}, Test set AUC macro: {:.4f}, Test set AUC weighted: {:.4f}, Test set Confmat: {}\n'.format(test_prf['accuracy'], test_prf['fscore_macro'], test_prf['roc_macro'], test_prf['roc_weighted'], str(test_prf['confusionMatrix'])))
    else:
        print('\nTest set MSE: {:.4f}, Test set MAE: {:.4f}, Test set R2: {:.4f}\n'.format(test_prf['mse'], test_prf['mae'], test_prf['r2']))
        #wandblog(train_prf, test_prf)

def get_pr_auc(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    minrp = np.minimum(precision, recall).max()
    return [pr_auc, minrp]

def Evaluate(Labels, Preds, PredScores, class_weights):
    avg = 'binary'
    NumClasses = 1
    if len(class_weights) > 2:
        avg = 'weighted'
        NumClasses = len(class_weights)
    if NumClasses > 1:
        PredScores = nn.Softmax(-1)(PredScores)
    percision, recall, fscore, support = precision_recall_fscore_support(Labels, Preds, average=avg)
    _, _, fscore_weighted, _ = precision_recall_fscore_support(Labels, Preds, average='weighted')
    _, _, fscore_macro, _ = precision_recall_fscore_support(Labels, Preds, average='macro')
    accuracy = accuracy_score(Labels, Preds) * 100
    confmat = confusion_matrix(Labels, Preds)
    loss = getLoss(PredScores.float(), Labels, class_weights=class_weights, classification=True, NumClasses=NumClasses)
    roc_macro, roc_weighted = roc_auc_score(Labels, PredScores, average='macro', multi_class='ovr'), roc_auc_score(Labels, PredScores, average='weighted', multi_class='ovr')
    if NumClasses == 1:
        pr_auc, minrp = get_pr_auc(Labels, PredScores)
    else:
        pr_auc, minrp = 0, 0
    prf_test = {'percision': percision, 'recall': recall, 'fscore': fscore, 'fscore_weighted': fscore_weighted,
                'fscore_macro': fscore_macro, 'accuracy': accuracy, 'confusionMatrix': confmat, 'roc_macro': roc_macro, 
                'roc_weighted': roc_weighted, 'loss': loss, 'minrp': minrp, 'pr_auc': pr_auc}
    return prf_test

"""
From https://en.wikipedia.org/wiki/Coefficient_of_determination
"""
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def EvaluateReg(Labels, Preds, PredScores, class_weights, scaler=None, Yscaled=False):
    if scaler is not None:
#         print('Preds before shape:', Preds.shape)
        Preds = scaler.inverse_transform(Preds.reshape(-1, 1))
        Preds = Preds.squeeze(-1)
#         print('Preds after shape:', Preds.shape)
        if Yscaled:
#             print('Labels before shape:', Labels.shape)
            Labels = scaler.inverse_transform(Labels.reshape(-1, 1))
            Labels = Labels.squeeze(-1)
#             print('Labels after shape:', Labels.shape)
    mae = mean_absolute_error(Labels, Preds)
    mse = mean_squared_error(Labels, Preds)
    r2 = r2_score(Preds, Labels)
    prf_test = {'Preds': Preds, 'Labels': Labels, 'r2': r2, 'mae':mae, 'mse':mse}
    return prf_test

def test(model, device, test_loader, class_weights=[1.0,1.0], classification=True, convertdirectly=False, scaler=None, Yscaled=False, NumClasses=1):
    model.eval()
    correct = 0
    corrects = torch.tensor([], dtype=torch.int64).to(device)
    preds = torch.tensor([], dtype=torch.int64).to(device)
    predScores = torch.tensor([], dtype=torch.float).to(device)
    FirstTime = True
    total_loss = 0.0
    total_num = 0
    CompoutsAll = []
    with torch.no_grad():
        for (batch_in, labels) in test_loader:
            labels = labels.to(device)
            if convertdirectly:
                batch_in = batch_in.to(device)
            else:
                for i in range(len(batch_in)):
                    if type(batch_in[i]) == list:
                        for j in range(len(batch_in[i])):
                            batch_in[i][j] = batch_in[i][j].to(device)
                    else:
                        batch_in[i] = batch_in[i].to(device)
            output, compouts = model(batch_in)
            if classification:
                if len(output.shape) > 1:
                    _, pred = torch.max(output.data, 1)
                else:
                    pred = output.data > 0.5
            else:
                pred = output.data
            if FirstTime:
                predScores = output; corrects = labels; preds = pred
                FirstTime = False
                CompoutsAll = compouts
            else:
                predScores = torch.cat((predScores, output))
                corrects = torch.cat((corrects, labels))
                preds = torch.cat((preds, pred))
                for i in range(len(CompoutsAll)):
                    CompoutsAll[i] = torch.cat((CompoutsAll[i], compouts[i]))
#     print('outs', corrects)
#     print('preds', preds)
#     print('predScores', predScores)
        CompoutLosses = []
        for compout in CompoutsAll:
            CompLoss = getLoss(compout, corrects, class_weights=class_weights, classification=classification, NumClasses=NumClasses)
            CompoutLosses.append(CompLoss)
        if classification:
            prf_test = Evaluate(corrects.cpu(), preds.cpu(), predScores.cpu(), class_weights.cpu())
        else:
            prf_test = EvaluateReg(corrects.cpu(), preds.cpu(), predScores.cpu(), class_weights.cpu(), scaler=scaler, Yscaled=Yscaled)
    #print(prf_test)
    return prf_test, CompoutLosses