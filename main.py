from utils.Dataset import get_RNNdataloader
import numpy as np
import torch

import pickle

from utils.ModelUtils import train, get_n_params, set_seed, boolean_string, converttoTensor
from Models.TorchModels import getFreqModel

import wandb, copy, argparse, os

from Models.Fusions import getFusion
from Models.Routines import getRoutine

from utils.WaveletUtils import getRNNFreqGroups_mr
from sklearn.utils.class_weight import compute_class_weight

import warnings
from sklearn.exceptions import UndefinedMetricWarning

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='The seed used for random seed generator', type=int, default=-1)
    parser.add_argument('--ExtraTag', help='extra tag for wandb', type=str, default='')
    parser.add_argument('--WandB', help='Use wandb', type=bool, default=True)
    parser.add_argument('--WandBEntity', help='Use wandb', type=str, default='')
    parser.add_argument('--d', help='dimension for all models', type=int, default=64)
    parser.add_argument('--hs', help='dimension for freq models', type=str, default=str([0,0,0,0,0,32]))
    parser.add_argument('--checkPath', help='path for checkpoint', type=str, default='')
    parser.add_argument('--Routine', help='The loss routine options: OnlyLastLoss, AllLosses, LowToHighFreq, CosineLosses', type=str, default='OnlyLastLoss')
    parser.add_argument('--SubRoutine', help='The routine for CosineSimilarityWrapper options: OnlyLastLoss, AllLosses, LowToHighFreq, CosineLosses', type=str, default='OnlyLastLossWithWarming')
    parser.add_argument('--epochstotrain', help='epochs to train on Routine', type=int, default=10)
    parser.add_argument('--UseExtraLinear', help='Use extra linear in fusion', type=boolean_string, default='False')
    parser.add_argument('--Fusion', help='The fusion options: LinearFusion, TransformerFusion, HieLinFusion, AttentionFusion', type=str, default='LinearFusion')
    parser.add_argument('--InitTemp', help='Initial temp for TempLossWrapper', type=float, default=10.0)
    parser.add_argument('--InitWs', help='Initial W multiplier for switch weights in NormLossWrapper', type=float, default=1.0)
    parser.add_argument('--LW', help='The norm weight in loss', type=float, default=2.0)
    parser.add_argument('--Model', help='The model for combining components', type=str, default='Modelfreq')
    parser.add_argument('--Comp', help='The model for components', type=str, default='BiLSTM')
    parser.add_argument('--NumLayers', help='number of layers for components', type=int, default=1)
    parser.add_argument('--WaveletType', help='the type of wavelet', type=str, default='db1')
    parser.add_argument('--LR', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--fold', help='The fold for running', type=int, default=1)
    args=parser.parse_args()

    # Convert the hidden unit string into a list of integers.
    hs = args.hs
    hs = list(map(int, hs.replace("[","").replace("]","").split(', ')))
    # Load the dataset for the specified fold from a pickle file.
    with open('datasets/WESAD/WESAD_data_fold' + str(args.fold) + '.pkl', 'rb') as f:
        (X_train, X_val, X_test, Y_train, Y_val, Y_test, times_train, times_val, times_test) = pickle.load(f)
    
    # Convert the training, validation, and test input data into tensors.
    X_train = converttoTensor(X_train)
    X_val = converttoTensor(X_val)
    X_test = converttoTensor(X_test)

    # Convert labels to tensors and extract class indices (assuming one-hot encoding).
    Y_train = torch.tensor(Y_train)
    Y_val = torch.tensor(Y_val)
    _, Y_train = Y_train.max(-1); _, Y_val = Y_val.max(-1)
    
    # Determine whether to apply regularization.
    regularize = True
    if 'perchannel' in args.Comp:
        # Disable regularization if the component type is 'perchannel'.
        regularize = False
    
    # Apply frequency grouping to the input data using wavelet decomposition.
    X_train_freq = getRNNFreqGroups_mr(X_train, times_train, maxlevels=len(hs)-2, imputation='forward', waveletType=args.WaveletType, regularize=regularize)
    X_val_freq = getRNNFreqGroups_mr(X_val, times_val, maxlevels=len(hs)-2, imputation='forward', waveletType=args.WaveletType, regularize=regularize)
    X_test_freq = getRNNFreqGroups_mr(X_test, times_test, maxlevels=len(hs)-2, imputation='forward', waveletType=args.WaveletType, regularize=regularize)
    
    ExtraTags = args.ExtraTag.split(',')
    # ExtraTags += ["fold" + str(args.fold)]
    seed = None
    if args.seed > -1:
        seed = args.seed
        set_seed(seed)
    print(' ... run starting ...', args)
    
    RoutineParams = {'LW': args.LW, 'InitWs': args.InitWs, 'InitTemp': args.InitTemp}
    routine = getRoutine(args.Routine, NumComps=len(hs), epochstrain=args.epochstotrain, OtherParams=RoutineParams)
    subroutine = getRoutine(args.SubRoutine, NumComps=len(hs), epochstrain=args.epochstotrain, OtherParams=RoutineParams)
    
    fusion = getFusion(args.Fusion)
    device = torch.device("cuda:0")
    
    bidirectional = False
    Comp = args.Comp
    if args.Comp == 'BiLSTM':
        bidirectional = True
    
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train.numpy()), y = Y_train.numpy())
    class_weights = torch.tensor(class_weights).to(device).float()
    if regularize:
        NumFeats = [x.shape[-1] for x in X_train_freq]
    else:
        NumFeats = [len(x) for x in X_train_freq]
    config = {'model': args.Model,
          'NumComps': len(hs),
          'NumFeats': NumFeats,
          'd': args.d,
          'hs': hs,
          'dropout': 0.0,
          'lr': args.LR,
          'patience': 15,
          'batch_size': 16,
          'seed': seed,
          'class_weights': class_weights,
          'Fusion': fusion,
          'RoutineEpochs': args.epochstotrain,
          'UseExtraLinear': args.UseExtraLinear,
          'LossRoutine': type(routine).__name__,
          'SubRoutine': type(subroutine).__name__,
          'Comp': Comp,
          'ExtraTags': ExtraTags,
          'NumLayers': args.NumLayers,
          'bidirectional': bidirectional,
          'NumHeads': 3,
          'InitMaskW' : args.InitWs,
          'Classification': True,
          'CNNKernelSize': 7,
          'WaveletType': args.WaveletType,
          'fold': args.fold,
          'NumClasses': 3,
          'FCNKernelMult': 1.0,
          'regularized': regularize} # HS: hidden size
    config.update(RoutineParams)
    routine.setConfig(config)
    subroutine.setConfig(config)
    
    modelfreq = getFreqModel(config)
    modelfreq = modelfreq.to(device)
    numparams = get_n_params(modelfreq)

    tags = [
            config['model']
        ]
    if args.ExtraTag != '':
        tags += ExtraTags
    if args.WandB:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(
            project="WESAD",
            config=copy.deepcopy(config),
            entity=args.WandBEntity,
            tags=tags,
        )
        wandb.log({'num_params': numparams})

    train_dataloaderFreq = get_RNNdataloader(X_train_freq, Y_train, config['batch_size'], shuffle=True, freq=True, regularized=regularize)
    val_dataloaderFreq = get_RNNdataloader(X_val_freq, Y_val, 128, shuffle=False, freq=True, regularized=regularize)
    test_dataloaderFreq = get_RNNdataloader(X_test_freq, Y_test, 128, shuffle=False, freq=True, regularized=regularize)

    optimizer = torch.optim.Adam(modelfreq.parameters(), lr=config['lr'])
#     optimizer = torch.optim.RMSprop(modelfreq.parameters(), lr=config['lr'])
    ChckPointfolder = 'Checkpoints/WESAD/'
    os.makedirs(ChckPointfolder, exist_ok=True)
    ChckPointPath = os.path.join(ChckPointfolder, 'CurrentChck_' + wandb.run.id)
    routine.setModelOptimizer(optimizer, modelfreq)
    subroutine.setModelOptimizer(optimizer, modelfreq)
    routine.SetSubRoutine(subroutine)
    train(modelfreq, device, train_dataloaderFreq, val_dataloaderFreq, test_dataloaderFreq, optimizer, 1000, 
      LossRoutine=routine, class_weights = config['class_weights'], patience = config['patience'], checkpointPath=ChckPointPath,  
      usewandb=True, convertdirectly=False, classification=config['Classification'], scaler=None, Yscaled=False, NumClasses=config['NumClasses'])
