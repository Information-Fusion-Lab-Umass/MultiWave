import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.fft import fft

class CVE(nn.Module):
    """
    CVE (Custom Variational Encoder) is a neural network module that performs a linear transformation
    followed by a non-linear activation and another linear transformation.
    """
    def __init__(self, hid_units, output_dim):
        """
        Initializes the CVE module.
        
        Parameters:
        hid_units (int): Number of hidden units in the first linear layer.
        output_dim (int): Dimension of the output.
        """
        super(CVE, self).__init__()
        self.hid_units = hid_units
        self.output_dim = output_dim
        self.W1 = nn.Linear(1, hid_units, bias=True)  # First linear layer
        self.W2 = nn.Linear(hid_units, output_dim, bias=False)  # Second linear layer
        self.reset_parameters()  # Initialize parameters with Xavier uniform distribution
        
    def forward(self, x):
        """
        Defines the forward pass of the CVE module.
        
        Parameters:
        x (Tensor): Input tensor.
        
        Returns:
        Tensor: Output tensor after applying the linear transformations and activation.
        """
        x = x.unsqueeze(-1)  # Add an extra dimension to the input tensor
        x = self.W2(torch.tanh(self.W1(x)))  # Apply first linear layer, tanh activation, and second linear layer
        return x
    
    def reset_parameters(self):
        """
        Initializes the parameters of the linear layers using Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.W1.weight)  # Initialize weights of the first linear layer
        nn.init.xavier_uniform_(self.W2.weight)  # Initialize weights of the second linear layer
        torch.nn.init.zeros_(self.W1.bias)  # Initialize biases of the first linear layer to zero
    
    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the CVE module given the input shape.
        
        Parameters:
        input_shape (tuple): Shape of the input tensor.
        
        Returns:
        tuple: Shape of the output tensor.
        """
        return input_shape + (self.output_dim,)
    
    
class Attention(nn.Module):
    """
    Attention is a neural network module that computes attention weights for input sequences.
    """
    
    def __init__(self, hid_dim, d):
        """
        Initializes the Attention module.
        
        Parameters:
        hid_dim (int): Number of hidden units in the attention mechanism.
        d (int): Dimension of the input features.
        """
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        self.W = nn.Linear(d, self.hid_dim, bias=True)  # Linear layer to transform input features
        self.u = nn.Linear(self.hid_dim, 1, bias=False)  # Linear layer to compute attention scores
        self.softmax = nn.Softmax(-2)  # Softmax activation to normalize attention scores
        self.reset_parameters()  # Initialize parameters with Xavier uniform distribution
        
    def forward(self, x, mask=None, mask_value=-1e30):
        """
        Defines the forward pass of the Attention module.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        mask (Tensor, optional): Mask tensor to apply to the attention weights. Defaults to None.
        mask_value (float, optional): Value to use for masked positions. Defaults to -1e30.
        
        Returns:
        Tensor: Attention weights of shape (batch_size, sequence_length, 1).
        """
        if mask is None:
            mask = torch.ones([x.shape[0], x.shape[1]], device=x.device)  # Create a mask of ones if not provided
        attn_weights = self.u(torch.tanh(self.W(x)))  # Compute attention scores
        mask = mask.unsqueeze(-1)  # Add an extra dimension to the mask
        attn_weights = mask * attn_weights + (1 - mask) * mask_value  # Apply mask to attention scores
        attn_weights = self.softmax(attn_weights)  # Normalize attention scores with softmax
        return attn_weights
        
    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the Attention module given the input shape.
        
        Parameters:
        input_shape (tuple): Shape of the input tensor.
        
        Returns:
        tuple: Shape of the output tensor.
        """
        return input_shape[:-1] + (1,)
    
    def reset_parameters(self):
        """
        Initializes the parameters of the linear layers using Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.W.weight)  # Initialize weights of the first linear layer
        nn.init.xavier_uniform_(self.u.weight)  # Initialize weights of the second linear layer
        torch.nn.init.zeros_(self.W.bias)  # Initialize biases of the first linear layer to zero
    
class Transformer(nn.Module):
    """
    Transformer is a neural network module that implements the Transformer architecture.
    """
    
    def __init__(self, d, N=2, h=8, dk=None, dv=None, dff=None, dropout=0):
        """
        Initializes the Transformer module.
        
        Parameters:
        d (int): Dimension of the input features.
        N (int): Number of layers. Defaults to 2.
        h (int): Number of heads. Defaults to 8.
        dk (int, optional): Dimension of the key vectors. Defaults to d // h.
        dv (int, optional): Dimension of the value vectors. Defaults to d // h.
        dff (int, optional): Dimension of the feed-forward network. Defaults to 2 * d.
        dropout (float, optional): Dropout rate. Defaults to 0.
        """
        super(Transformer, self).__init__()
        self.N, self.h, self.dk, self.dv, self.dff, self.dropout = N, h, dk, dv, dff, dropout
        eps = torch.finfo(torch.float32).eps
        self.epsilon = eps * eps
        if self.dk is None:
            self.dk = d // self.h
        if self.dv is None:
            self.dv = d // self.h
        if self.dff is None:
            self.dff = 2 * d
        self.Wq = nn.Parameter(torch.empty((self.N, self.h, d, self.dk)))
        self.Wk = nn.Parameter(torch.empty((self.N, self.h, d, self.dk)))
        self.Wv = nn.Parameter(torch.empty((self.N, self.h, d, self.dv)))
        self.Wo = nn.Parameter(torch.empty((self.N, self.dv * self.h, d)))
        self.W1 = nn.Parameter(torch.empty((self.N, d, self.dff)))
        self.b1 = nn.Parameter(torch.zeros((self.N, self.dff)))
        self.W2 = nn.Parameter(torch.empty((self.N, self.dff, d)))
        self.b2 = nn.Parameter(torch.zeros((self.N, d)))
        self.gamma = nn.Parameter(torch.zeros((2 * self.N,)))
        self.beta = nn.Parameter(torch.zeros((2 * self.N,)))
        self.dropoutA = nn.Dropout(p=self.dropout)
        self.dropoutproj = nn.Dropout(p=self.dropout)
        self.dropoutffn = nn.Dropout(p=self.dropout)
        self.reset_parameters()
        
    def forward(self, x, mask=None, mask_value=-1e-30):
        """
        Defines the forward pass of the Transformer module.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        mask (Tensor, optional): Mask tensor to apply to the attention weights. Defaults to None.
        mask_value (float, optional): Value to use for masked positions. Defaults to -1e-30.
        
        Returns:
        Tensor: Output tensor after applying the Transformer layers.
        """
        if mask:
            mask = mask.unsqueeze(-2)
        else:
            mask = torch.ones([x.shape[0], x.shape[1], 1], device=x.device)
        for i in range(self.N):
            # Multi-Head Attention (MHA)
            mha_ops = []
            for j in range(self.h):
                q = torch.matmul(x, self.Wq[i, j, :, :])
                k = torch.matmul(x, self.Wk[i, j, :, :]).permute((0, 2, 1))
                v = torch.matmul(x, self.Wv[i, j, :, :])
                A = torch.matmul(q, k)
                # Mask unobserved steps.
                A = mask * A + (1 - mask) * mask_value
                # Mask for attention dropout.
                A = self.dropoutA(A)
                A = nn.Softmax(dim=-1)(A)
                mha_ops.append(torch.matmul(A, v))
            conc = torch.cat(mha_ops, dim=-1)
            proj = torch.matmul(conc, self.Wo[i, :, :])
            # Dropout.
            proj = self.dropoutproj(proj)
            # Add & Layer Normalization (LN)
            x = x + proj
            mean = torch.mean(x, dim=-1, keepdim=True)
            variance = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
            std = torch.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x * self.gamma[2 * i] + self.beta[2 * i]
            # Feed-Forward Network (FFN)
            ffn_op = torch.matmul(nn.ReLU()(torch.matmul(x, self.W1[i, :, :]) + self.b1[i, :]), self.W2[i, :, :]) + self.b2[i, :, ]
            # Dropout.
            ffn_op = self.dropoutffn(ffn_op)
            # Add & Layer Normalization (LN)
            x = x + ffn_op
            mean = torch.mean(x, dim=-1, keepdim=True)
            variance = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
            std = torch.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x * self.gamma[2 * i + 1] + self.beta[2 * i + 1]
        return x
        
    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the Transformer module given the input shape.
        
        Parameters:
        input_shape (tuple): Shape of the input tensor.
        
        Returns:
        tuple: Shape of the output tensor.
        """
        return input_shape
    
    def reset_parameters(self):
        """
        Initializes the parameters of the linear layers using Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        nn.init.xavier_uniform_(self.Wo)
        nn.init.xavier_uniform_(self.W1)
        torch.nn.init.zeros_(self.b1)
        nn.init.xavier_uniform_(self.W2)
        torch.nn.init.zeros_(self.b2)
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

class RNNModel(nn.Module):
    """
    RNNModel is a neural network module that implements a recurrent neural network (RNN) using LSTM.
    """
    
    def __init__(self, dropout, h, d, numfeats, numLayers=1, bidirectional=True):
        """
        Initializes the RNNModel module.
        
        Parameters:
        dropout (float): Dropout rate.
        h (int): Number of hidden units.
        d (int): Dimension of the input features.
        numfeats (int): Number of input features.
        numLayers (int, optional): Number of LSTM layers. Defaults to 1.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        """
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(numfeats, h, numLayers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.DO1 = nn.Dropout(p=dropout)
        
    def forward(self, data):
        """
        Defines the forward pass of the RNNModel module.
        
        Parameters:
        data (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
        Tensor: Output tensor after applying the LSTM and dropout layers.
        """
        out, _ = self.lstm(data)
        out = self.DO1(out[:, -1, :])
        return out
    
    def load(self, checkpath):
        """
        Loads the model parameters from a checkpoint file.
        
        Parameters:
        checkpath (str): Path to the checkpoint file.
        """
        self.load_state_dict(torch.load(checkpath))
    
class TransformerModel(nn.Module):
    """
    TransformerModel is a neural network module that implements a Transformer-based model.
    """
    
    def __init__(self, dropout, h, d, numfeats, numLayers=1, NumHeads=10, bidirectional=True):
        """
        Initializes the TransformerModel module.
        
        Parameters:
        dropout (float): Dropout rate.
        h (int): Number of hidden units.
        d (int): Dimension of the input features.
        numfeats (int): Number of input features.
        numLayers (int, optional): Number of Transformer layers. Defaults to 1.
        NumHeads (int, optional): Number of attention heads. Defaults to 10.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        """
        super(TransformerModel, self).__init__()
        self.linearFirst = nn.Linear(numfeats, h)
        self.transformer = Transformer(d=h, N=numLayers, h=NumHeads, dk=None, dv=None, dff=None, dropout=dropout)
        self.attn = Attention(2 * h, h)
        self.DO1 = nn.Dropout(p=dropout)
        
    def forward(self, data):
        """
        Defines the forward pass of the TransformerModel module.
        
        Parameters:
        data (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
        Tensor: Output tensor after applying the Transformer, attention, and dropout layers.
        """
        out = self.linearFirst(data)
        out = self.transformer(out)
        attn_weights = self.attn(out)
        out = torch.sum(out * attn_weights, dim=-2)
        out = self.DO1(out)
        return out
    
    def load(self, checkpath):
        """
        Loads the model parameters from a checkpoint file.
        
        Parameters:
        checkpath (str): Path to the checkpoint file.
        """
        self.load_state_dict(torch.load(checkpath))


class BlockFCNConv(nn.Module):
    """
    BlockFCNConv is a convolutional block used in the Fully Convolutional Network (FCN).
    """
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
        """
        Initializes the BlockFCNConv module.
        
        Parameters:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        momentum (float): Momentum for batch normalization.
        epsilon (float): Epsilon for batch normalization.
        squeeze (bool): Whether to apply squeeze operation.
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Defines the forward pass of the BlockFCNConv module.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, num_variables, time_steps).
        
        Returns:
        Tensor: Output tensor after applying convolution, batch normalization, and ReLU activation.
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        y = self.relu(x)
        return y

class FCN(nn.Module):
    """
    FCN is a Fully Convolutional Network for time series data.
    """
    def __init__(self, dropout, h, d, numfeats, kernels=[8, 5, 3], kernelsizemult=1.0, mom=0.99, eps=0.001):
        """
        Initializes the FCN module.
        
        Parameters:
        dropout (float): Dropout rate.
        h (int): Number of hidden units.
        d (int): Dimension of the input features.
        numfeats (int): Number of input features.
        kernels (list): List of kernel sizes for convolutional layers.
        kernelsizemult (float): Multiplier for kernel sizes.
        mom (float): Momentum for batch normalization.
        eps (float): Epsilon for batch normalization.
        """
        super().__init__()
        kernels = np.array(kernels) * kernelsizemult
        kernels = kernels.astype(int)
        channels = [h, 2*h, h]
        self.conv1 = BlockFCNConv(numfeats, channels[0], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[0], channels[1], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[1], channels[2], kernels[2], momentum=mom, epsilon=eps)
        self.DO1 = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Defines the forward pass of the FCN module.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
        Tensor: Output tensor after applying convolutional layers and dropout.
        """
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.DO1(x)
        y = x.mean(axis=-1)
        return y

class FCN_perchannel(nn.Module):
    """
    FCN_perchannel is a Fully Convolutional Network that processes each channel separately.
    """
    def __init__(self, dropout, h, d, numfeats, kernels=[8, 5, 3], kernelsizemult=1.0, mom=0.99, eps=0.001, input_size_perchannel=1):
        """
        Initializes the FCN_perchannel module.
        
        Parameters:
        dropout (float): Dropout rate.
        h (int): Number of hidden units.
        d (int): Dimension of the input features.
        numfeats (int): Number of input features.
        kernels (list): List of kernel sizes for convolutional layers.
        kernelsizemult (float): Multiplier for kernel sizes.
        mom (float): Momentum for batch normalization.
        eps (float): Epsilon for batch normalization.
        input_size_perchannel (int): Size of the input for each channel.
        """
        super().__init__()
        kernels = np.array(kernels) * kernelsizemult
        kernels = kernels.astype(int)
        channels = [h, 2*h, h]
        self.FCNs = []
        self.finalAct = nn.ReLU()
        for i in range(numfeats):
            fcn_convs = nn.Sequential(
                BlockFCNConv(input_size_perchannel, channels[0], kernels[0], momentum=mom, epsilon=eps, squeeze=True),
                BlockFCNConv(channels[0], channels[1], kernels[1], momentum=mom, epsilon=eps, squeeze=True),
                BlockFCNConv(channels[1], channels[2], kernels[2], momentum=mom, epsilon=eps),
                nn.Dropout(p=dropout))
            self.FCNs.append(fcn_convs)
        self.FCNs = nn.ModuleList(self.FCNs)
        self.LastLinear = nn.Linear(numfeats * h, h)
        print('numfeats', numfeats)
    
    def forward(self, x):
        """
        Defines the forward pass of the FCN_perchannel module.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
        Tensor: Output tensor after applying convolutional layers and dropout.
        """
        Outs = []
        for i in range(len(x)):
            if len(x[i].shape) == 2:
                Out = self.FCNs[i](x[i][:, None, :])
            else:
                Out = self.FCNs[i](x[i])
            Out = Out.mean(axis=-1)
            Outs.append(Out)
        y = torch.cat(Outs, 1)
        y = self.LastLinear(y)
        return y

class Transformer_perchannel(nn.Module):
    """
    Transformer_perchannel is a Transformer-based model that processes each channel separately.
    """
    def __init__(self, dropout, h, d, numfeats, numLayers=1, NumHeads=10, bidirectional=True):
        """
        Initializes the Transformer_perchannel module.
        
        Parameters:
        dropout (float): Dropout rate.
        h (int): Number of hidden units.
        d (int): Dimension of the input features.
        numfeats (int): Number of input features.
        numLayers (int, optional): Number of Transformer layers. Defaults to 1.
        NumHeads (int, optional): Number of attention heads. Defaults to 10.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        """
        super().__init__()
        self.Transformers = []
        self.finalAct = nn.ReLU()
        for i in range(numfeats):
            transformermodel = TransformerModel(dropout=dropout, h=h, d=d, numfeats=1, numLayers=numLayers, NumHeads=NumHeads, bidirectional=bidirectional)
            self.Transformers.append(transformermodel)
        self.Transformers = nn.ModuleList(self.Transformers)
        self.LastLinear = nn.Linear(numfeats * h, h)
        
    def forward(self, x):
        """
        Defines the forward pass of the Transformer_perchannel module.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
        Tensor: Output tensor after applying Transformer layers and dropout.
        """
        Outs = []
        for i in range(len(x)):
            Out = self.Transformers[i](x[i][:, :, None])
            Outs.append(Out)
        y = torch.cat(Outs, 1)
        y = self.LastLinear(y)
        return y

class CNNAttnModel(nn.Module):
    """
    CNNAttnModel is a neural network module that combines CNN and attention mechanisms.
    """
    def __init__(self, dropout, h, d, numfeats, kernelsize=3, numLayers=1, bidirectional=True):
        """
        Initializes the CNNAttnModel module.
        
        Parameters:
        dropout (float): Dropout rate.
        h (int): Number of hidden units.
        d (int): Dimension of the input features.
        numfeats (int): Number of input features.
        kernelsize (int, optional): Size of the convolutional kernel. Defaults to 3.
        numLayers (int, optional): Number of CNN layers. Defaults to 1.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        """
        super(CNNAttnModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(numfeats, h, kernel_size=kernelsize, padding=kernelsize//2),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        for i in range(1, numLayers):
            self.cnn_layers.add_module("conv_" + str(i), nn.Conv1d(h, h, kernel_size=kernelsize, padding=kernelsize//2))
            self.cnn_layers.add_module("batchnorm_" + str(i), nn.BatchNorm1d(h))
            self.cnn_layers.add_module("relu_" + str(i), nn.ReLU(inplace=True))
            self.cnn_layers.add_module("maxpool_" + str(i), nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.DO1 = nn.Dropout(p=dropout)
        self.attn = Attention(2*h, h)
        
    def forward(self, data):
        """
        Defines the forward pass of the CNNAttnModel module.
        
        Parameters:
        data (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
        Tensor: Output tensor after applying CNN and attention layers.
        """
        out = self.cnn_layers(data.transpose(1, 2))
        attn_weights = self.attn(out.transpose(1, 2))
        out = torch.sum(out.transpose(1, 2) * attn_weights, dim=-2)
        self.DO1(out)
        return out
        
    def load(self, checkpath):
        """
        Loads the model parameters from a checkpoint file.
        
        Parameters:
        checkpath (str): Path to the checkpoint file.
        """
        self.load_state_dict(torch.load(checkpath))

class CNNAttn_perchannel(nn.Module):
    """
    CNNAttn_perchannel is a CNN and attention-based model that processes each channel separately.
    """
    def __init__(self, dropout, h, d, numfeats, kernelsize=3, numLayers=1, bidirectional=True):
        """
        Initializes the CNNAttn_perchannel module.
        
        Parameters:
        dropout (float): Dropout rate.
        h (int): Number of hidden units.
        d (int): Dimension of the input features.
        numfeats (int): Number of input features.
        kernelsize (int, optional): Size of the convolutional kernel. Defaults to 3.
        numLayers (int, optional): Number of CNN layers. Defaults to 1.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        """
        super(CNNAttn_perchannel, self).__init__()
        self.CNNmodels = []
        self.finalAct = nn.ReLU()
        for i in range(numfeats):
            cnnmodel = CNNAttnModel(dropout=dropout, h=h, d=d, numfeats=1, kernelsize=kernelsize, numLayers=numLayers, bidirectional=bidirectional)
            self.CNNmodels.append(cnnmodel)
        self.CNNmodels = nn.ModuleList(self.CNNmodels)
        self.LastLinear = nn.Linear(numfeats * h, h)
        
    def forward(self, x):
        """
        Defines the forward pass of the CNNAttn_perchannel module.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
        Tensor: Output tensor after applying CNN and attention layers.
        """
        Outs = []
        for i in range(len(x)):
            Out = self.CNNmodels[i](x[i][:, :, None])
            Outs.append(Out)
        y = torch.cat(Outs, 1)
        y = self.LastLinear(y)
        return y

class CNNLSTMModel(nn.Module):
    """
    CNNLSTMModel is a neural network module that combines CNN and LSTM layers.
    """
    def __init__(self, dropout, h, d, numfeats, kernelsize=3, numLayers=1, bidirectional=True):
        """
        Initializes the CNNLSTMModel module.
        
        Parameters:
        dropout (float): Dropout rate.
        h (int): Number of hidden units.
        d (int): Dimension of the input features.
        numfeats (int): Number of input features.
        kernelsize (int, optional): Size of the convolutional kernel. Defaults to 3.
        numLayers (int, optional): Number of CNN layers. Defaults to 1.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        """
        super(CNNLSTMModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(numfeats, h, kernel_size=kernelsize, padding=kernelsize//2),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        for i in range(1, numLayers):
            self.cnn_layers.add_module("conv_" + str(i), nn.Conv1d(h, h, kernel_size=kernelsize, padding=kernelsize//2))
            self.cnn_layers.add_module("batchnorm_" + str(i), nn.BatchNorm1d(h))
            self.cnn_layers.add_module("relu_" + str(i), nn.ReLU(inplace=True))
            self.cnn_layers.add_module("maxpool_" + str(i), nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.DO1 = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(h, h, 1, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
    def forward(self, data):
        """
        Defines the forward pass of the CNNLSTMModel module.
        
        Parameters:
        data (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
        Tensor: Output tensor after applying CNN and LSTM layers.
        """
        out = self.cnn_layers(data.transpose(1, 2))
        out = self.DO1(out.transpose(1, 2))
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        return out
        
    def load(self, checkpath):
        """
        Loads the model parameters from a checkpoint file.
        
        Parameters:
        checkpath (str): Path to the checkpoint file.
        """
        self.load_state_dict(torch.load(checkpath))


class TA_LSTMCell(nn.Module):
    """
    TA_LSTMCell is a custom LSTM cell that incorporates time-aware mechanisms.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initializes the TA_LSTMCell module.
        
        Parameters:
        input_size (int): Dimension of the input features.
        hidden_size (int): Number of hidden units.
        """
        super(TA_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.W_decomp = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_decomp = nn.Parameter(torch.randn(hidden_size))
        
    def g(self, t):
        """
        Computes the time-aware gating mechanism.
        
        Parameters:
        t (Tensor): Time intervals.
        
        Returns:
        Tensor: Time-aware gating values.
        """
        T = torch.zeros_like(t).to(t.device)
        T[t.nonzero(as_tuple=True)] = 1 / t[t.nonzero(as_tuple=True)]
        
        Ones = torch.ones([1, self.hidden_size], dtype=torch.float32).to(t.device)
        T = torch.mm(T, Ones)
        return T      

    def forward(self, input, t, state):
        """
        Defines the forward pass of the TA_LSTMCell module.
        
        Parameters:
        input (Tensor): Input tensor.
        t (Tensor): Time intervals.
        state (Tuple[Tensor, Tensor]): Previous hidden and cell states.
        
        Returns:
        Tuple[Tensor, Tensor]: Updated hidden and cell states.
        """
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        T = self.g(t)
        
        C_ST = torch.tanh(torch.mm(cx, self.W_decomp) + self.b_decomp)
        C_ST_dis = T * C_ST
        cx = cx - C_ST + C_ST_dis
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return (hy, cy)

class TA_LSTM(nn.Module):
    """
    TA_LSTM is a custom LSTM module that incorporates time-aware mechanisms.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initializes the TA_LSTM module.
        
        Parameters:
        input_size (int): Dimension of the input features.
        hidden_size (int): Number of hidden units.
        """
        super(TA_LSTM, self).__init__()
        self.TA_lstm = TA_LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, X, time):
        """
        Defines the forward pass of the TA_LSTM module.
        
        Parameters:
        X (Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        time (Tensor): Time intervals.
        
        Returns:
        Tensor: Output tensor after applying the TA_LSTM layers.
        """
        time = time[None, :, None].repeat(X.shape[0], 1, 1).float().to(X.device)
        c = torch.zeros([X.shape[0], self.hidden_size]).to(X.device)
        h = torch.zeros([X.shape[0], self.hidden_size]).to(X.device)
        state = (h, c)
        AllStates = []
        time_steps = X.shape[1]
        for i in range(time_steps):
            state = self.TA_lstm(X[:, i, :], time[:, i, :], state)
            AllStates.append(state[0])
        return state[0]

def get_Model(Comp, dropout, multiplier, h, d, NumFeats, numLayers, bidirectional, config, input_size_perchannel=1):
    """
    Returns the appropriate model based on the specified component type.
    
    Parameters:
    Comp (str): Component type.
    dropout (float): Dropout rate.
    multiplier (int): Multiplier for hidden units.
    h (int): Number of hidden units.
    d (int): Dimension of the input features.
    NumFeats (int): Number of input features.
    numLayers (int): Number of layers.
    bidirectional (bool): Whether to use bidirectional LSTM.
    config (dict): Configuration dictionary.
    input_size_perchannel (int, optional): Size of the input for each channel. Defaults to 1.
    
    Returns:
    nn.Module: The appropriate model.
    """
    if (Comp == 'LSTM') or (Comp == 'BiLSTM'):
        model = RNNModel(dropout, h, d, NumFeats, numLayers=config['NumLayers'], bidirectional=bidirectional)
    elif Comp == 'Transformer':
        model = TransformerModel(dropout, multiplier*h, d, NumFeats, numLayers=config['NumLayers'], NumHeads=config['NumHeads'], bidirectional=bidirectional)
    elif Comp == 'CNNAttn':
        model = CNNAttnModel(dropout, multiplier*h, d, NumFeats, kernelsize=config['CNNKernelSize'], numLayers=config['NumLayers'], bidirectional=bidirectional)
    elif Comp == 'CNNLSTM':
        model = CNNLSTMModel(dropout, h, d, NumFeats, kernelsize=config['CNNKernelSize'], numLayers=config['NumLayers'], bidirectional=bidirectional)
    elif Comp == 'FCN':
        model = FCN(dropout, h, d, NumFeats, kernelsizemult=config['FCNKernelMult'])
    elif Comp == 'FCN_perchannel':
        model = FCN_perchannel(dropout, h, d, NumFeats, kernelsizemult=config['FCNKernelMult'], input_size_perchannel=input_size_perchannel)
    elif Comp == 'Transformer_perchannel':
        model = Transformer_perchannel(dropout, multiplier*h, d, NumFeats, numLayers=config['NumLayers'], NumHeads=config['NumHeads'], bidirectional=bidirectional)
    elif Comp == 'CNNAttn_perchannel':
        model = CNNAttn_perchannel(dropout, multiplier*h, d, NumFeats, kernelsize=config['CNNKernelSize'], numLayers=config['NumLayers'], bidirectional=bidirectional)
    elif Comp == 'TLSTM':
        model = TA_LSTM(NumFeats, h)
    else:
        raise ValueError('Comp value provided not found: ' + Comp)
    return model

class Modelfreq(nn.Module):
    """
    Modelfreq is a neural network module that combines multiple frequency models.
    """
    def __init__(self, dropout, hs, d, numfreqs, NumFeats, Fusion, Comp='LSTM', classification=True, bidirectional=True, UseExtraLinear=False, config=None, regularized=True):
        """
        Initializes the Modelfreq module.
        
        Parameters:
        dropout (float): Dropout rate.
        hs (list): List of hidden units for each frequency model.
        d (int): Dimension of the input features.
        numfreqs (int): Number of frequency models.
        NumFeats (list): List of input features for each frequency model.
        Fusion (nn.Module): Fusion module to combine frequency models.
        Comp (str, optional): Component type. Defaults to 'LSTM'.
        classification (bool, optional): Whether the task is classification. Defaults to True.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        UseExtraLinear (bool, optional): Whether to use extra linear layer in fusion. Defaults to False.
        config (dict, optional): Configuration dictionary. Defaults to None.
        regularized (bool, optional): Whether to apply regularization. Defaults to True.
        """
        super(Modelfreq, self).__init__()
        self.numfreqs = numfreqs
        j = 0
        self.modelindx = {}
        Ms = []
        for i in range(numfreqs):
            multiplier = 2 if bidirectional else 1
            self.modelindx[i] = j
            if hs[i] != 0:
                Ms.append(get_Model(Comp, dropout, multiplier, hs[i], d, NumFeats[i], config['NumLayers'], bidirectional, config))
                # if (Comp == 'LSTM') or (Comp == 'BiLSTM'):
                #     Ms.append(RNNModel(dropout, hs[i], d, NumFeats[i], numLayers=config['NumLayers'], bidirectional=bidirectional))
                # elif Comp == 'Transformer':
                #     Ms.append(TransformerModel(dropout, multiplier*hs[i], d, NumFeats[i], numLayers=config['NumLayers'], NumHeads=config['NumHeads'], bidirectional=bidirectional))
                # elif Comp == 'CNNAttn':
                #     Ms.append(CNNAttnModel(dropout, multiplier*hs[i], d, NumFeats[i], kernelsize=config['CNNKernelSize'], numLayers=config['NumLayers'], bidirectional=bidirectional))
                # elif Comp == 'CNNLSTM':
                #     Ms.append(CNNLSTMModel(dropout, hs[i], d, NumFeats[i], kernelsize=config['CNNKernelSize'], numLayers=config['NumLayers'], bidirectional=bidirectional))
                # elif Comp == 'FCN':
                #     Ms.append(FCN(dropout, hs[i], d, NumFeats[i], kernelsizemult=config['FCNKernelMult']))
                # elif Comp == 'FCN_perchannel':
                #     Ms.append(FCN_perchannel(dropout, hs[i], d, NumFeats[i], kernelsizemult=config['FCNKernelMult']))
                # elif Comp == 'Transformer_perchannel':
                #     Ms.append(Transformer_perchannel(dropout, multiplier*hs[i], d, NumFeats[i], numLayers=config['NumLayers'], NumHeads=config['NumHeads'], bidirectional=bidirectional))
                # elif Comp == 'CNNAttn_perchannel':
                #     Ms.append(CNNAttn_perchannel(dropout, multiplier*hs[i], d, NumFeats[i], kernelsize=config['CNNKernelSize'], numLayers=config['NumLayers'], bidirectional=bidirectional))
                # elif Comp == 'TLSTM':
                #     Ms.append(TA_LSTM(NumFeats[i], hs[i]))
                # else:
                #     raise ValueError('Comp value provided not found: ' + Comp)
                j += 1
        print(self.modelindx, j)
        self.freqmodels = nn.ModuleList(Ms)
        self.hs = hs
        NumClasses = 1
        if 'NumClasses' in config:
            NumClasses = config['NumClasses']
        self.fusion = Fusion(hs, d, NumClasses, bidirectional=bidirectional, useExtralin=UseExtraLinear)
        self.finalAct = None
        if NumClasses == 1 and classification:
            print('Using Sigmoid')
            self.finalAct = nn.Sigmoid()
        
        self.FeatIdxs = None
        self.regularized = regularized
        if 'times' in config:
            self.times = config['times']
        else:
            self.times = None
    
    def forward(self, data):
        """
        Defines the forward pass of the Modelfreq module.
        
        Parameters:
        data (list): List of input tensors for each frequency model.
        
        Returns:
        Tuple[Tensor, list]: Output tensor and list of component outputs.
        """
        out = []
        compouts = []
        for i in range(self.numfreqs):
            if self.hs[i] != 0:
                if self.FeatIdxs is not None:
                    if self.regularized:
                        data[i] = data[i][:, :, self.FeatIdxs[i]]
                    else:
                        data[i] = np.array(data[i])[self.FeatIdxs[i].numpy()]
                if self.times:
                    o = self.freqmodels[self.modelindx[i]](data[i], self.times[i])
                    # print('o shape', o.shape)
                else:
                    o = self.freqmodels[self.modelindx[i]](data[i])
                out.append(o)
        op, compouts = self.fusion(out)
        if self.finalAct:
            op = self.finalAct(op)
            for i in range(len(compouts)):
                compouts[i] = self.finalAct(compouts[i])
#         op = self.LastLinear(op)
#         op = op.squeeze()
        self.wandblog()
        return op, compouts
    
    def wandblog(self):
        """
        Logs mask weights to Weights & Biases.
        """
        WeightDict = {}
        if self.FeatIdxs is not None:
            for i, Ms in enumerate(self.FeatIdxs):
                for j, m in enumerate(Ms):
                    WeightDict['FeatIndxs_' + str(i) + '_' + str(j)] = m.int()
        wandb.log(WeightDict, commit=False)

class model_FFT(nn.Module):
    """
    model_FFT is a neural network module that combines FFT and base models.
    """
    def __init__(self, dropout, hs, d, numfreqs, NumFeats, Fusion, Comp='LSTM', classification=True, bidirectional=True, UseExtraLinear=False, config=None, regularized=True):
        """
        Initializes the model_FFT module.
        
        Parameters:
        dropout (float): Dropout rate.
        hs (list): List of hidden units for each frequency model.
        d (int): Dimension of the input features.
        numfreqs (int): Number of frequency models.
        NumFeats (list): List of input features for each frequency model.
        Fusion (nn.Module): Fusion module to combine frequency models.
        Comp (str, optional): Component type. Defaults to 'LSTM'.
        classification (bool, optional): Whether the task is classification. Defaults to True.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        UseExtraLinear (bool, optional): Whether to use extra linear layer in fusion. Defaults to False.
        config (dict, optional): Configuration dictionary. Defaults to None.
        regularized (bool, optional): Whether to apply regularization. Defaults to True.
        """
        super(model_FFT, self).__init__()
        self.regularized = regularized
        multiplier = 2 if bidirectional else 1
        NumClasses = 1
        if 'NumClasses' in config:
            NumClasses = config['NumClasses']
        if hs[-1] != 0:
            self.basemodel = get_Model(Comp, dropout, multiplier, hs[-1], d, NumFeats[-1], config['NumLayers'], bidirectional, config)
            numF = NumFeats[-1]
            if regularized:
                numF = NumFeats[-1] * 2
            self.fftmodel = get_Model(Comp, dropout, multiplier, hs[-1], d, numF, config['NumLayers'], bidirectional, config, input_size_perchannel=2)
        self.fc1 = nn.Linear(hs[-1] * 2, d)
        self.firstAct = nn.ReLU()
        self.fc2 = nn.Linear(d, NumClasses)
        self.finalAct = None
        if NumClasses == 1 and classification:
            print('Using Sigmoid')
            self.finalAct = nn.Sigmoid()

    def forward(self, data):
        """
        Defines the forward pass of the model_FFT module.
        
        Parameters:
        data (list): List of input tensors for each frequency model.
        
        Returns:
        Tuple[Tensor, list]: Output tensor and list of component outputs.
        """
        x = data[-1]
        if self.regularized:
            print(f'x.shape {x.shape}')
            fft_out = fft(x, dim=-2)
            print(f'fft_out.shape {fft_out.shape}')
            # fft_out is of shape (batch_size, sequence_length, input_dim)
            real_part = torch.real(fft_out)
            imag_part = torch.imag(fft_out)
            # real_part and imag_part are of shape (batch_size, sequence_length, input_dim)
            print(f'real_part.shape {real_part.shape}')
            fft_in = torch.cat((real_part, imag_part), dim=-1)
            print(f'fft_in.shape {fft_in.shape}')
        else:
            print(f'x shapes {[z.shape for z in x]}')
            fft_outs = [fft(z, dim=-1) for z in x]
            fft_in = [torch.stack((torch.real(fft_out), torch.imag(fft_out)), dim=1) for fft_out in fft_outs]
            print(f'fft_in shapes {[z.shape for z in fft_in]}')
        op = self.basemodel(x)
        fft_out = self.fftmodel(fft_in)
        print(f'op.shape {op.shape}, fft_out.shape {fft_out.shape}')
        concat_out = torch.cat((op, fft_out), dim=-1)
        print(f'concat_out.shape {concat_out.shape}')
        fc1_out = self.firstAct(self.fc1(concat_out))
        print(f'fc1_out.shape {fc1_out.shape}')
        fc2_out = self.fc2(fc1_out)
        if self.finalAct:
            fc2_out = self.finalAct(fc2_out)
        print(f'fc2_out.shape {fc2_out.shape}')
        return fc2_out, []

class Modelfreq_featMasks(nn.Module):
    """
    Modelfreq_featMasks is a neural network module that combines multiple frequency models with feature masks.
    """
    def __init__(self, dropout, hs, d, numfreqs, NumFeats, Fusion, Comp='LSTM', classification=True, bidirectional=True, UseExtraLinear=False, config=None, MaskWeightInit=0.5, regularized=True):
        """
        Initializes the Modelfreq_featMasks module.
        
        Parameters:
        dropout (float): Dropout rate.
        hs (list): List of hidden units for each frequency model.
        d (int): Dimension of the input features.
        numfreqs (int): Number of frequency models.
        NumFeats (list): List of input features for each frequency model.
        Fusion (nn.Module): Fusion module to combine frequency models.
        Comp (str, optional): Component type. Defaults to 'LSTM'.
        classification (bool, optional): Whether the task is classification. Defaults to True.
        bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
        UseExtraLinear (bool, optional): Whether to use extra linear layer in fusion. Defaults to False.
        config (dict, optional): Configuration dictionary. Defaults to None.
        MaskWeightInit (float, optional): Initial weight for feature masks. Defaults to 0.5.
        regularized (bool, optional): Whether to apply regularization. Defaults to True.
        """
        super(Modelfreq_featMasks, self).__init__()
        self.numfreqs = numfreqs
        j = 0
        self.modelindx = {}
        FeatMaskWeights = []
        Ms = []
        multiplier = 2 if bidirectional else 1
        self.FeatIdxs = []
        for i in range(numfreqs):
            self.modelindx[i] = j
            self.FeatIdxs.append(torch.ones(NumFeats[i]).bool())
            if hs[i] != 0:
                Ms.append(get_Model(Comp, dropout, multiplier, hs[i], d, NumFeats[i], config['NumLayers'], bidirectional, config))
                FeatMaskWeights.append(nn.Parameter(torch.tensor([MaskWeightInit for _ in range(NumFeats[i])])))
                j += 1
        print(self.modelindx, j)
        self.FeatMaskWeights = nn.ParameterList(FeatMaskWeights)
        self.freqmodels = nn.ModuleList(Ms)
        self.hs = hs
        NumClasses = 1
        if 'NumClasses' in config:
            NumClasses = config['NumClasses']
            print('NumClasses', NumClasses)
        self.fusion = Fusion(hs, d, NumClasses, bidirectional=bidirectional, useExtralin=UseExtraLinear)
        self.activation = nn.ReLU()
        self.finalAct = None
        self.regularized = regularized
        if 'times' in config:
            self.times = config['times']
        else:
            self.times = None
        if NumClasses == 1 and classification:
            print('Using Sigmoid')
            self.finalAct = nn.Sigmoid()
    
    def forward(self, data):
        """
        Defines the forward pass of the Modelfreq_featMasks module.
        
        Parameters:
        data (list): List of input tensors for each frequency model.
        
        Returns:
        Tuple[Tensor, list]: Output tensor and list of component outputs.
        """
        out = []
        compouts = []
        FeatMasks = []
        for i in range(self.numfreqs):
            if self.hs[i] != 0:
                if self.FeatIdxs is not None:
                    if self.regularized:
                        data[i] = data[i][:, :, self.FeatIdxs[i]]
                    else:
                        data[i] = np.array(data[i])[self.FeatIdxs[i].numpy()]
                FeatMask = self.activation(self.FeatMaskWeights[self.modelindx[i]])
                if self.regularized:
                    D = data[i] * FeatMask
                else:
                    D = [data[i][j] * FeatMask[j] for j in range(len(data[i]))]
                if self.times:
                    o = self.freqmodels[self.modelindx[i]](D, self.times[i])
                else:
                    o = self.freqmodels[self.modelindx[i]](D)
                # o = self.freqmodels[self.modelindx[i]](D)
                out.append(o)
                FeatMasks.append(FeatMask)
        op, compouts = self.fusion(out)
        if self.finalAct:
            op = self.finalAct(op)
            for i in range(len(compouts)):
                compouts[i] = self.finalAct(compouts[i])
        self.FeatMasks = FeatMasks
        self.wandblog(FeatMasks)
        return op, compouts
    
    def l1_norm(self):
        """
        Computes the L1 norm of the feature masks.
        
        Returns:
        float: L1 norm of the feature masks.
        """
        norm = 0.0
        for fm in self.FeatMasks:
            norm += torch.norm(fm, 1)
        return norm
    
    def wandblog(self, Masks):
        """
        Logs feature masks and weights to Weights & Biases.
        
        Parameters:
        Masks (list): List of feature masks.
        """
        WeightDict = {}
        for i, Ms in enumerate(Masks):
            for j, m in enumerate(Ms):
                WeightDict['FeatMask_' + str(i) + '_' + str(j)] = m
                WeightDict['FeatWeight_' + str(i) + '_' + str(j)] = self.FeatMaskWeights[i][j]
                WeightDict['ModelNorm_' + str(i)] = torch.norm(Ms, 1)
                WeightDict['FeatIndxs_' + str(i) + '_' + str(j)] = self.FeatIdxs[i][j].int()
        wandb.log(WeightDict, commit=False)


def getFreqModel(config):
    """
    Returns the appropriate frequency model based on the specified configuration.
    
    Parameters:
    config (dict): Configuration dictionary.
    
    Returns:
    nn.Module: The appropriate frequency model.
    
    Raises:
    ValueError: If the model type is not recognized.
    """
    regularized = True
    if 'regularized' in config:
        regularized = config['regularized']
    
    if config['model'] == 'Modelfreq':
        modelfreq = Modelfreq(
            config['dropout'], 
            hs=config['hs'], 
            d=config['d'], 
            Comp=config['Comp'], 
            numfreqs=config['NumComps'], 
            NumFeats=config['NumFeats'], 
            classification=config['Classification'], 
            Fusion=config['Fusion'], 
            UseExtraLinear=config['UseExtraLinear'], 
            bidirectional=config['bidirectional'], 
            config=config, 
            regularized=regularized
        )
    elif config['model'] == 'Modelfreq_featMasks':
        modelfreq = Modelfreq_featMasks(
            config['dropout'], 
            hs=config['hs'], 
            d=config['d'], 
            Comp=config['Comp'], 
            numfreqs=config['NumComps'], 
            NumFeats=config['NumFeats'], 
            classification=config['Classification'], 
            Fusion=config['Fusion'], 
            UseExtraLinear=config['UseExtraLinear'], 
            bidirectional=config['bidirectional'], 
            config=config, 
            MaskWeightInit=config['InitMaskW'], 
            regularized=regularized
        )
    elif config['model'] == 'model_FFT':
        modelfreq = model_FFT(
            config['dropout'], 
            hs=config['hs'], 
            d=config['d'], 
            Comp=config['Comp'], 
            numfreqs=config['NumComps'], 
            NumFeats=config['NumFeats'], 
            classification=config['Classification'], 
            Fusion=config['Fusion'], 
            UseExtraLinear=config['UseExtraLinear'], 
            bidirectional=config['bidirectional'], 
            config=config, 
            regularized=regularized
        )
    else:
        raise ValueError('Model type not found: ' + config['model'])
    
    return modelfreq