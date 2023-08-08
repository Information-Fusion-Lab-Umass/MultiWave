import torch
import torch.nn as nn
import torch.nn.functional as F

class TLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, train):
        super(TLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.train = train

        self.Wi = nn.Linear(input_dim, hidden_dim)
        self.Ui = nn.Linear(hidden_dim, hidden_dim)
        self.bi = nn.Linear(hidden_dim, hidden_dim)

        self.Wf = nn.Linear(input_dim, hidden_dim)
        self.Uf = nn.Linear(hidden_dim, hidden_dim)
        self.bf = nn.Linear(hidden_dim, hidden_dim)

        self.Wog = nn.Linear(input_dim, hidden_dim)
        self.Uog = nn.Linear(hidden_dim, hidden_dim)
        self.bog = nn.Linear(hidden_dim, hidden_dim)

        self.Wc = nn.Linear(input_dim, hidden_dim)
        self.Uc = nn.Linear(hidden_dim, hidden_dim)
        self.bc = nn.Linear(hidden_dim, hidden_dim)

        self.W_decomp = nn.Linear(hidden_dim, hidden_dim)
        self.b_decomp = nn.Linear(hidden_dim, hidden_dim)

        self.Wo = nn.Linear(hidden_dim, fc_dim)
        self.bo = nn.Linear(fc_dim, fc_dim)

        self.W_softmax = nn.Linear(fc_dim, output_dim)
        self.b_softmax = nn.Linear(output_dim, output_dim)

    def forward(self, input, labels, time, keep_prob, hidden, prev_cell):
        # time decay
        T = self.map_elapse_time(time)

        # Decompose the previous cell if there is a elapse time
        C_ST = F.tanh(self.W_decomp(prev_cell) + self.b_decomp)
        C_ST_dis = torch.matmul(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # input gate
        i = F.sigmoid(self.Wi(input) + self.Ui(hidden) + self.bi(hidden))

        # forget gate
        f = F.sigmoid(self.Wf(input) + self.Uf(hidden) + self.bf(hidden))

        # output gate
        og = F.sigmoid(self.Wog(input) + self.Uog(hidden) + self.bog(hidden))

        # state
        state = F.tanh(self.Wc(input) + self.Uc(hidden) + self.bc(hidden))
        c = f * prev_cell + i * state

        # ct-1 decomp
        hidden = og * F.tanh(c)

        return hidden, c
