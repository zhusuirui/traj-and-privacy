# from sam_cells import SAM_LSTMCell,SAM_GRUCell
from geo_rnns.sam_cells import SAM_LSTMCell,SAM_GRUCell
from torch.nn import Module
from tools import config

import torch.autograd as autograd
import torch.nn.functional as F
import torch
device = 'cuda' if torch.cuda.is_available() else "cpu"

class RNNEncoder(Module):
    def __init__(self, input_size, hidden_size, grid_size, stard_LSTM= False, incell = True):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        if self.stard_LSTM:
            if config.recurrent_unit=='GRU':
                self.cell = torch.nn.GRUCell(input_size - 2, hidden_size).to(device)
            elif config.recurrent_unit=='SimpleRNN':
                self.cell = torch.nn.RNNCell(input_size - 2, hidden_size).to(device)
            else:
                self.cell = torch.nn.LSTMCell(input_size - 2, hidden_size).to(device)
        else:
            if config.recurrent_unit=='GRU':
                self.cell = SAM_GRUCell(input_size, hidden_size, grid_size, incell=incell).to(device)
            # elif config.recurrent_unit=='SimpleRNN':
            #     self.cell = SpatialRNNCell(input_size, hidden_size, grid_size, incell=incell).to(device)
            else:
                self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell).to(device)

        print (self.cell)
        print ('in cell update: {}'.format(incell))
        self.linear_s=torch.nn.Linear(2, 64).to(device)
        self.linear_t=torch.nn.Linear(2, 64).to(device)
        self.nonLeaky = torch.nn.LeakyReLU(0.1)
        # self.para = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.para1=torch.sigmoid(self.para)

        # self.cell = torch.nn.LSTMCell(input_size-2, hidden_size).to(device)
    def forward(self, inputs_a, initial_state = None):
        inputs_st, inputs_len = inputs_a
        inputs_s=inputs_st[:,:,:2]
        inputs_t=inputs_st[:,:,2:4]
        time_steps_a=inputs_st.size(1)
        inputs_grid_st=(inputs_st[:, :, 5] * 1100 +inputs_st[:, :, 4] - 2202 + 1).clamp(0, 1100 * 1100).long()
        mask_a = (inputs_grid_st != 0).unsqueeze(-2).to(device)
        inputs_s_em=self.nonLeaky(self.linear_s(inputs_s))
        inputs_t_em=self.nonLeaky(self.linear_t(inputs_t))

        attention_ss=torch.matmul(inputs_s_em, inputs_s_em.transpose(-2, -1))
        attention_ss = attention_ss.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        attention_ss = attention_ss.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        p_attention_ss = F.softmax(attention_ss, dim=-1)
        p_attention_ss = p_attention_ss.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        p_attention_ss = p_attention_ss.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        attn_ss = p_attention_ss.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size // 2)

        attention_st=torch.matmul(inputs_s_em, inputs_t_em.transpose(-2, -1))
        attention_st = attention_st.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        attention_st = attention_st.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        p_attention_st = F.softmax(attention_st, dim=-1)
        p_attention_st = p_attention_st.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        p_attention_st = p_attention_st.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        attn_st = p_attention_st.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size // 2)

        attention_ts=torch.matmul(inputs_t_em, inputs_s_em.transpose(-2, -1))
        attention_ts = attention_ts.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        attention_ts = attention_ts.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        p_attention_ts = F.softmax(attention_ts, dim=-1)
        p_attention_ts = p_attention_ts.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        p_attention_ts = p_attention_ts.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        attn_ts = p_attention_ts.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size // 2)

        attention_tt=torch.matmul(inputs_t_em, inputs_t_em.transpose(-2, -1))
        attention_tt = attention_tt.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        attention_tt = attention_tt.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        p_attention_tt = F.softmax(attention_tt, dim=-1)
        p_attention_tt = p_attention_tt.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        p_attention_tt = p_attention_tt.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        attn_tt = p_attention_tt.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size // 2)

#tt*t+ts*s, st*t+ss*s

        sum_ss = inputs_s_em.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ss).sum(dim=-2)
        sum_st = inputs_t_em.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_st).sum(dim=-2)
        sum_ts = inputs_s_em.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ts).sum(dim=-2)
        sum_tt = inputs_t_em.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_tt).sum(dim=-2)
        inputs_s_em_new = (inputs_s_em-sum_ss)+(inputs_s_em-sum_st)
        inputs_t_em_new = (inputs_t_em-sum_tt)+(inputs_t_em-sum_ts)
        inputs_st_em =0.5*inputs_s_em+0.5*inputs_t_em
        inputs_st_em_new =0.5*inputs_s_em_new+0.5*inputs_t_em_new

        inputs=torch.cat((inputs_st_em, inputs_st_em_new,inputs_st[:,:,4:6]), dim=-1)



        time_steps = inputs.size(1)
        out = None
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            out = initial_state
        else:
            out, state = initial_state

        outputs = []
        for t in range(time_steps):
            if self.stard_LSTM:
                cell_input = inputs[:, t, :][:,:-2]
            else:
                cell_input = inputs[:, t, :]
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                out = self.cell(cell_input, out)
            else:
                out, state = self.cell(cell_input, (out, state))
            outputs.append(out)
        mask_out = []
        for b, v in enumerate(inputs_len):
            mask_out.append(outputs[v-1][b,:].view(1,-1))
        return torch.cat(mask_out, dim = 0)

    def batch_grid_state_gates(self, inputs_a, initial_state = None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out, state = initial_state
        outputs = []
        gates_out_all = []
        batch_weight_ih = autograd.Variable(self.cell.weight_ih.data, requires_grad=False).to(device)
        batch_weight_hh = autograd.Variable(self.cell.weight_hh.data, requires_grad=False).to(device)
        batch_bias_ih = autograd.Variable(self.cell.bias_ih.data, requires_grad=False).to(device)
        batch_bias_hh = autograd.Variable(self.cell.bias_hh.data, requires_grad=False).to(device)
        for t in range(time_steps):
            # cell_input = inputs[:, t, :][:,:-2]
            cell_input = inputs[:, t, :]
            self.cell.update_memory(cell_input, (out, state),
                                    batch_weight_ih, batch_weight_hh,
                                    batch_bias_ih, batch_bias_hh)


class NeuTraj_Network(Module):
    def __init__(self,input_size, target_size, grid_size, batch_size, sampling_num, stard_LSTM = False, incell = True):
        super(NeuTraj_Network, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        if config.recurrent_unit=='GRU' or config.recurrent_unit=='SimpleRNN':
            self.hidden = autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                             requires_grad=False).to(device)
        else:
            self.hidden = (autograd.Variable(torch.zeros(self.batch_size*(1+self.sampling_num), self.target_size),requires_grad=False).to(device),
                      autograd.Variable(torch.zeros(self.batch_size*(1+self.sampling_num), self.target_size),requires_grad=False).to(device))
        self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM= stard_LSTM,
                              incell = incell).to(device)

    def forward(self, inputs_arrays, inputs_len_arrays):
        anchor_input = torch.Tensor(inputs_arrays[0])
        trajs_input = torch.Tensor(inputs_arrays[1])
        negative_input = torch.Tensor(inputs_arrays[2])

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        anchor_embedding = self.rnn([autograd.Variable(anchor_input,requires_grad=False).to(device), anchor_input_len], self.hidden)
        trajs_embedding = self.rnn([autograd.Variable(trajs_input,requires_grad=False).to(device), trajs_input_len], self.hidden)
        negative_embedding = self.rnn([autograd.Variable(negative_input,requires_grad=False).to(device), negative_input_len], self.hidden)

        trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
        negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        return trajs_loss, negative_loss


    def spatial_memory_update(self, inputs_arrays, inputs_len_arrays):
        batch_traj_input = torch.Tensor(inputs_arrays[3])
        batch_traj_len = inputs_len_arrays[3]
        batch_hidden = (autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size),requires_grad=False).to(device),
                        autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size),requires_grad=False).to(device))
        self.rnn.batch_grid_state_gates([autograd.Variable(batch_traj_input).to(device), batch_traj_len],batch_hidden)