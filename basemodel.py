'''
Author: Yu Zhang
Date: 2025/4/14
'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter
class MakeMLP(nn.Module):
    def __init__(self,args,layer_num,layer_name,input_num,hunit_num,output_num,active_fun,drop_ratio,ifbias,iflastac=True):
        super(MakeMLP, self).__init__()
        layers=[]
        self.args=args
        if iflastac:
            lastac=active_fun
            lastdrop=drop_ratio
        else:
            lastac=''
            lastdrop=0
        if layer_num>1:
            self.addLayer(layers, input_num, hunit_num, ifbias, active_fun, drop_ratio)
            for i in range(layer_num-2):
                self.addLayer(layers, hunit_num, hunit_num, ifbias, active_fun, drop_ratio)
            self.addLayer(layers, hunit_num, output_num, ifbias, lastac, lastdrop)
        else:
            self.addLayer(layers, input_num, output_num, ifbias, lastac, lastdrop)
        self.MLP=nn.Sequential(*layers)
        if layer_name=='rel':
            self.MLP.apply(self.init_weights_rel)
        elif layer_name=='nei':
            self.MLP.apply(self.init_weights_nei)
        elif layer_name=='attR':
            self.MLP.apply(self.init_weights_attr)
        elif layer_name=='ngate':
            self.MLP.apply(self.init_weights_ngate)

    def addLayer(self,layers,input_num,output_num,ifbias,active_fun,drop_ratio):
        layers.append(nn.Linear(input_num, output_num, bias=ifbias))
        if active_fun == 'sig':
            Active_fun = nn.Sigmoid
            layers.append(Active_fun())
        elif active_fun== 'relu':
            Active_fun = nn.ReLU
            layers.append(Active_fun())
        elif active_fun == 'lrelu':
            Active_fun = nn.LeakyReLU
            layers.append(Active_fun(0.1))
        elif active_fun == 'tanh':
            Active_fun = nn.Tanh
            layers.append(Active_fun())
        if drop_ratio!=0:
            layers.append(nn.Dropout(drop_ratio))
        return layers
    def init_weights(self,m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform(m.weight)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass

    def init_weights_ngate(self,m):
        if type(m)==nn.Linear:
            nn.init.normal(m.weight, std=0.005)

            if self.args.ifbias_gate:
                nn.init.constant(m.bias,0)
    def init_weights_nei(self,m):
        if type(m)==nn.Linear:

            nn.init.orthogonal(m.weight,gain=self.args.nei_std)
            if self.args.ifbias_nei:
                nn.init.constant(m.bias,0)

    def init_weights_attr(self,m):
        if type(m)==nn.Linear:
            #nn.init.normal(m.weight,mean=0,std=self.args.WAq_std)
            #nn.init.xavier_uniform(m.weight)
            nn.init.normal(m.weight,std=self.args.WAq_std)
            try:
                nn.init.constant(m.bias,0)
            except:
                pass

    def init_weights_rel(self,m):
        if type(m)==nn.Linear:
            nn.init.normal_(m.weight,mean=0,std=self.args.rela_std)
            #nn.init.xavier_uniform(m.weight)
            #m.weight.data+=0.1
            if self.args.ifbias_nei:
                nn.init.constant(m.bias,0)

class LSTMCell(RNNCellBase):
    '''
    Copied from torch.nn
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__(input_size, hidden_size,bias=True,num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, input, hx,update_mode=''):

        hx, cx = hx
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate_ )

        cy = forgetgate * cx +ingate * cellgate
        hy = outgate * F.tanh(cy)

        return outgate,hy, cy

class message_passing(nn.Module):
    def __init__(self,args,r_embed_size,output_size):
        super(GCN, self).__init__()
        self.args=args
        self.relu=nn.ReLU()
        self.R=r_embed_size
        self.D=output_size

        self.D1 = self.args.hidden_dot_size

        # VIFG
        self.ngate = MakeMLP(self.args, 1, 'ngate', self.R+self.D + self.D, self.args.nei_hidden_size,
                                 self.D, 'sig', self.args.nei_drop, ifbias=self.args.ifbias_gate)

        # Relative spatial embedding layer
        self.relativeLayer = MakeMLP(self.args, self.args.rela_layers, 'rel', self.args.rela_input,
                                     self.args.rela_hidden_size,
                                     self.R, self.args.rela_ac, self.args.rela_drop, ifbias=True, iflastac=True)
        # Message passing transform
        self.W_nei = MakeMLP(self.args, self.args.nei_layers, 'nei', self.D, self.args.nei_hidden_size,
                         self.D, self.args.nei_ac, self.args.nei_drop, ifbias=self.args.ifbias_nei,iflastac=True)

        tmp=self.R+self.D*2

        # VA 船舶注意力
        self.WAr = MakeMLP(self.args,1,'attR',tmp,self.D1,1, self.args.WAr_ac, drop_ratio=0.3, ifbias=self.args.ifbias_WAr)

        #not used
        # self.WAr1 = MakeMLP(self.args,1,'attR',tmp,self.D1,self.args.hidden_dot_size, '', drop_ratio=0, ifbias=False)
        # self.WAr2 = MakeMLP(self.args, 1, 'attR2', self.D1, self.args.hidden_dot_size, 1, '', drop_ratio=0,ifbias=False)


    def forward(self, corr_index,nei_index,nei_num,lstm_state,W):
        '''
        States Refinement process.
        Params:
            corr_index: relative coords of each pedestrian pair
            nei_index: neighbor exsists flag
            nei_num: neighbor number
            lstm_state: output states of LSTM cell
            W: message passing weight, namely self.W_nei when train one SR layer
        Return:
            Refined states
            Tracked variable
        '''
        outgate, self_h, self_c = lstm_state

        # If you want to track some variables
        value1,value2,value3=torch.zeros(1),torch.zeros(1),torch.zeros(1)

        self.N = corr_index.shape[0]
        nei_inputs = self_h.repeat(self.N, 1)

        nei_index_t = nei_index.view((-1))

        # corr_t=corr_index.view((self.N * self.N, -1))
        corr_t = corr_index.reshape((self.N * self.N, -1))

        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch 忽略没有领域节点的batch
            return lstm_state, (0, 0, 0),(0,0)

        r_t = self.relativeLayer.MLP(corr_t[nei_index_t > 0])
        inputs_part = nei_inputs[nei_index_t > 0]
        hi_t = nei_inputs.view((self.N, self.N, self.D)).permute(1, 0, 2).contiguous().view(-1, self.D)

        tmp = torch.cat((r_t, hi_t[nei_index_t > 0],nei_inputs[nei_index_t > 0]), 1)
        # print('tmp',tmp)

        # VIFG
        nGate = self.ngate.MLP(tmp)

        # Vessel Attention
        Pos_t = torch.full((self.N * self.N,1), 0, device=torch.device("cuda")).view(-1)
        tt = self.WAr.MLP(torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)).view((-1))

        # Pos_t[nei_index_t > 0] = tt
        Pos_t[nei_index_t > 0] = tt.long()
        Pos = Pos_t.view((self.N, self.N))
        # Pos[Pos == 0] = -np.Inf

        # Pos[Pos == 0] = float('-inf')
        Pos = Pos.float()
        Pos[Pos == 0] = float('-inf')
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)
        # print('Pos_t',Pos_t)

        # Message Passing
        H = torch.full((self.N * self.N, self.D), 0, device=torch.device("cuda"))
        # H[nei_index_t > 0] = inputs_part * nGate
        H[nei_index_t > 0] = (inputs_part * nGate).long()
        # print('self.N',self.N)
        # print('H',H[nei_index_t > 0])
        # with open('VIFG.txt', 'w') as file:
        #     # 遍历张量中的每个元素
        #     for element in H:
        #         # 将每个元素转换为字符串并写入文件，后面跟一个换行符
        #         file.write(str(element) + '\n')
        # H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.D, 1).transpose(0, 1)
        # 加权聚合
        H[nei_index_t > 0] = (H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.D, 1).transpose(0, 1)).long()
        H = H.view(self.N, self.N, -1)
        # H_sum = W.MLP(torch.sum(H, 1))
        H_sum = W.MLP(torch.sum(H.float(), 1))
        print('H_sum',H_sum)

        # Update Cell states
        C = H_sum + self_c
        # print('C',C)
        H = outgate * F.tanh(C)

        if self.args.ifdebug:
            value1 = torch.norm(H_sum[nei_num > 0]*self.args.nei_ratio ) / torch.norm(self_c[nei_num > 0])
            return (outgate, H, C), (value1.item(), value2.item(),value3.item())
        else:
            return (outgate, H, C), (0, 0, 0)







class BiLSTMCell(nn.Module):  # 继承自nn.Module
    def __init__(self, input_size, hidden_size):
        super(BiLSTMCell, self).__init__()  # 调用父类的构造函数
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 为正向和反向LSTM分别创建权重和偏置
        self.weight_ih_forward = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_forward = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_forward = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.weight_ih_backward = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_backward = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_backward = nn.Parameter(torch.Tensor(4 * hidden_size))

        # 初始化参数
        self.init_parameters()

    def init_parameters(self):
        # 使用xavier_uniform初始化权重
        nn.init.xavier_uniform_(self.weight_ih_forward)
        nn.init.xavier_uniform_(self.weight_hh_forward)
        nn.init.xavier_uniform_(self.weight_ih_backward)
        nn.init.xavier_uniform_(self.weight_hh_backward)

        # 使用uniform初始化偏置
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for bias in [self.bias_forward, self.bias_backward]:
            nn.init.uniform_(bias, -stdv, stdv)

    def forward(self, input, hx):
        # 正向LSTM单元
        output_forward, (hy_forward, cy_forward) = self.lstm_cell(input, hx[0])

        # 反向LSTM单元
        reversed_input = input.flip(0)  # 反向序列
        output_backward, (hy_backward, cy_backward) = self.lstm_cell(reversed_input, hx[1])

        # 将反向序列的结果再次翻转回来
        output_backward = output_backward.flip(0)

        # 合并正向和反向的输出
        outgate = (output_forward[-1] + output_backward[-1]) / 2
        hy = (hy_forward + hy_backward) / 2
        cy = (cy_forward + cy_backward) / 2

        return outgate, hy, cy

    def lstm_cell(self, input, hx):
        # 定义LSTM单元的计算逻辑
        hx, cx = hx
        gates = F.linear(input, self.weight_ih_forward, self.bias_forward) + F.linear(hx, self.weight_hh_forward, None)
        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate_)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * F.tanh(cy)

        return hy, cy
    
    
    
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        hx, cx = hx
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        resetgate, updategate, newgate = gates.chunk(3, 1)

        resetgate = torch.sigmoid(resetgate)
        updategate = torch.sigmoid(updategate)
        newgate = torch.tanh(newgate + resetgate * hx)

        hy = (1 - updategate) * hx + updategate * newgate

        return updategate, hy, cx  # GRU 不需要 cell state


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRU, self).__init__()
        self.gru_forward = GRUCell(input_size, hidden_size)
        self.gru_backward = GRUCell(input_size, hidden_size)

    def forward(self, input_seq):
        seq_len, batch_size, _ = input_seq.size()
        h_forward = torch.zeros(batch_size, self.gru_forward.hidden_size).to(input_seq.device)
        h_backward = torch.zeros(batch_size, self.gru_backward.hidden_size).to(input_seq.device)

        outputs_forward = []
        outputs_backward = []

        # 正向传播
        for t in range(seq_len):
            updategate, hy, _ = self.gru_forward(input_seq[t], h_forward)
            h_forward = hy
            outputs_forward.append((updategate, hy, _))

        # 反向传播
        for t in range(seq_len - 1, -1, -1):
            updategate, hy, _ = self.gru_backward(input_seq[t], h_backward)
            h_backward = hy
            outputs_backward.append((updategate, hy, _))

        outputs_backward.reverse()  # 反转以匹配正向顺序

        # 合并正向和反向输出
        outputs = []
        for (update_forward, hy_forward, _), (update_backward, hy_backward, _) in zip(outputs_forward, outputs_backward):
            combined_updategate = (update_forward + update_backward) / 2  # 或者选择其他合并方式
            outputs.append((combined_updategate, hy_forward, hy_backward))

        return torch.stack([out[0] for out in outputs]), torch.stack([out[1] for out in outputs]), None
