'''
Author: Yu Zhang
Date: 2025/4/14
'''
from utils import *
from basemodel import *

class VMP(nn.Module):
    def __init__(self, args):
        super(VMP, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.using_cuda = args.using_cuda
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)
        self.cell = LSTMCell(args.input_embed_size, args.rnn_size)


        if self.args.passing_time>=1:
            self.gcn1 = message_passing(args, self.args.rela_embed_size, args.rnn_size)
        if self.args.passing_time>=2:
            self.gcn2 = message_passing(args, self.args.rela_embed_size, args.rnn_size)
        if self.args.passing_time==3:
            self.gcn3 = message_passing(args, self.args.rela_embed_size, args.rnn_size)

        self.outputLayer = nn.Linear(args.rnn_size, args.output_size)
        self.dropout = nn.Dropout(args.dropratio)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant(self.inputLayer.bias, 0.0)
        nn.init.normal(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform(self.cell.weight_ih)
        nn.init.orthogonal(self.cell.weight_hh, gain=0.001)

        nn.init.constant(self.cell.bias_ih, 0.0)
        nn.init.constant(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant(self.cell.bias_ih[n // 4:n // 2], 1.0)

        

        
        nn.init.constant(self.outputLayer.bias, 0.0)
        nn.init.normal(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs,iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum=inputs
        num_Ped = nodes_norm.shape[1]
        
        # 初始化输出和LSTM状态
        outputs=torch.zeros(nodes_norm.shape[0],num_Ped, self.args.output_size)
        # 预测轨迹容器
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)

        value1_sum=0
        value2_sum=0
        value3_sum=0
        
        value1 = 0
        value2 = 0
        value3 = 0
        
        # 启用GPU
        if self.using_cuda:
            outputs=outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            
        # For each frame in the sequence 逐帧处理序列
        for framenum in range(self.args.seq_length-1):
             # 测试模式且超过观察期时，使用预测值代替真实位置
            if framenum >= self.args.obs_length and iftest:
                # 从最后一帧观察帧获取有效船舶索引
                node_index = seq_list[self.args.obs_length - 1] > 0
                # 使用上一帧的预测位置作为当前位置
                nodes_current = outputs[framenum - 1, node_index].clone()
                
                # 计算绝对坐标（预测值 + 偏移量）
                nodes_abs=shift_value[framenum,node_index]+nodes_current
                # 为相对位置计算做准备（复制张量）
                nodes_abs=nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index=nodes_abs.transpose(0,1)-nodes_abs
            else:
                # 训练/观察阶段使用真实坐标
                node_index=seq_list[framenum]>0
                nodes_current = nodes_norm[framenum,node_index]
                # 准备邻居信息
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0,1)-corr
                nei_num_index=nei_num[framenum,node_index]
            
            # 获取当前有效船舶的LSTM状态
            hidden_states_current=hidden_states[node_index]
            cell_states_current=cell_states[node_index]
            # print('nodes_current',nodes_current)
            # print('corr_index',corr_index)
            # print('nei_index',nei_index)
            # print('nei_num_index',nei_num_index)
            
            # 输入嵌入层（坐标 -> 特征向量）
            input_embedded = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))

            lstm_state = self.cell.forward(input_embedded, (hidden_states_current,cell_states_current))

            for p in range(self.args.passing_time ):
                if p==0:
                    # 第一层
                    lstm_state, look = self.gcn1.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn1.W_nei)
                    value1, value2, value3 = look
                if p==1:
                    # 第二层
                    lstm_state, look = self.gcn2.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn2.W_nei)
                    value1, value2, value3 = look
                if p==2:
                    # 第三层
                    lstm_state, look = self.gcn3.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn3.W_nei)
                    value1, value2, value3 = look
            
            
            # 更新LSTM状态
            _, hidden_states_current, cell_states_current = lstm_state

            value1_sum+=value1
            value2_sum+=value2
            value3_sum+=value3

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum,node_index]=outputs_current
            hidden_states[node_index]=hidden_states_current
            cell_states[node_index] = cell_states_current

        return outputs, hidden_states, cell_states,(value1_sum/self.args.seq_length,value2_sum/self.args.seq_length,value3_sum/self.args.seq_length)



