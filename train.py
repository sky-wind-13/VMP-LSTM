'''
Train script
Author: YU Zhang
Date: 2025/4/14
'''
import argparse
import ast
from Processor import *

def get_parser():

    parser = argparse.ArgumentParser(
        description='States Refinement LSTM')
    parser.add_argument(
        '--using_cuda',default=True,type=ast.literal_eval) # We did not test on cpu
    # You may change these arguments (model selection and dirs)
    parser.add_argument(
        '--test_set',default=1,type=int,
        help='Set this value to 0~4 for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument(
        '--gpu', default=0,type=int,
        help='gpu id')
    # 基本的脚本路径
    parser.add_argument(
        '--base_dir',default='.',
        help='Base directory including these scrits.')
    # 存放的位置
    parser.add_argument(
        '--save_base_dir',default='./taiwan-VIFG-tt/',
        help='Directory for saving caches and models.')
    # 训练或者测试
    parser.add_argument(
        '--phase', default='train',
        help='Set this value to \'train\' or \'test\'')
    # 保存目录的名字
    parser.add_argument(
        '--train_model', default='Bi-LSTM-taiwan',
        help='Your model name')
    # 测试或者
    parser.add_argument(
        '--load_model', default=0,type=int,
        help="load model weights from this index before training or testing")
    # 预训练模型加载
    parser.add_argument(
        '--pretrain_model', default='',
        help='Your pretrained model name. Used in training second states refienemnt layer.')
    # 可以将
    parser.add_argument(
        '--pretrain_load', default=0,type=int,
        help="load pretrained model from this index. Used in training second states refienemnt layer.")
    # 没有用的参数
    parser.add_argument(
        '--model', default='models.VMP')
    ######################################

    parser.add_argument(
        '--dataset',default='shanghai')
    parser.add_argument(
        '--save_dir')
    parser.add_argument(
        '--model_dir')
    parser.add_argument(
        '--config')
    
    # 是否开启验证
    parser.add_argument(
        '--ifvalid',default=True,type=ast.literal_eval,
        help="=False,use all train set to train,"
             "=True,use train set to train and valid")
    # 这个不用改
    parser.add_argument(
        '--val_fraction',default=0.2,type=float)

    #Model parameters

    #LSTM
    # 输出大小
    parser.add_argument(
        '--output_size',default=2,type=int)
    # 输入嵌入层大小
    parser.add_argument(
        '--input_embed_size',default=32,type=int)
    # 创建的RNN大小
    parser.add_argument(
        '--rnn_size',default=64,type=int)
    # 仅在basemodel中使用，相当于中间的某一层
    parser.add_argument(
        '--hidden_dot_size',default=32,type=int)
    
    parser.add_argument(
        '--ifdropout',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--dropratio',default=0.1,type=float)
    # 初始化标准差为0.2的权重
    parser.add_argument(
        '--std_in',default=0.2,type=float)
    # 输出权重
    parser.add_argument(
        '--std_out',default=0.1,type=float)

    
    #States Refinement
    # motion gate
    parser.add_argument(
        '--ifbias_gate',default=True,type=ast.literal_eval)
    # PA 行人注意力
    parser.add_argument(
        '--WAr_ac',default='')
    # 这个也是行人注意力
    parser.add_argument(
        '--ifbias_WAr',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_size',default=2,type=int)
    
    # GCN的嵌入层大小，也就是输入的大小
    parser.add_argument(
        '--rela_embed_size', default=32,type=int)
    
    # 中间相关层的输出大小
    parser.add_argument(
        '--rela_hidden_size', default=16,type=int)
    # 中间层的数量
    parser.add_argument(
        '--rela_layers', default=1,type=int)
    # 中间层输入大小
    parser.add_argument(
        '--rela_input', default=2,type=int)
    # 中间层的dropout
    parser.add_argument(
        '--rela_drop', default=0.1,type=float)
    # 中间层的激活函数
    parser.add_argument(
        '--rela_ac', default='relu')
    # 可以作为中间层是否使用的标志
    parser.add_argument(
        '--ifbias_rel',default=True,type=ast.literal_eval)

    # motion gate和message passing的隐藏层大小
    parser.add_argument(
        '--nei_hidden_size', default=64,type=int)
    # 下面三个参数同上
    parser.add_argument(
        '--nei_layers', default=1,type=int)
    parser.add_argument(
        '--nei_drop', default=0,type=int)
    parser.add_argument(
        '--nei_ac', default='')
    
    # 是否初始化网络偏置
    parser.add_argument(
        '--ifbias_nei',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--mp_ac', default='')
    # 名称为nei的标准化
    parser.add_argument(
        '--nei_std',default=0.01,type=float)
    # 中间层nei的标准化
    parser.add_argument(
        '--rela_std',default=0.3,type=float)
    parser.add_argument(
        '--WAq_std',default=0.05,type=float)
    parser.add_argument(
        '--passing_time',default=0,type=int,
        help='States refinement layers. We did not test on training multiple SR layers in end2end mode')

    #Social LSTM
    parser.add_argument(
        '--grid_size',default=4,type=int)
    parser.add_argument(
        '--nei_thred_slstm',default=2,type=int)

    #Perprocess
    parser.add_argument(
        '--seq_length',default=144,type=int)
    parser.add_argument(
        '--obs_length',default=96,type=int)
    parser.add_argument(
        '--pred_length',default=48,type=int)
    parser.add_argument(
        '--batch_around_ped',default=128,type=int)
    parser.add_argument(
        '--batch_size',default=8,type=int)
    parser.add_argument(
        '--val_batch_size',default=8,type=int)
    parser.add_argument(
        '--test_batch_size',default=4,type=int)
    parser.add_argument(
        '--show_step',default=40,type=int)
    parser.add_argument(
        '--start_test',default=0,type=int)
    parser.add_argument(
        '--num_epochs',default=150,type=int)
    parser.add_argument(
        '--ifshow_detail',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--ifdebug',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--ifsave_results', default=False,type=ast.literal_eval)
    # 随机裁剪
    parser.add_argument(
        '--randomRotate',default=False,type=ast.literal_eval,
        help="=True:random rotation of each trajectory fragment")
    # NS 邻居数量
    parser.add_argument(
        '--neighbor_thred',default=0.04,type=int)
    parser.add_argument(
        '--learning_rate',default=0.002,type=float)
    parser.add_argument(
        '--clip',default=1,type=int)
    return parser
def load_arg(p):
    # save arg
    if  os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False
def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)
if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    p.save_dir=p.save_base_dir+str(p.test_set)+'/'
    p.model_dir=p.save_base_dir+str(p.test_set)+'/'+p.train_model+'/'
    p.config=p.model_dir+'/config_'+p.phase+'.yaml'

    if not load_arg(p):
        save_arg(p)
    args = load_arg(p)
    torch.cuda.set_device(args.gpu)
    processor = Processor(args)
    if args.phase=='test':
        processor.playtest()
    else:
        processor.playtrain()