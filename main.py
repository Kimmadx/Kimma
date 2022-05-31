from __future__ import absolute_import
import os
import time
import argparse                        #argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
import torch
from torch.utils.data import DataLoader
from utilsgai import SG
from worksgai import train
import warnings

warnings.filterwarnings('ignore')


def main(params, model):
    if not os.path.exists('checkpoints'):  #os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
        os.mkdir('checkpoints')            #方法用于以数字权限模式创建目录。
    if not os.path.exists('records'):
        os.mkdir('records')
    dir = os.path.dirname(os.path.abspath(__file__)).replace('\\','/') + '/'       # 返回seg的绝对路径
    #1、os.path.dirname(file)返回的是.py文件的目录
    #2、os.path.abspath(file)返回的是.py文件的绝对路径（完整路径）
    #3、在命令行运行时，如果输入完整的执行的路径，则返回.py文件所在的目录，否则返回空目录

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs',      type=int,   default=300,    help='Number of epochs to train')
    parser.add_argument('--validation_step', type=int,   default=1,      help='How often to perform validation (epochs)')
    parser.add_argument('--crop_height',     type=int,   default=224,    help='Height of resized input image to network')
    parser.add_argument('--crop_width',      type=int,   default=224,    help='Width of resized input image to network')
    parser.add_argument('--batch_size',       type=int,   default=1,      help='Number of images in each batch')
    parser.add_argument('--learning_rate',   type=float, default=0.001,  help='learning rate used for train')
    parser.add_argument('--num_workers',     type=int,   default=4,      help='num of workers')
    parser.add_argument('--num_classes',     type=int,   default=3,      help='num of object classes (with void)')
    parser.add_argument('--use_gpu',         type=bool,  default=True,   help='whether to user gpu for training')
    parser.add_argument('--model_name',      type=str,   default=None,   help='path to model')
    parser.add_argument('--save_each_model', type=bool,  default=False,  help='whether to save all models')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='None')
    parser.add_argument('--test_pic_name',   type=str,   default=None,   help='which picture to test')
    parser.add_argument('--base_dir',        type=str,   default=dir,    help='project directory')
    parser.add_argument('--data_name',       type=str,   default=None,   help='data directory')
    #parser.add_argument('--pretrained_model_path', type=str, default=None, help='None')
    args = parser.parse_args(params)

    parser_ = argparse.ArgumentParser()
    parser_.add_argument('-cuda',            type=int,   default=0,      help='choose GPU ID')
    args_ = parser_.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args_.cuda)

    assert args.data_name is not None, 'Please input dataset.'
    assert args.model_name is not None, 'Please name model.'
    assert args.test_pic_name is not None, 'Please input test picture.'

    train_path = args.base_dir + f'datasets/{args.data_name}/train/image/'           # 训练集图片存放地址
    train_label_path = args.base_dir + f'datasets/{args.data_name}/train/label/'     # 训练集标签存放地址
    val_path = args.base_dir + f'datasets/{args.data_name}/val/image/'               # 验证集图片存放地址
    val_label_path = args.base_dir + f'datasets/{args.data_name}/val/label/'         # 验证集标签存放地址
    csv_path = args.base_dir + f'datasets/{args.data_name}/class_dict.csv'           # 标签分类的颜色值存放地址

    dataset_train = SG(train_path, train_label_path, csv_path)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataset_val = SG(val_path, val_label_path, csv_path)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=args.num_workers)
    #print(len(dataloader_train)*args.batch_size)
    #print(len(dataloader_val))

    miou_path = os.listdir(args.base_dir + 'checkpoints')
    #方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 ‘.’ 和’…’ 即使它在文件夹中。

    if args.data_name not in miou_path:
        os.mkdir(args.base_dir + f'checkpoints/{args.data_name}')
    #os.mkdir()创建路径中的最后一级目录，即：只创建path_03目录，而如果之前的目录不存在并且也需要创建的话，就会报错。
    #os.makedirs()创建多层目录，即：Test,path_01,path_02,path_03如果都不存在的话，会自动创建

    miou_path_list = os.listdir(args.base_dir + f'checkpoints/{args.data_name}')
    if args.model_name not in miou_path_list:
        os.mkdir(args.base_dir + f'checkpoints/{args.data_name}/{args.model_name}')

    demo_path = os.listdir(args.base_dir + 'demo')
    if args.data_name not in demo_path:
        os.mkdir(args.base_dir + f'demo/{args.data_name}')

    demo_path_list = os.listdir(args.base_dir + f'demo/{args.data_name}')
    if args.model_name not in demo_path_list:
        os.mkdir(args.base_dir + f'demo/{args.data_name}/{args.model_name}')

    records_path = os.listdir(args.base_dir + 'records')
    if args.data_name not in records_path:
        os.mkdir(args.base_dir + f'records/{args.data_name}')

    datetime = time.strftime("%Y%m%d%H%M", time.localtime())

    # 如果存在预训练好的模型，就加载它
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)

        # loading the part of network params
        pretrained_dict = torch.load(args.pretrained_model_path)
        # 给model_dict 赋予运行的那个神经网络（这里是UNet）的参数字典，形式可以查看笔记
        # 下面的代码到print，加载预训练模型,并去除需要再次训练的层，注意：需要重新训练的层的名字要和之前的不同。k 神经网络层名字.Weight或者 神经网络层名字.bias
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the model state dict
        model.load_state_dict(model_dict)
        print('Done!')
    #转换一个元组或 struct_time 表示的由 gmtime() 或 localtime() 返回的时间到由 format 参数指定的字符串。如果未提供 t ，则使用由 localtime() 返回的当前时间。
    #format 必须是一个字符串。如果 t 中的任何字段超出允许范围，则引发 ValueError 。
    #以下指令可以嵌入 format 字符串中，被 strftime() 结果中的指示字符替换：

    #%y 两位数的年份表示（00-99）
    #%Y 四位数的年份表示（000-9999）
    #%m 月份（01-12）
    #%d 月内中的一天（0-31）
    #%H 24小时制小时数（0-23）
    #%I 12小时制小时数（0-12）
    #%M 分钟数（0-59）
    #%S 秒（00-59）
    #%a 本地简化星期名称
    #%A 本地完整星期名称
    #%b 本地简化的月份名称
    #%B 本地完整的月份名称
    #%c 本地相应的日期表示和时间表示（e.g Thu Dec 10 09:54:27 2020）
    #%j 年内的一天（001-366）
    #%p 本地A.M.或P.M.的等价符
    #%U 一年中的星期数（00-53）星期天为星期的开始
    #%w 星期（0-6），星期天为星期的开始
    #%W 一年中的星期数（00-53）星期一为星期的开始
    #%x 本地相应的日期表示(e.g 12/10/20)
    #%X 本地相应的时间表示(e.g 09:58:15)
    #%Z 当前时区的名称(e.g 中国标准时间)
    #%% %号本身



    train(args, model, dataloader_train, dataloader_val, csv_path, datetime)


if __name__ == '__main__':
    print('==========Main==========')




    params_5 = [
        '--num_epochs', '300',
        '--crop_height', '224',
        '--crop_width', '224',
        '--learning_rate', '0.001',
        '--num_workers', '2',
        '--num_classes', '3',
        '--batch_size', '16',
        '--save_each_model', False,
        '--test_pic_name', 'test_landseg',  # demo原图名称
        '--data_name', 'land_seg4.0',  # 数据集名称
        '--model_name', 'LLAMNet_non_local_6'  # 模型保存名称
    ]
    model_5 = LLAMNet_non_local(classes=3)
    main(params_5, model_5)


