import argparse
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2,3'
import shutil
import time
import torch
torch.set_printoptions(profile="full", precision=3)
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from utils.scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from lib.dataset import MyDataset
from lib.utils import create_label, mkdir_or_exist, get_root_logger
from datetime import datetime
from lib.model_search_2 import Similarity
# from lib.FPN import FPN
import logging

# CUDA
use_cuda = torch.cuda.is_available()




def parsing():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument(
        '--num_epochs', type=int, default=2000,
        help='number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-6,
        help='initial learning rate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='batch size'
    )
    parser.add_argument(
        '--valid_batch_size', type=int, default=4,
        help='batch size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='number of workers for data loading'
    )
    parser.add_argument(
        '--use_validation', dest='use_validation', action='store_true',
        help='use the validation split'
    )
    parser.set_defaults(use_validation=False)
    parser.add_argument(
        '--z_size', type=int, default=512,
        help='size of the small img'
    )
    parser.add_argument(
        '--x_size', type=int, default=600,
        help='size of the big img'
    )

    parser.add_argument(
        '--log_interval', type=int, default=10,
        help='loss logging interval'
    )
    parser.add_argument(
        '--log_interval_v', type=int, default=20,
        help='loss logging interval'
    )

    parser.add_argument(
        '--eval_interval', type=int, default=400,
        help='eval interval per iter'
    )
    parser.add_argument(
        '--ckpt_interval', type=int, default=10,
        help='checkpoint interval per epoch'
    )
    parser.add_argument(
        '--work_dir', type=str, default=f'./workdirs/similarity_v1',
        help='loss logging file'
    )

    parser.add_argument(
        '--checkpoint_directory', type=str, default='checkpoints',
        help='directory for training checkpoints'
    )
    parser.add_argument(
        '--checkpoint_prefix', type=str, default='map2opt',
        help='prefix for training checkpoints'
    )
    parser.add_argument('--gpu_ids', type=str, default='0,1,5,6', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument(
        '--resume_model_file', type=str, default=None,
        help='path to the full model'
    )
    args = parser.parse_args()
    arg_text = ''  # change dict to str
    for eachArg, value in args.__dict__.items():
        arg_text += str(eachArg) + ' : ' + str(value) + '\n'

    return args, arg_text


def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def main():
    args, arg_text = parsing()

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.work_dir = os.path.join(args.work_dir, timestamp)  # 创建work_dir
    mkdir_or_exist(os.path.abspath(args.work_dir))

    # init the logger
    log_file = os.path.join(args.work_dir, 'root.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    # log some basic info
    logger.info('training gpus num: {}'.format(args.gpu_ids))
    logger.info('Config:\n{}'.format(arg_text))

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if use_cuda else torch.device('cpu')

    model = Similarity(z_size=args.z_size,x_size=args.x_size)
    if len(args.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        model.to(args.gpu_ids[0])
        model = torch.nn.DataParallel(model, args.gpu_ids)  # multi-GPUs

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    param_nums = 0
    for item in model.parameters():  # 得到类的参数
        param_nums += np.prod(np.array(item.shape))
    logger.info("model: {} 's total parameter nums: {}".format(model.module.__class__.__name__,
                                                               param_nums))  # 如果用model.__class__ 则输出的类名为 DataParallel 用model.module则为similarity类

    '''
    使用Warmup预热学习率的方式,即先用最初的小学习率训练，然后每个step增大一点点，
    直到达到最初设置的比较大的学习率时（注：此时预热学习率完成），采用最初设置的学习率进行训练
    '''
    scheduler_reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                             factor=0.1,
                                                                             patience=5,
                                                                             verbose=True)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=5,
    #                                           after_scheduler=scheduler_reduce_on_plateau)
    # Dataset
    if args.use_validation:
        validation_dataloader = DataLoader(MyDataset(mode='eval'), batch_size=args.batch_size,
                                           num_workers=args.num_workers)
    training_dataloader = DataLoader(MyDataset(mode='Train'), batch_size=args.valid_batch_size,
                                     num_workers=args.num_workers, shuffle=True)

    # Create the checkpoint directory
    args.checkpoint_directory = os.path.join(args.work_dir, args.checkpoint_directory)  # 创建checkpoint 文件夹
    mkdir_or_exist(os.path.abspath(args.checkpoint_directory))

    epoch_start = 1
    min_eval_dis = 1000
    margin = 1

    if args.resume_model_file is not None:
        checkpoint = torch.load(args.resume_model_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # epoch_start = checkpoint['epoch_idx']
        # min_validation_loss = checkpoint['validation_loss_history']
        # args = checkpoint['args'] #如果加载了args 那么work_dir等等就会变成之前的
        # print('min_validation_loss', min_validation_loss[-1])
        # print('epoch start', epoch_start)
        # print('args', args.lr)
        # print('load model from checkpoint')
        logger.info('load checkpoint from %s', args.resume_model_file)
    else:
        model.apply(init_xavier)

    train_loss_history = []  # 记录所有epoch的loss
    validation_loss_history = []
    dis_history = []
    # criterion = nn.BCELoss(reduction='sum')  # 求和，不求均值，后面再除以正样本个数
    max_iters = args.num_epochs * len(training_dataloader)  # 记录迭代次数
    logger.info("{} iters for one training epoch, total iters: {}".format(len(training_dataloader), max_iters))
    logger.info("Start running, work_dir: {}, max epochs : {}".format(args.work_dir, args.num_epochs))

    _iter = inner_iter = 0
    for epoch_idx in range(epoch_start, epoch_start + args.num_epochs):
        # scheduler_warmup.step()  #lr预热
        train_epoch_loss = []  # 记录每一轮epoch的loss 每次都要清空
        valid_epoch_loss = []

        for batch_idx, batch in enumerate(training_dataloader):
            _iter += 1  # 总iter
            model.train()

            inner_iter = batch_idx

            optimizer.zero_grad()
            score = model(batch['pair_1'].to(device), batch['pair_2'].to(device))  # b,s,s s=19
            search_size = score.shape[1]
            label = create_label(batch['label'].to(device), search_size)  # b,s,s
            B, H, W = score.shape
            label = label.reshape(B, -1)  # (B,25)
            score = torch.sigmoid(score).reshape(B, -1)  # (B,25)
            loss = -1.0 * (label * torch.log(score) + (1.0 - label) * torch.log(1 - score)).mean()

            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())

            output = []
            max_id = torch.argmax(score, 1)  # (B, )
            predefined_lt_point = model.module.bbox_corner
            for i in range(B):
                H_id = max_id[i] // H
                W_id = max_id[i] % H
                output.append(predefined_lt_point[0, :, H_id, W_id].unsqueeze(0))  # (1,2)
                # print(output.shape)

            output = torch.cat(output, 0)  # (B, 2)
            dis = torch.norm(torch.floor(output + 0.5) - batch['label'], p=2, dim=1).mean()  # (B, )

            if _iter % args.log_interval == 0:
                logger.info('[%s] epoch %d - now_iter %d / %d - now_loss:%.5f - avg_loss: %.5f -dis: %.5f -iter: %d' % (
                    'train',
                    epoch_idx, inner_iter, len(training_dataloader), loss.item(), np.mean(train_epoch_loss), dis, _iter
                ))

            if _iter % args.eval_interval == 0:
                logger.info("start to eval for iter: {}".format(_iter))
                with torch.no_grad():  # 不加这句话要爆显存 就离谱
                    model.eval()
                    dis_all_data = []


                    for batch_idx, batch in enumerate(validation_dataloader):  # 验证集应该不需要Loss,应该遍历 得到所有的结果后再一起eval！
                        """
                        首先根据每个batch 的label 和预测的bbox 求得当前batch的误差，再存入一个list中
                        等eval所有数据遍历完成，再求eval上的平均误差？
                        """
                        output = []
                        score = model(batch['pair_1'].to(device), batch['pair_2'].to(device))  # b,s,s s=19

                        B, H, W = score.shape
                        score = torch.sigmoid(score).reshape(B, -1)  # (B,25)

                        max_id = torch.argmax(score, 1)  # (B, )
                        predefined_lt_point = model.module.bbox_corner
                        for i in range(B):
                            # tmp_result = dict()
                            H_id = max_id[i] // H
                            W_id = max_id[i] % H
                            # file_name = str(batch['class_id'][i]) + '_' + str(batch['file_id'][i])
                            # pre_lt_point = predefined_lt_point[0, :, H_id, W_id].unsqueeze(0)
                            # result=model.module.cal_for_eval(file_name,pre_lt_point)
                            # tmp_result['file_name'] = file_name
                            # tmp_result['dis'] = torch.norm(torch.floor(pre_lt_point + 0.5) - batch['label'][i], p=2,dim=1)  # (1)
                            # print(tmp_result['dis'])
                            # print(type(output))
                            output.append(predefined_lt_point[0, :, H_id, W_id].unsqueeze(0))  # (1,2)
                            # output.append(tmp_result)

                        output = torch.cat(output, 0)  # (2, 2)
                        dis = torch.norm(torch.floor(output + 0.5) - batch['label'], p=2,
                                          dim=1)  # .mean()  # (B, ) 不求平均则维度为 B,
                        dis_all_data.append(dis)
                    dis_all_data = torch.cat(dis_all_data, 0).mean()  # torch.Size([470]) mean之后size=1

                    logger.info(
                        '[%s] epoch %d - now_iter %d - now_dis:%.5f ' % ('valid', epoch_idx, _iter, dis_all_data))
                    if dis_all_data < min_eval_dis:
                        min_eval_dis = dis_all_data
                        best_checkpoint_path = os.path.join(
                            args.checkpoint_directory,
                            '%s.best.pth' % args.checkpoint_prefix
                        )
                        checkpoint = {
                            'args': args,
                            'epoch_idx': epoch_idx,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'train_loss_history': train_loss_history,
                            'dis': dis_all_data
                        }
                        torch.save(checkpoint, best_checkpoint_path)
                        logger.info('save best checkpoint to {}'.format(best_checkpoint_path))

                # validation_loss_history.append(np.mean(valid_epoch_loss))

        train_loss_history.append(np.mean(train_epoch_loss))
        scheduler_reduce_on_plateau.step(np.mean(valid_epoch_loss))

        # if _iter % args.ckpt_interval == 0:
        # Save the current checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_directory,
            '%s.%02d.pth' % (args.checkpoint_prefix, epoch_idx)
        )
        checkpoint = {
            'args': args,
            'epoch_idx': epoch_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss_history': train_loss_history,
            'validation_loss_history': validation_loss_history
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info('save checkpoint to {}'.format(checkpoint_path))
        # logger.info('save checkpoint to {}'.format(checkpoint_path))
        # if args.use_validation and validation_loss_history[-1] < min_validation_loss:
        #     min_validation_loss = validation_loss_history[-1]
        #     best_checkpoint_path = os.path.join(
        #         args.checkpoint_directory,
        #         '%s.best.pth' % args.checkpoint_prefix
        #     )
        #     shutil.copy(checkpoint_path, best_checkpoint_path)
        #     logger.info('save best checkpoint to {}'.format(best_checkpoint_path))


if __name__ == '__main__':
    main()
