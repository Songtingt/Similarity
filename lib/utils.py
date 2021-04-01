import matplotlib.pyplot as plt

import numpy as np

import torch
import os
import os.path as osp
from torch.distributed import get_rank
import logging
logger_initialized = {}
def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized.
    By default a StreamHandler will be added.
    If `log_file` is specified, a FileHandler will also be added.
    The name of the root logger is the top-level package name, e.g., "edit".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    # root logger name: mmedit
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger
def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger


    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    # fix stream twice bug
    # while logger.handlers:
    #     logger.handlers.pop()
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    rank = 0

    # only rank 0 will add a FileHandler

    file_handler = logging.FileHandler(log_file, 'w')
    handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger
def create_label(location, search_size, stride=16):  # location torch.size(4,2)
    """
    Get the cls label
    the location should be downsample due to stride
    Args:
        location: The gt label
        search_size: The size of the score map



    """
    x = torch.arange(search_size)
    y = torch.arange(search_size)
    ys, xs = torch.meshgrid(x, y)
    location = torch.floor_divide(location, stride)  # 因为是对feature map进行搜索所以要降采样

    b, _ = location.shape
    xs = xs.repeat(b, 1, 1).float().cuda()  # 因为bs=16所以要重复16组一模一样的x坐标  b,s,s
    ys = ys.repeat(b, 1, 1).float().cuda()

    gt_x = location[:, 0].reshape(-1, 1).unsqueeze(1).float()  # b,1,1
    gt_y = location[:, 1].reshape(-1, 1).unsqueeze(1).float()  # b,1,1
    dist = torch.sqrt((xs - gt_x) ** 2 + (ys - gt_y) ** 2)  # block distance b,s,s

    # print(dist)
    labels = torch.where(dist <= 5,  # np.where(condition, x, y) 条件为真就返回x 为假则返回y
                      ((-dist/6)+1),
                      torch.zeros_like(xs))
    labels=torch.clamp(labels,0,1)
    # print(location[0])
    # print(labels[0])

    return labels


def create_label_same_size(location, search_size):  # location torch.size(4,2)
    x = torch.arange(search_size)
    y = torch.arange(search_size)
    ys, xs = torch.meshgrid(x, y)
    # print(xs.shape)

    b, _ = location.shape
    xs = xs.repeat(b, 1, 1).float()  # 因为bs=16所以要重复16组一模一样的x坐标  b,s,s
    ys = ys.repeat(b, 1, 1).float()

    gt_x = location[:, 0].reshape(-1, 1).unsqueeze(1).float()  # b,1,1
    gt_y = location[:, 1].reshape(-1, 1).unsqueeze(1).float()  # b,1,1

    dist = torch.sqrt((xs - gt_x) ** 2 + (ys - gt_y) ** 2)  # block distance b,s,s

    labels = torch.where(dist < 100,  # np.where(condition, x, y) 条件为真就返回x 为假则返回y
                         (1 / (0.3 * np.pi)) * torch.exp(-dist ** 2 / 2),
                         torch.zeros_like(xs))

    return labels

