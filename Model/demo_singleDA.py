import os
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from Net_singleDA import train_network
import time
from utils import setup_seed, show_calaError, print_args, single_data_loading, single_data_patching, sample_gt
import hdf5storage
# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("SDAnet")
parser.add_argument('--gpu_id', default='3', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--epochs', type=int, default=200, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--dataset', choices=['S_Pavia', 'S_YRD'], default='S_YRD', help='dataset to use')
parser.add_argument('--source_name', choices=['paviaU', 'NC16'], default='NC16', help='the name of the source dir')
parser.add_argument('--target_name', choices=['paviaC', 'NC13'], default='NC13', help='the name of the target dir')
parser.add_argument('--in_channel', choices=[102, 270], default=270, help='number of channel')
parser.add_argument('--train_size', choices=[100], default=100, help='training sample size')
parser.add_argument('--flag_load', choices=['Y', 'N'], default='N', help='loading mark')
parser.add_argument('--flag_record', choices=[True, False], default=False, help='loading mark')
parser.add_argument('--num_classes', choices=[7, 7], default=7, help='number of classes')
parser.add_argument('--factor_lambda', choices=[0, 0.001, 0.01, 0.1, 1], default=0.01, help='number of classes')  # 0.001 for S_Pavia and 0.01 for S_YRD
parser.add_argument('--batch_size', type=int, default=36, help='number of batch')
parser.add_argument('--patch_size', type=int, default=5, help='size of patch')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def experiment():
    src_img, src_gt, tar_img, tar_gt, l1 = single_data_loading(args.dataset, args.source_name, args.target_name)
    print('Source domain image size is:', src_img.shape)
    print('Target domain image size is:', tar_img.shape)
    if args.flag_load == 'Y':
        data_src = hdf5storage.loadmat(
            "/home/server04/dkx/dkx_experiment/MDAnet/SDA/data/%s/%d_src_train_gt_%d.mat" % (
                args.dataset, args.train_size, args.seed))
        train_gt_src = data_src['train_gt_src']
        data_tar = hdf5storage.loadmat(
            "/home/server04/dkx/dkx_experiment/MDAnet/SDA/data/%s/tar_train_test_patch_%d.mat" % (args.dataset,
                                                                                                  args.patch_size))
        train_patch_src, train_y_src = single_data_patching(src_img, train_gt_src, patchsize=args.patch_size)
        tar_patch, tar_y = data_tar['tar_patch'], data_tar['tar_y']
        tar_patch = torch.from_numpy(tar_patch)
        tar_y = torch.from_numpy(tar_y.squeeze())
    else:
        train_gt_src, val_gt_src = sample_gt(src_gt, train_size=args.train_size, seed=args.seed, mode='fix')
        train_patch_src, train_y_src = single_data_patching(src_img, train_gt_src, patchsize=args.patch_size)
        tar_patch, tar_y = single_data_patching(tar_img, tar_gt, patchsize=args.patch_size)
    print('Dataset:', args.dataset, '| Source domain:', args.source_name, '| Target domain:', args.target_name,)
    print('Patch size:', args.patch_size, '| SD sample number:', train_y_src.shape, '| TD sample number:', tar_y.shape)
    train_sd_dataset = TensorDataset(train_patch_src, train_y_src)  # part of SD data and labels for training
    train_td_dataset = TensorDataset(tar_patch)  # all TD data for training
    test_dataset = TensorDataset(tar_patch, tar_y)  # all TD data and labels for test
    train_sd_loader = DataLoader(train_sd_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_td_loader = DataLoader(train_td_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)  # shuffle=True
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # train
    print('#######################Training ########################', args.seed)
    tic1 = time.time()
    """train Network"""
    pred, acc = train_network(train_sd_loader, train_patch_src, train_y_src, train_td_loader, test_loader,
                              LR=args.learning_rate,
                              EPOCH=args.epochs, l1=args.in_channel, Classes=args.num_classes,
                              dataset=args.dataset, factor_lambda=args.factor_lambda)
    print("***********************Train raw***************************")
    print("Maxmial Accuracy: %f, index: %i" % (max(acc), acc.index(max(acc))))
    toc1 = time.time()
    time_1 = toc1 - tic1
    print('1st training complete in {:.0f}m {:.0f}s'.format(time_1 / 60, time_1 % 60))
    OA, Kappa, CA, AA = show_calaError(pred, tar_y)
    toc = time.time()
    time_all = toc - tic1
    print('All process complete in {:.0f}m {:.0f}s'.format(time_all / 60, time_all % 60))
    print("**************************************************")
    print("Parameter:")
    print_args(vars(args))



if __name__ == '__main__':
    setup_seed(args.seed)
    experiment()