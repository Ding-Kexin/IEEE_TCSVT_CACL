import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import preprocessing
import torch
import hdf5storage
import random


SINGLE_DATASETS_CONFIG = {
    'S_Pavia': {
        'data_path': '/home/server04/dkx/dkx_experiment/dataset/HSI_cross-scene/Pavia/Pavia/',
        'paviaU': {
            'img_name': 'paviaU',
            'gt_name': 'paviaU_7gt',
        },
        'paviaC': {
            'img_name': 'paviaC',
            'gt_name': 'paviaC_7gt',
        },
    },

    'S_YRD': {
        'data_path': '/home/server04/dkx/dkx_experiment/dataset/HSI_cross-scene/YRD/',
        'NC16': {
            'img_name': 'NC16',
            'gt_name': 'NC16_7gt',
        },
        'NC13': {
            'img_name': 'NC13',
            'gt_name': 'NC13_7gt',
        },
    },


}
MULTI_DATASETS_CONFIG = {
    'M_Houston': {
        'data_path': '/home/server04/dkx/dkx_experiment/dataset/Multi_cross_scene/Houston_cross_scene/',
        'Houston13': {
            'img1_name': 'Houston13_HSI48',
            'img2_name': 'Houston13_DSM',
            'gt_name': 'Houston13_GT7',
        },
        'Houston18': {
            'img1_name': 'Houston18_HSI',
            'img2_name': 'Houston18_DSM',
            'gt_name': 'Houston18_GT7',
        },
    },

}


def multi_data_loading(dataset_name, source_name, target_name, datasets=MULTI_DATASETS_CONFIG):
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    dataset = datasets[dataset_name]
    data_path = dataset['data_path']
    src_data = dataset[source_name]
    tar_data = dataset[target_name]
    src_img1 = hdf5storage.loadmat(data_path + src_data['img1_name'] + '.mat')[src_data['img1_name']]
    src_img2 = hdf5storage.loadmat(data_path + src_data['img2_name'] + '.mat')[src_data['img2_name']]
    src_gt = hdf5storage.loadmat(data_path + src_data['gt_name'] + '.mat')[src_data['gt_name']]
    tar_img1 = hdf5storage.loadmat(data_path + tar_data['img1_name'] + '.mat')[tar_data['img1_name']]
    tar_img2 = hdf5storage.loadmat(data_path + tar_data['img2_name'] + '.mat')[tar_data['img2_name']]
    tar_gt = hdf5storage.loadmat(data_path + tar_data['gt_name'] + '.mat')[tar_data['gt_name']]
    # Normalization
    [m1, n1, l1] = np.shape(src_img1)
    [m2, n2, _] = np.shape(tar_img1)
    src_img1_2d = src_img1.reshape((m1 * n1, -1))  # 2D
    src_img2_2d = src_img2.reshape((m1 * n1, -1))  # 2D
    tar_img1_2d = tar_img1.reshape((m2 * n2, -1))
    tar_img2_2d = tar_img2.reshape((m2 * n2, -1))
    src_img1_2d = preprocessing.minmax_scale(src_img1_2d)  # Normalization
    src_img2_2d = preprocessing.minmax_scale(src_img2_2d)  # Normalization
    tar_img1_2d = preprocessing.minmax_scale(tar_img1_2d)
    tar_img2_2d = preprocessing.minmax_scale(tar_img2_2d)  # Normalization
    src_img1 = np.reshape(src_img1_2d, (m1, n1, -1))
    src_img2 = np.reshape(src_img2_2d, (m1, n1, -1))
    tar_img1 = np.reshape(tar_img1_2d, (m2, n2, -1))
    tar_img2 = np.reshape(tar_img2_2d, (m2, n2, -1))
    return src_img1, src_img2, src_gt, tar_img1, tar_img2, tar_gt, l1


def multi_data_patching(Data1, Data2, Label, patchsize):
    # Filter NaN out
    nan_mask1 = np.isnan(Data1.sum(axis=-1))
    nan_mask2 = np.isnan(Data2.sum(axis=-1))
    if np.count_nonzero(nan_mask1) > 0:
        print(
            "Warning: NaN have been found in the Data1. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    if np.count_nonzero(nan_mask2) > 0:
        print(
            "Warning: NaN have been found in the Data2. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    Data1[nan_mask1] = 0
    Data2[nan_mask2] = 0
    Label[nan_mask1] = 0
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    r = patchsize // 2
    x1_pad = np.pad(Data1, ((r, r), (r, r), (0, 0)), 'symmetric')
    x2_pad = np.pad(Data2, ((r, r), (r, r), (0, 0)), 'symmetric')
    # construct patches
    [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + r
    ind4 = ind2 + r
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - r):(ind3[i] - r + patchsize), (ind4[i] - r):(ind4[i] - r + patchsize), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - r):(ind3[i] - r + patchsize), (ind4[i] - r):(ind4[i] - r + patchsize), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainLabel


def single_data_loading(dataset_name, source_name, target_name, datasets=SINGLE_DATASETS_CONFIG):
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    dataset = datasets[dataset_name]
    data_path = dataset['data_path']
    src_data = dataset[source_name]
    tar_data = dataset[target_name]
    src_img = hdf5storage.loadmat(data_path + src_data['img_name'] + '.mat')['ori_data']
    src_gt = hdf5storage.loadmat(data_path + src_data['gt_name'] + '.mat')['map']
    tar_img = hdf5storage.loadmat(data_path + tar_data['img_name'] + '.mat')['ori_data']
    tar_gt = hdf5storage.loadmat(data_path + tar_data['gt_name'] + '.mat')['map']
    # Normalization
    [m1, n1, l1] = np.shape(src_img)
    [m2, n2, _] = np.shape(tar_img)
    src_img_2d = src_img.reshape((m1 * n1, -1))  # 2D
    tar_img_2d = tar_img.reshape((m2 * n2, -1))
    src_img_2d = preprocessing.minmax_scale(src_img_2d)  # Normalization
    tar_img_2d = preprocessing.minmax_scale(tar_img_2d)
    src_img = np.reshape(src_img_2d, (m1, n1, -1))
    tar_img = np.reshape(tar_img_2d, (m2, n2, -1))
    return src_img, src_gt, tar_img, tar_gt, l1


def single_data_patching(Data1, Label, patchsize):
    # Filter NaN out
    nan_mask1 = np.isnan(Data1.sum(axis=-1))
    if np.count_nonzero(nan_mask1) > 0:
        print(
            "Warning: NaN have been found in the Data1. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    Data1[nan_mask1] = 0
    Label[nan_mask1] = 0
    [m1, n1, l1] = np.shape(Data1)
    r = patchsize // 2
    x1_pad = np.pad(Data1, ((r, r), (r, r), (0, 0)), 'symmetric')
    # construct patches
    [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + r
    ind4 = ind2 + r
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - r):(ind3[i] - r + patchsize), (ind4[i] - r):(ind4[i] - r + patchsize), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainLabel


def sampling_random_ratio(sample_ratio, gt, seed):
    np.random.seed(seed)
    indices = [j for j, x in enumerate(gt.ravel().tolist()) if x > 0]
    np.random.shuffle(indices)
    train_random_indices = indices[:sample_ratio * len(indices)]
    test_random_indices = indices[sample_ratio * len(indices):]
    np.random.shuffle(train_random_indices)
    np.random.shuffle(test_random_indices)
    return train_random_indices, test_random_indices


def sampling_fixed_num(sample_num, gt, seed):
    labels_loc = {}
    train_ = {}
    test_ = {}
    np.random.seed(seed)
    m = max(gt)
    for i in range(m):
        indices = [j for j, x in enumerate(gt.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        train_[i] = indices[:sample_num]
        test_[i] = indices[sample_num:]
    train_fix_indices = []
    test_fix_indices = []
    for i in range(m):
        train_fix_indices += train_[i]
        test_fix_indices += test_[i]
    np.random.shuffle(train_fix_indices)
    np.random.shuffle(test_fix_indices)
    return train_fix_indices, test_fix_indices


def sampling_fixed_list(sample_num_list, gt, seed):
    labels_loc = {}
    train_ = {}
    test_ = {}
    np.random.seed(seed)
    for idx, each_num in enumerate(sample_num_list):
        indices = [j for j, x in enumerate(gt.ravel().tolist()) if x == idx + 1]
        np.random.shuffle(indices)
        labels_loc[idx] = indices
        train_[idx] = indices[:each_num]
        test_[idx] = indices[each_num:]
    train_diff_indices = []
    test_diff_indices = []
    for i in range(len(sample_num_list)):
        train_diff_indices += train_[i]
        test_diff_indices += test_[i]
    np.random.shuffle(train_diff_indices)
    np.random.shuffle(test_diff_indices)
    return train_diff_indices, test_diff_indices


def sample_gt(gt, train_size, seed, mode='fix'):
    gt_1d = gt.reshape(np.prod(gt.shape[:2]), ).astype(np.int)
    train_data = np.zeros(np.prod(gt.shape[:2]), ).astype(np.int)
    test_data = np.zeros(np.prod(gt.shape[:2]), ).astype(np.int)

    if mode == 'fix':
        train_index, test_index = sampling_fixed_num(train_size, gt_1d, seed)
    elif mode == 'random':
        train_index, test_index = sampling_random_ratio(train_size, gt_1d, seed)
    elif mode == 'list':
        train_index, test_index = sampling_fixed_list(train_size, gt_1d, seed)
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))

    train_data[train_index] = gt_1d[train_index]
    test_data[test_index] = gt_1d[test_index]
    train_data = train_data.reshape(np.prod(gt.shape[:1]), np.prod(gt.shape[1:]))
    test_data = test_data.reshape(np.prod(gt.shape[:1]), np.prod(gt.shape[1:]))

    return train_data, test_data


def whole_data_patching(Data1, Label, patchsize):
    # Filter NaN out
    nan_mask1 = np.isnan(Data1.sum(axis=-1))
    if np.count_nonzero(nan_mask1) > 0:
        print(
            "Warning: NaN have been found in the Data1. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    Data1[nan_mask1] = 0
    Label[nan_mask1] = 0
    [m1, n1, l1] = np.shape(Data1)
    r = patchsize // 2
    x1_pad = np.pad(Data1, ((r, r), (r, r), (0, 0)), 'symmetric')
    # construct patches
    # whole_Label = np.ones((m1, n1))
    [ind1, ind2] = np.where(Label == 1)  # [300,300]
    # [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + r
    ind4 = ind2 + r
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - r):(ind3[i] - r + patchsize), (ind4[i] - r):(ind4[i] - r + patchsize), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainLabel


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


def CalAccuracy(predict, label):
    n = label.shape[0]
    OA = torch.sum(predict==label)*1.0/n
    correct_sum = torch.zeros((max(label)+1))
    reali = torch.zeros((max(label)+1))
    predicti = torch.zeros((max(label)+1))
    CA = torch.zeros((max(label)+1))
    for i in range(0, max(label) + 1):
        correct_sum[i] = torch.sum(label[np.where(predict == i)] == i)
        reali[i] = torch.sum(label == i)
        predicti[i] = torch.sum(predict == i)
        CA[i] = correct_sum[i] / reali[i]

    Kappa = (n * torch.sum(correct_sum) - torch.sum(reali * predicti)) * 1.0 / (n * n - torch.sum(reali * predicti))
    AA = torch.mean(CA)
    return OA, Kappa, CA, AA


def show_calaError(val_predict_labels, val_true_labels):
   val_predict_labels = torch.squeeze(val_predict_labels)
   val_true_labels = torch.squeeze(val_true_labels)
   OA, Kappa, CA, AA = CalAccuracy(val_predict_labels, val_true_labels)
   # ic(OA, Kappa, CA, AA)
   print("OA: %f, Kappa: %f,  AA: %f" % (OA, Kappa, AA))
   print("CA: ",)
   print(CA)
   return OA, Kappa, CA, AA

