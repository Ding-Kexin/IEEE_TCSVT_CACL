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
def single_data_loading_pingdong(dataset_name, source_name, target_name, datasets=SINGLE_DATASETS_CONFIG):
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    dataset = datasets[dataset_name]
    data_path = dataset['data_path']
    src_data = dataset[source_name]
    tar_data = dataset[target_name]
    src_img = hdf5storage.loadmat(data_path + src_data['img1_name'] + '.mat')[src_data['img1_name']]
    src_gt = hdf5storage.loadmat(data_path + src_data['gt_name'] + '.mat')[src_data['gt_name']]
    tar_img = hdf5storage.loadmat(data_path + tar_data['img1_name'] + '.mat')[tar_data['img1_name']]
    # tar_gt = hdf5storage.loadmat(data_path + tar_data['gt_name'] + '.mat')[tar_data['gt_name']]
    tar_gt = np.ones_like(src_gt)
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
def single_data_loading2(dataset_name, source_name, target_name, datasets=SINGLE_DATASETS_CONFIG):
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    dataset = datasets[dataset_name]
    data_path = dataset['data_path']
    src_data = dataset[source_name]
    tar_data = dataset[target_name]
    src_img = hdf5storage.loadmat(data_path + src_data['img1_name'] + '.mat')[src_data['img1_name']]
    src_gt = hdf5storage.loadmat(data_path + src_data['gt_name'] + '.mat')[src_data['gt_name']]
    tar_img = hdf5storage.loadmat(data_path + tar_data['img1_name'] + '.mat')[tar_data['img1_name']]
    tar_gt = hdf5storage.loadmat(data_path + tar_data['gt_name'] + '.mat')[tar_data['gt_name']]
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


def train_patch(Data1, Data2, patchsize, Label):
    r = patchsize // 2
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([Data2.shape[0], Data2.shape[1], -1])
    [m2, n2, l2] = np.shape(Data2)
    # Filter NaN out
    nan_mask1 = np.isnan(Data1.sum(axis=-1))
    if np.count_nonzero(nan_mask1) > 0:
       print("Warning: NaN have been found in the Data1. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    Data1[nan_mask1] = 0
    Label[nan_mask1] = 0
    nan_mask2 = np.isnan(Data2.sum(axis=-1))
    if np.count_nonzero(nan_mask2) > 0:
       print("Warning: NaN have been found in the Data2. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    Data2[nan_mask1] = 0
    Label[nan_mask2] = 0
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, ((r, patchsize - r), (r, patchsize - r)), 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, ((r, patchsize - r), (r, patchsize - r)), 'symmetric')
        x2_pad[:, :, i] = temp2
    # construct the training and testing set
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


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def cost_matrix_batch_torch(x, y):
    "Returns the cosine distance batchwise"
    # x is the image feature: bs * d * m * m
    # y is the audio feature: bs * d * nF
    # return: bs * n * m
    # print(x.size())
    bs = list(x.size())[0]
    D = x.size(1)
    assert(x.size(1)==y.size(1))
    # x = x.contiguous().view(bs, D, -1) # bs * d * m^2
    x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
    y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
    cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)#.transpose(1,2)
    cos_dis = 1 - cos_dis # to minimize this value
    # cos_dis = - cos_dis
    return cos_dis.transpose(2, 1)

def cos_batch_torch(x, y):
    "Returns the cosine distance batchwise"
    # x is the image feature: bs * d * m * m
    # y is the audio feature: bs * d * nF
    # return: bs * n * m
    # print(x.size())
    bs = x.size(0)
    D = x.size(1)
    assert(x.size(1)==y.size(1))
    x = x.contiguous().view(bs, D, -1) # bs * d * m^2
    x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
    y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
    cos_dis = torch.bmm(torch.transpose(x,1,2), y)#.transpose(1,2)
    cos_dis = 1 - cos_dis # to minimize this value
    # return cos_dis.transpose(2,1)
    # TODO:
    beta = 0.1
    min_score = cos_dis.min()
    max_score = cos_dis.max()
    threshold = min_score + beta * (max_score - min_score)
    res = cos_dis - threshold
    # res = torch.nn.ReLU()

    return torch.nn.functional.relu(res.transpose(2,1))

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def IPOT_distance_torch_batch_uniform(C, bs, n, m, iteration=50):
    C = C.float().cuda()
    T = IPOT_torch_batch_uniform(C, bs, n, m, iteration=iteration)
    temp = torch.bmm(torch.transpose(C,1,2), T)
    distance = batch_trace(temp, m, bs)
    return -distance

def IPOT_torch_batch_uniform(C, bs, n, m, beta=0.5, iteration=50):
    # C is the distance matrix
    # c: bs by n by m
    sigma = torch.ones(bs, int(m), 1).cuda()/float(m)
    T = torch.ones(bs, n, m).cuda()
    A = torch.exp(-C/beta).float().cuda()
    for t in range(1):
        Q = A * T # bs * n * m
        del T
        for k in range(iteration):
            delta = 1 / (n * torch.bmm(Q, sigma))
            a = torch.bmm(torch.transpose(Q,1,2), delta)
            sigma = 1 / (float(m) * a)
        T = delta * Q * sigma.transpose(2,1)
        del Q

    return T#.detach()

def GW_distance(X, Y, p, q, lamda=0.5, iteration=5, OT_iteration=20, **kwargs):
    '''
    :param X, Y: Source and target embeddings , batchsize by embed_dim by n
    :param p, q: probability vectors
    :param lamda: regularization
    :return: GW distance
    '''
    Cs = cos_batch_torch(X, X).float().cuda()
    Ct = cos_batch_torch(Y, Y).float().cuda()

    # pdb.set_trace()
    bs = Cs.size(0)
    m = Ct.size(2)
    n = Cs.size(2)
    T, Cst = GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
    temp = torch.bmm(torch.transpose(Cst,1,2), T)
    distance = batch_trace(temp, m, bs)
    return distance

def GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
    one_m = torch.ones(bs, m, 1).float().cuda()
    one_n = torch.ones(bs, n, 1).float().cuda()

    Cst = torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_m, 1, 2)) + \
          torch.bmm(one_n, torch.bmm(torch.transpose(q,1,2), torch.transpose(Ct**2, 1, 2))) # bs by n by m
    gamma = torch.bmm(p, q.transpose(2,1)) # outer product, init
    # gamma = torch.einsum('bi,bj->bij', (torch.squeeze(p), torch.squeeze(q))) # outer product, initialization
    for i in range(iteration):
        C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
        # # Sinkhorn iteration
        # b = torch.ones(bs, m, 1).cuda()
        # K = torch.exp(-C_gamma/beta)
        # for i in range(50):cd
        # 	a = p/(torch.bmm(K, b))
        # 	b = q/torch.bmm(K.transpose(1,2), a)
        # gamma = a * K * b
        gamma = IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
    Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
    return gamma.detach(), Cgamma

def GW_distance_uniform(X, Y, lamda=1e-1, iteration=5, OT_iteration=20, **kwargs):
    m = X.size(2)
    n = Y.size(2)
    bs = X.size(0)
    p = (torch.ones(bs, m, 1)/m).cuda()
    q = (torch.ones(bs, n, 1)/n).cuda()
    return GW_distance(X, Y, p, q, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration, **kwargs)

def batch_trace(input_matrix, n, bs):
    a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
    b = a * input_matrix
    return torch.sum(torch.sum(b,-1),-1).unsqueeze(1)


def train_patch2(Data1, Data2, patchsize, Label):
    pad_width = patchsize // 2

    x1_pad = Data1
    x2_pad = Data2

    # construct the training and testing set
    [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, Data1.shape[2], patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, Data2.shape[2], patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width

    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] - pad_width + patchsize), (ind4[i] - pad_width):(ind4[i] - pad_width + patchsize), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] - pad_width + patchsize), (ind4[i] - pad_width):(ind4[i] - pad_width + patchsize), :]
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

