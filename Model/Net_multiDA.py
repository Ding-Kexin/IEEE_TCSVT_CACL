import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, l1, l2):
        super(Encoder, self).__init__()
        self.spa1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.spa2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.spa3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.spa4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

    def forward(self, h_spa, l_spa):
        h_spa1 = self.spa1(h_spa)
        l_spa1 = self.spa2(l_spa)
        h_spa2 = self.spa3(h_spa1)
        l_spa2 = self.spa3(l_spa1)
        h_spa3 = self.spa4(h_spa2)
        l_spa3 = self.spa4(l_spa2)
        encode_x = self.xishu1 * h_spa3 + self.xishu2 * l_spa3
        return encode_x


class Classifier(nn.Module):
    def __init__(self, Classes):
        super(Classifier, self).__init__()
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, Classes)

    def forward(self, x):
        x = self.conv1_3(x)
        x = self.conv2_3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x


class Discriminator(nn.Module):
    def __init__(self, domain_classes=2):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(32, domain_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.conv1(x)
        mapping_x = self.conv2(x)

        return mapping_x


class Network(nn.Module):
    def __init__(self, l1, l2,  Classes):
        super(Network, self).__init__()
        self.encoder = Encoder(l1, l2)
        self.classifier = Classifier(Classes)
        self.mapper = Mapper()  # Deep mapping head

    def forward(self, x1, x2):
        encode_x = self.encoder(x1, x2)
        cls_x = self.classifier(encode_x)
        mapping_x = self.mapper(encode_x)  # Deep feature mapping

        return encode_x, cls_x, mapping_x


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None, adv=False):
        """Compute loss for Model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if self.device is not None:
            device = self.device
        else:
            device = (torch.device('cuda')
                      if features.is_cuda
                      else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        if adv:
            log_prob = torch.log(1 - exp_logits / (exp_logits.sum(1, keepdim=True) + 1e-6) - 1e-6)
        else:
            log_prob = torch.log(exp_logits / (exp_logits.sum(1, keepdim=True) + 1e-6) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def ss_analysis(src_spa_prototype, src_spe_prototype, tar_encode, Classes):
    """
    :param src_spa_prototype:  dict:num_classes (dim)
    :param src_spe_prototype:   dict:num_classes (patch_size**2)
    :param tar_encode: batch_size*dim*patch_size*patch_size
    :param Classes: num_classes
    :return:
    """

    tar_encode = tar_encode.flatten(2)  # N*dim*(patch_size**2)
    tar_spa_prototype = torch.mean(tar_encode, dim=-1)  # N*dim
    tar_spe_prototype = torch.mean(tar_encode, dim=1)  # N*(patch_size**2)
    spa_cosine_similarity = torch.empty(tar_spa_prototype.shape[0], Classes)
    spe_cosine_similarity = torch.empty(tar_spe_prototype.shape[0], Classes)
    for n in range(tar_spa_prototype.shape[0]):
        for c in range(Classes):
            spa_cosine_similarity[n][c] = torch.cosine_similarity(src_spa_prototype[c].cuda(), tar_spa_prototype[n, :], dim=0, eps=1e-6)
            spe_cosine_similarity[n][c] = torch.cosine_similarity(src_spe_prototype[c].cuda(), tar_spe_prototype[n, :],
                                                                  dim=0, eps=1e-6)
    # similarity pseudo-label estimate
    spa_cosine_similarity = F.softmax(spa_cosine_similarity, dim=1)
    spe_cosine_similarity = F.softmax(spe_cosine_similarity, dim=1)
    ss_similarity = spa_cosine_similarity + spe_cosine_similarity
    ss_label = torch.argmax(ss_similarity, dim=1)
    spa_label = torch.argmax(spa_cosine_similarity, dim=1)
    spe_label = torch.argmax(spe_cosine_similarity, dim=1)

    # spa_label_rank = torch.topk(spa_cosine_similarity, k=int(Classes/2))[1]
    # spe_label_rank = torch.topk(spe_cosine_similarity, k=int(Classes / 2))[1]
    # index_consistent = np.where(spa_label == spe_label)[0]

    return spa_label, spe_label


def prototype_calc(src_spa1, src_spa2, src_y, model, Classes):
    # src feature centroid calculation
    src_spa_prototype = {}
    src_spe_prototype = {}
    for c in range(Classes):
        indices = [j for j, x in enumerate(src_y.ravel()) if src_y[j] == c]  # num_c
        src_spa1_c = src_spa1[indices]  # num_c*dim*p*p
        src_spa2_c = src_spa2[indices]  # num_c*dim*p*p
        num_src = src_spa1_c.shape[0]
        src_spa_prototype_c = np.empty((num_src, 128), dtype='float32')
        src_spe_prototype_c = np.empty((num_src, 25), dtype='float32')
        number = num_src // 100
        for i in range(number):
            temp1_1 = src_spa1_c[i * 100:(i + 1) * 100, :, :, :]
            temp1_2 = src_spa2_c[i * 100:(i + 1) * 100, :, :, :]
            temp1_1 = temp1_1.cuda()
            temp1_2 = temp1_2.cuda()
            with torch.no_grad():
                temp_2_1, _, _ = model(temp1_1, temp1_2)
            temp_2_1 = temp_2_1.flatten(2)
            temp_3_1 = torch.mean(temp_2_1, dim=-1)  # spatial
            temp_3_2 = torch.mean(temp_2_1, dim=1)  # spectral
            src_spa_prototype_c[i * 100:(i + 1) * 100] = temp_3_1.cpu()
            src_spe_prototype_c[i * 100:(i + 1) * 100] = temp_3_2.cpu()
            del temp1_1, temp_2_1, temp_3_1, temp_3_2
        if number * 100 < num_src:
            temp1_1 = src_spa1_c[number * 100:num_src, :, :, :]
            temp1_2 = src_spa2_c[number * 100:num_src, :, :, :]
            temp1_1 = temp1_1.cuda()
            temp1_2 = temp1_2.cuda()
            with torch.no_grad():
                temp_2_1, _, _ = model(temp1_1, temp1_2)
            temp_2_1 = temp_2_1.flatten(2)
            temp_3_1 = torch.mean(temp_2_1, dim=-1)  # spatial
            temp_3_2 = torch.mean(temp_2_1, dim=1)  # spectral
            src_spa_prototype_c[number * 100:num_src] = temp_3_1.cpu()
            src_spe_prototype_c[number * 100:num_src] = temp_3_2.cpu()
            del temp1_1, temp_2_1, temp_3_1
        src_spa_prototype_c = torch.from_numpy(src_spa_prototype_c)
        src_spe_prototype_c = torch.from_numpy(src_spe_prototype_c)
        src_spa_prototype_c = torch.mean(src_spa_prototype_c, dim=0)  # [128]
        src_spe_prototype_c = torch.mean(src_spe_prototype_c, dim=0)  # [25]
        src_spa_prototype[c] = src_spa_prototype_c  # dict:C, [128]
        src_spe_prototype[c] = src_spe_prototype_c  # dict:C, [25]

    return src_spa_prototype, src_spe_prototype


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', eps=1e-7):
        super(FocalLoss, self).__init__()
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target, gamma=1):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** gamma * logp
        return loss.mean()


def train_network(train_sd_loader, train_patch1_src, train_patch2_src, train_y_src, train_td_loader, test_loader, LR, EPOCH, l1, Classes,
                  dataset, factor_lambda):
    cnn = Network(l1=l1, l2=1, Classes=Classes)
    dis = Discriminator()
    cnn.cuda()
    dis.cuda()
    g_optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # Optimize all cnn parameters
    d_optimizer = torch.optim.Adam(dis.parameters(), lr=LR)
    loss_fun1 = nn.CrossEntropyLoss()  # Cross entropy loss
    loss_func2 = SupConLoss()
    loss_func3 = FocalLoss()
    test_acc = []
    BestAcc = 0
    """training"""
    cnn.train()
    for epoch in range(EPOCH):
        num_cal = 20
        if (epoch >= num_cal and epoch < EPOCH) and epoch % num_cal == 0:
            src_spa_prototype, src_spe_prototype = prototype_calc(train_patch1_src, train_patch2_src, train_y_src, cnn, Classes)

        for step, ((b_x1, b_x2, b_y), (b_u1, b_u2)) in enumerate(
                zip(train_sd_loader, train_td_loader)):  # Semi-supervised train_loader
            b_x1, b_x2, b_y = b_x1.cuda(), b_x2.cuda(), b_y.cuda()  # Move data to GPU
            b_u1, b_u2 = b_u1.cuda(), b_u2.cuda()
            encode_src, cls_src, mapping_src = cnn(b_x1, b_x2)  # Model output according to labelled train set
            encode_tar, cls_tar, mapping_tar = cnn(b_u1, b_u2)
            dis.zero_grad()
            domain_probability_src = dis(encode_src)
            domain_probability_tar = dis(encode_tar)
            src_domain_y = torch.ones_like(b_y)
            tar_domain_y = torch.zeros_like(b_y)
            d_loss = loss_fun1(domain_probability_src, src_domain_y.long()) + loss_fun1(domain_probability_tar, tar_domain_y.long())
            d_loss.backward(retain_graph=True)
            cnn.zero_grad()
            ce_loss_src = loss_fun1(cls_src, b_y.long())
            a_loss = 2 -(loss_fun1(domain_probability_src, src_domain_y.long()) + loss_fun1(domain_probability_tar, tar_domain_y.long()))
            """Consistency-aware contrastive learning"""
            num_tar = b_u1.shape[0]
            if epoch < num_cal:
                tar_loss = torch.zeros(1).cuda()
            else:
                tar_label = torch.argmax(cls_tar, dim=1)
                spa_label, spe_label = ss_analysis(src_spa_prototype, src_spe_prototype, encode_tar, Classes)
                index_ss = np.where(spa_label == spe_label)[0]
                index_sp = np.where(spa_label == tar_label.cpu())[0]
                index_consistent = np.intersect1d(index_sp, index_ss)
                index_all = np.arange(0, len(tar_label), 1, dtype=np.int16)
                index_inconsistent = np.setdiff1d(index_all, index_consistent)
                if len(index_consistent) == 0:
                    num_c = 0
                    consistent_con_loss = torch.zeros(1).cuda()
                    consistent_focal_loss = torch.zeros(1).cuda()

                else:
                    num_c = len(index_consistent)
                    mapping_c = mapping_tar[index_consistent]
                    y_c = tar_label[index_consistent]
                    map_cross = torch.cat((mapping_src, mapping_c), dim=0)
                    y_cross = torch.cat((b_y, y_c), dim=0)
                    consistent_con_loss = loss_func2(map_cross, y_cross.long())
                    cls_tar_c = cls_tar[index_consistent]
                    consistent_focal_loss = loss_func3(cls_tar_c, y_c.long(), gamma=1-(num_c/num_tar))
                if len(index_inconsistent) == 0:
                    num_i = 0
                    inconsistent_con_loss = torch.zeros(1).cuda()
                    inconsistent_focal_loss = torch.zeros(1).cuda()

                else:
                    num_i = len(index_inconsistent)
                    mapping_i = mapping_tar[index_inconsistent]
                    y_i = tar_label[index_inconsistent]
                    inconsistent_con_loss = loss_func2(mapping_i, y_i.long())
                    cls_tar_i = cls_tar[index_inconsistent]
                    inconsistent_focal_loss = loss_func3(cls_tar_i, y_i.long(), gamma=1-(num_i/num_tar))
                consistent_loss = consistent_con_loss + consistent_focal_loss
                inconsistent_loss = inconsistent_con_loss + inconsistent_focal_loss
                tar_loss = (num_c / num_tar) * consistent_loss + (num_i / num_tar) * inconsistent_loss
            g_loss = ce_loss_src + factor_lambda * a_loss + tar_loss
            g_loss.backward(retain_graph=True)
            d_optimizer.step()
            g_optimizer.step()  # Update parameters of cnn

            if step % 500 == 0:
                cnn.eval()
                test_pred_all = []
                test_all = []
                predict = np.array([], dtype=np.int64)
                test_loss = 0
                correct_add = 0
                size = 0
                for batch_idx, data in enumerate(test_loader):
                    img1, img2, label = data
                    img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
                    _, output, _ = cnn(img1, img2)
                    pred = output.data.max(1)[1]
                    test_loss += F.nll_loss(F.log_softmax(output, dim=1), label, reduction='sum').item()
                    correct_add += pred.eq(label.data).cpu().sum()
                    size += label.data.size()[0]
                    test_all = np.concatenate([test_all, label.data.cpu().numpy()])
                    test_pred_all = np.concatenate([test_pred_all, pred.cpu().numpy()])
                    predict = np.append(predict, pred.cpu().numpy())
                test_accuracy = 100. * float(correct_add) / size
                test_loss /= len(test_loader.dataset)  # loss function already averages over batch size
                print('Epoch: ', epoch, '| cls_src loss: %.6f' % ce_loss_src.data.cpu().numpy(),
                      '| dis loss: %.6f' % d_loss.data.cpu().numpy(), '| adv loss: %.6f' % a_loss.data.cpu().numpy(),
                      '| test loss: %.6f' % test_loss,
                      '| test accuracy: %d/%d (%.6f)' % (correct_add, size, test_accuracy))
                test_acc.append(test_accuracy)
                # Save the parameters in network
                if test_accuracy > BestAcc:
                    torch.save(cnn.state_dict(),
                               './log/Net_multiDA_%s.pkl' % (dataset))
                    BestAcc = test_accuracy

                cnn.train()  # Open Batch Normalization and Dropout

    cnn.load_state_dict(torch.load('./log/Net_multiDA_%s.pkl' % (dataset)))
    cnn.eval()
    predict = np.array([], dtype=np.int64)
    for batch_idx, data in enumerate(test_loader):
        img1, img2, label = data
        img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
        _, output, _ = cnn(img1, img2)
        pred = output.data.max(1)[1]
        test_loss += F.nll_loss(F.log_softmax(output, dim=1), label, reduction='sum').item()
        correct_add += pred.eq(label.data).cpu().sum()
        size += label.data.size()[0]
        test_all = np.concatenate([test_all, label.data.cpu().numpy()])
        test_pred_all = np.concatenate([test_pred_all, pred.cpu().numpy()])
        predict = np.append(predict, pred.cpu().numpy())
    predict = torch.from_numpy(predict)
    return predict, test_acc