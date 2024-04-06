import random

import torch
import torch.nn as nn


class infoNCE_g():
    def __init__(self, features=None, labels=None, class_num=10, feature_dim=512):
        super(infoNCE_g, self).__init__()
        self.features = features
        self.labels = labels
        self.class_num = class_num
        self.fc_infoNCE = nn.Linear(feature_dim, 1).cuda()

    def get_posAndneg(self, features, labels, feature_q_idx=None):
        self.features = features  # [62, 512]
        self.labels = labels  # [62]

        # get the label of q
        q_label = self.labels[feature_q_idx]  # [1] 当前处理的label

        # get the positive sample  # 和当前处理的类别一直的为正样本
        positive_sample_idx = []
        for i, label in enumerate(self.labels):
            if label == q_label and i != feature_q_idx:
                positive_sample_idx.append(i)

        if len(positive_sample_idx) != 0:
            feature_pos = self.features[random.choice(positive_sample_idx)].unsqueeze(0)  # 从里面随机选择一个正样本
        else:
            feature_pos = self.features[feature_q_idx].unsqueeze(0)  # 否则正样本为当前的样本

        # get the negative samples
        negative_sample_idx = []
        for idx in range(features.shape[0]):
            if self.labels[idx] != q_label:
                negative_sample_idx.append(idx)  # 负样本的index 这里的负样本是包含所有不同的类别的

        negative_pairs = torch.tensor([]).cuda()
        # 一共有self.class_num - 1种不同的负样本，一共从所有类别负样本集合中随机选择这么多个样本因此shape应该为[30, 512]
        for i in range(self.class_num - 1):
            negative_pairs = torch.cat((negative_pairs, self.features[random.choice(negative_sample_idx)].unsqueeze(0)))
        if negative_pairs.shape[0] == self.class_num - 1:
            features_neg = negative_pairs
        else:
            raise Exception('Negative samples error!')

        return torch.cat((feature_pos, features_neg))
