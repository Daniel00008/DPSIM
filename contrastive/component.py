import torch
import torch.nn as nn
import torch.nn.functional as F


from contrastive.utils import infoNCE_g


class Projector(nn.Module):
    def __init__(self, input_dim=256, out_dim=512, class_num=12):
        super(Projector, self).__init__()
        self.gen_c = infoNCE_g(class_num=class_num)  # 原本为12
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, out_dim),
        )

    def cosine_similarity(self, feature, pairs):
        feature = F.normalize(feature)  # F.normalize只能处理两维的数据，L2归一化
        pairs = F.normalize(pairs)
        similarity = feature.mm(pairs.t())  # 计算余弦相似度
        return similarity  # 返回余弦相似度

    def forward(self, x):
        out = self.fc(x)
        return out

    def get_parameters(self):
        return [{"params": self.fc.parameters(), "lr": 1.}]
