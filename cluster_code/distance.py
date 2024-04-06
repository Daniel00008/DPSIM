import torch.nn as nn
import torch.nn.functional as F
eps = 1e-7  # Avoid calculating log(0). Use the small value of float16. It also works fine using 1e-35 (float32).


class KLDiv(nn.Module):
    # Calculate KL-Divergence

    def forward(self, predict, target):
        assert predict.ndimension() == 2, 'Input dimension must be 2'
        target = target.detach()

        # KL(T||I) = \sum T(logT-logI)
        predict += eps
        target += eps
        logI = predict.log()
        logT = target.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld


def kldivergence(predict, target):
    assert predict.ndimension() == 2, 'Input dimension must be 2'
    target = target.detach()

    # KL(T||I) = \sum T(logT-logI)
    predict += eps
    target += eps
    logI = predict.log()
    logT = target.log()
    TlogTdI = target * (logT - logI)
    kld = TlogTdI.sum(1)
    return kld


def cosine_similarity(feature1, feature2):
    """
    :param feature1: N * C
    :param feature2:
    :return:
    """
    feature3 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
    feature4 = F.normalize(feature2)
    distance = feature3.mm(feature4.t())  # 计算余弦相似度
    return distance
