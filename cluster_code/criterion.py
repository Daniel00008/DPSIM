import torch.nn as nn
from torch.nn import HingeEmbeddingLoss

from cluster_code.distance import KLDiv


class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0):
        super(KCL, self).__init__()
        self.kld = KLDiv()
        self.hingeloss = nn.HingeEmbeddingLoss(margin)

    def forward(self, prob1, prob2, simi, extent=None):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(
            str(len(prob1)), str(len(prob2)), str(len(simi)))
        kld = self.kld(prob1, prob2)
        output = self.hingeloss(kld, simi)
        return output


class Wt_KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0):
        super(Wt_KCL, self).__init__()
        self.kld = KLDiv()
        self.hingeloss = nn.HingeEmbeddingLoss(margin, reduction='none')

    def forward(self, prob1, prob2, simi, extent):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(
            str(len(prob1)), str(len(prob2)), str(len(simi)))
        kld = self.kld(prob1, prob2)
        output = self.hingeloss(kld, simi)  # [8649]
        size = output.shape[0]
        wt_output = output.dot(extent)
        avg_wt_output = wt_output / size
        return avg_wt_output


class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    eps = 1e-7  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(
            str(len(prob1)), str(len(prob2)), str(len(simi)))

        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(MCL.eps).log_()
        return neglogP.mean()
