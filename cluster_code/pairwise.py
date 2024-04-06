import torch

"""
    x = tensor([[0.2000, 0.3000, 0.5000],
                [0.1000, 0.4000, 0.7000],
                [0.6000, 0.8000, 0.9000]])
                
    x1 = tensor([[0.2000, 0.3000, 0.5000],
                 [0.1000, 0.4000, 0.7000],
                 [0.6000, 0.8000, 0.9000],
                 [0.2000, 0.3000, 0.5000],
                 [0.1000, 0.4000, 0.7000],
                 [0.6000, 0.8000, 0.9000],
                 [0.2000, 0.3000, 0.5000],
                 [0.1000, 0.4000, 0.7000],
                 [0.6000, 0.8000, 0.9000]])
    x2 = tensor([[0.2000, 0.3000, 0.5000],
                 [0.2000, 0.3000, 0.5000],
                 [0.2000, 0.3000, 0.5000],
                 [0.1000, 0.4000, 0.7000],
                 [0.1000, 0.4000, 0.7000],
                 [0.1000, 0.4000, 0.7000],
                 [0.6000, 0.8000, 0.9000],
                 [0.6000, 0.8000, 0.9000],
                 [0.6000, 0.8000, 0.9000]])
    x1和x2各按照顺序取一个就可以达到pairwise枚举的效果
"""
def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x  [80, 31]
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)  # 行复制平方倍 [6400, 31]
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))  # 列复制80倍 shape = (80, 31*80=2480)
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        # dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


def Class2Simi(x, mode='cls', mask=None):
    # Convert class label to pairwise similarity
    n = x.nelement()
    assert (n - x.ndimension() + 1) == n, 'Dimension of Label is not right'
    expand1 = x.view(-1, 1).expand(n, n)
    expand2 = x.view(1, -1).expand(n, n)
    out = expand1 - expand2
    out[out != 0] = -1  # dissimilar pair: label=-1
    out[out == 0] = 1  # Similar pair: label=1
    if mode == 'cls':
        out[out == -1] = 0  # dissimilar pair: label=0
    if mode == 'hinge':
        out = out.float()  # hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out


def PairEnumEntorpy_normalize(h, n_class):
    """

    :param h: torch.tensor([1,2,3,4,……]) 每一个元素都为熵
    :param n_class:
    :return:
    """
    assert h.ndimension() == 1, 'Input dimension must be 1'
    n = h.nelement()
    h = h.expand(n, n)
    x1 = h.T
    x1 = x1.reshape(1, -1).squeeze()
    x2 = h.reshape(1, -1).squeeze()
    x3 = x1 + x2
    H_max = torch.tensor(n_class).log()
    x3 = (2 * H_max - x3) / 2 * H_max
    return x3


def probmatrix2simextent(probMx, sim_label):
    """

    :param probMx: n_t × n_cls
    :param sim_label: 1或者-1
    :return: sim_extent unsim_extent
    """
    m2, m1 = PairEnum(probMx)
    hadamard = m1.mul(m2)
    sim_extent = torch.sum(hadamard, dim=1)
    sig = sim_extent * sim_label  # 使不相似的前面有负号  相似的程度直接为正  不相似的程度前面有一个负号
    sim_label = (sim_label - 1) * (-1/2)  # 相似为0， 不相似为1
    return sim_label + sig



