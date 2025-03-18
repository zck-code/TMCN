import torch
import torch.nn as nn
from kernel import *
from ddcloss import *

class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def Structure_guided_Contrastive_Loss(self, h_i, h_j, S):
        S_1 = S.repeat(2, 2)
        all_one = torch.ones(self.batch_size*2, self.batch_size*2).to(DEVICE)
        S_2 = all_one - S_1
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim1 = torch.multiply(sim, S_2)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


    def DDCLoss(self, output, hidden, n_clusters):
        hidden_kernel = vector_kernel(hidden, 0.15)
        # ddc1
        loss1 = self.ddc1(output, hidden_kernel, n_clusters)
        # ddc2
        loss2 = self.ddc2(output)
        # ddc3
        loss3 = self.ddc3(output, n_clusters, hidden_kernel)
        return loss1 + loss2 + loss3


    def ddc1(self, output, hidden_kernel, n_clusters):
        nom = th.t(output) @ hidden_kernel @ output
        dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

        nom = _atleast_epsilon(nom)
        dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
        return d

    def ddc2(self, output):
        n = output.size(0)
        return 2 / (n * (n - 1)) * triu(output @ th.t(output))


    def ddc3(self, output, n_clusters, hidden_kernel):
        eye = th.eye(n_clusters, device=DEVICE)
        m = th.exp(-kernel.cdist(output, eye))
        return self.ddc1(m, hidden_kernel, n_clusters)

def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))

def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)
