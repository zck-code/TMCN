import torch as th
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import kernel
import numpy as np


def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))


def at_least_epsilon(X, eps=1E-9):
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


def d_cs(A, K, n_clusters, numerator_epsilon=True, normalize=True):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    num = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(num), -1) @ th.unsqueeze(th.diagonal(num), 0)

    if numerator_epsilon:
        num = at_least_epsilon(num)

    dnom_squared = at_least_epsilon(dnom_squared, eps=1E-9 ** 2)

    d = triu(num / th.sqrt(dnom_squared))
    if normalize:
        d *= (2 / (n_clusters * (n_clusters - 1)))
    return d


def log_d_cs(A, K, n_clusters, numerator_epsilon=True, lamb=0.0):
    num = th.t(A) @ K @ A
    dnom = th.diag(num)

    if numerator_epsilon:
        num = at_least_epsilon(num)

    dnom = at_least_epsilon(dnom)
    log_dnom = th.log(dnom)

    losses = th.log(num) - 0.5 * (1 - lamb) * (log_dnom[None, :] + log_dnom[:, None])
    loss = 2 / (n_clusters * (n_clusters - 1)) * triu(losses)
    return loss


# ======================================================================================================================
# Extra functions
# ======================================================================================================================

def hidden_kernel(net, cfg):
    # Compute pairwise distances
    dist = kernel.cdist(net.hidden, net.hidden)
    # Get kernel width from clustering module
    sigma = net.clustering_module.kernel_width(inputs=net.hidden, distances=dist, assignments=net.output)
    # Return computed kernel matrix and sigma
    return {
        "sigma": sigma,
        "hidden_kernel": kernel.kernel_from_distance_matrix(dist, sigma=sigma),
    }


def encoder_kernels(net, cfg):
    kernels = []
    for x in net.encoder_outputs:
        dist = kernel.cdist(x, x)
        sigma = net.encoder_kernel_width(inputs=x, distances=dist, assignments=net.output)
        kernels.append(kernel.kernel_from_distance_matrix(dist, sigma=sigma))
    return {"encoder_kernels": kernels}


def fused_kernel(net, cfg):
    dist = kernel.cdist(net.fused, net.fused)
    sigma = net.fused_kernel_width(inputs=net.fused, distances=dist, assignments=net.output)
    return {"fused_kernel": kernel.kernel_from_distance_matrix(dist, sigma=sigma)}


def ordered_cmat(labels, pred):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered