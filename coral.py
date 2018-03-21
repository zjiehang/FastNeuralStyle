import numpy as np
# import torch

### NumPy implementation

def matSqrt_numpy(x):
    U,D,V = np.linalg.svd(x)
    result = U.dot(np.diag(np.sqrt(D))).dot(V.T)
    return result

def coral_numpy(source, target):
    n_channels = source.shape[-1]

    source = np.moveaxis(source, -1, 0)  # HxWxC -> CxHxW
    target = np.moveaxis(target, -1, 0)  # HxWxC -> CxHxW

    source_flatten = source.reshape(n_channels, source.shape[1]*source.shape[2])
    target_flatten = target.reshape(n_channels, target.shape[1]*target.shape[2])

    source_flatten_mean = source_flatten.mean(axis=1, keepdims=True)
    source_flatten_std = source_flatten.std(axis=1, keepdims=True)
    source_flatten_norm = (source_flatten - source_flatten_mean) / source_flatten_std

    target_flatten_mean = target_flatten.mean(axis=1, keepdims=True)
    target_flatten_std = target_flatten.std(axis=1, keepdims=True)
    target_flatten_norm = (target_flatten - target_flatten_mean) / target_flatten_std

    source_flatten_cov_eye = source_flatten_norm.dot(source_flatten_norm.T) + np.eye(n_channels)
    target_flatten_cov_eye = target_flatten_norm.dot(target_flatten_norm.T) + np.eye(n_channels)

    source_flatten_norm_transfer = matSqrt_numpy(target_flatten_cov_eye).dot(np.linalg.inv(matSqrt_numpy(source_flatten_cov_eye))).dot(source_flatten_norm)
    source_flatten_transfer = source_flatten_norm_transfer * target_flatten_std + target_flatten_mean

    coraled = source_flatten_transfer.reshape(source.shape)
    coraled = np.moveaxis(coraled, 0, -1)  # CxHxW -> HxWxC

    return coraled
