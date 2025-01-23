import torch
import numpy as np
import subprocess
from io import BytesIO
import pandas as pd


def spiralTraj(fov, res, nshots=1, npoints=int(7e3)):
    """
    Designs a spiral trajectory in arbitrary parameterization with the desired k-space coverage.

    Parameters
    ----------
    fov : float
        Field of view for trajectory [cm].
    res : float
        Resolution for trajectory [cm].
    nshots : int, optional
        Number of shots to use. Defaults to 1.
    npoints : int, optional
        Number of points to discretize trajectory into. Longer trajectories require more points. Defaults to 7000.

    Returns
    -------
    C : ndarray
        K-space spiral trajectory as a 1D complex array [cm^-1].

    Notes
    -----
    (c) Matthew A. McCready 2024
    """
    nt = fov / (2 * res)
    kmax = 1 / (2 * res)
    dt = 1 / npoints
    t = np.arange(0, 1 + dt, dt)
    C = kmax * t * np.exp(1j * 2 * np.pi * nt * t)
    C = C[:, np.newaxis]
    return C


def rosetteTraj(res, n1=7, n2=5, npoints=int(1e3)):
    """
    Designs a rosette trajectory in arbitrary parameterization with the desired k-space coverage.

    Parameters
    ----------
    fov : float
        Field of view for trajectory [cm].
    res : float
        Resolution for trajectory [cm].
    nshots : int, optional
        Number of shots to use. Defaults to 1.
    npoints : int, optional
        Number of points to discretize trajectory into. Longer trajectories require more points. Defaults to 7000.

    Returns
    -------
    C : ndarray
        K-space spiral trajectory as a 1D complex array [cm^-1].

    Notes
    -----
    (c) Matthew A. McCready 2024
    """
    t = np.linspace(0, 0.5, num=npoints)
    kmax = 1 / (2 * res)
    C = kmax * np.sin(2 * np.pi * n1 * t) * np.exp(1j * 2 * np.pi * n2 * t)
    C = C[:, np.newaxis]
    return C


def tensorInterp(y, x, xx):
    """
    Performs 1D linear interpolation of PyTorch tensors allowing backpropogation through any of the input parameters.
    Interpolates from N points to M points over D axes. y, x, and xx must be on the same device.

    Parameters
    ----------
    y : tensor
        Dependent data to interpolate. Tensor size (N x D) or (N).
    x : tensor
        Original independent variable values. Tensor size (N).
    xx : tensor
        New independent variable values to interpolate at. Must lie within range of x. Tensor size (M).

    Returns
    -------
    yy : tensor
        Data y interpolated to points xx. Tensor size (M x D) or (M).

    Notes
    -----
    (c) Matthew A. McCready 2024
    """
    device = y.device  # Get device to use
    # get indices of elements in xx and their bounding elements in x
    # col 0: contains indices of xx elements (1 to M)
    # col 1: contains indices of elements in x surrounding element of xx from col 0
    # rows 0 to M-1: left bounding element
    # rows M to 2M-1: right bounding element
    indices = torch.zeros((2 * xx.numel(), 2), dtype=int, device=device) - 1
    indices[:xx.numel(), 0] = torch.arange(xx.numel())  # fill in col 0
    indices[xx.numel():, 0] = torch.arange(xx.numel())
    idx = torch.searchsorted(x.detach(), xx.detach(), right=True)  # get bounding element indices from xx
    indices[:xx.numel(), 1] = idx - 1  # left bounding
    indices[xx.numel():, 1] = idx  # right bounding
    dx = x[indices[xx.numel():, 1]] - x[indices[:xx.numel(), 1]]  # step sizes
    values = torch.zeros((2 * xx.numel()), device=device)  # calculate interpolation coefficients
    values[:xx.numel()] = (x[indices[xx.numel():, 1]] - xx) / dx  # left coeff
    values[xx.numel():] = (xx - x[indices[:xx.numel(), 1]]) / dx  # right coeff
    # Do interpolation
    if y.dim() > 1:  # pad empty dimension so Torch is happy
        yy = values[:xx.numel()][:, np.newaxis] * y[indices[:xx.numel(), 1], :] + values[xx.numel():][:, np.newaxis] * y[indices[xx.numel():, 1], :]
    else:
        yy = values[:xx.numel()] * y[indices[:xx.numel(), 1]] + values[xx.numel():] * y[indices[xx.numel():, 1]]

    return yy


def get_free_gpu():
    """
    Determines index of GPU with the largest available memory and returns.
    usage: idx = get_free_gpu()

    Returns
    -------
    idx : int
        Index of GPU with the largest available memory.

    Notes
    -----
    (c) Matthew A. McCready 2024
    """
    print('Selecting GPU with largest free memory...')
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx
