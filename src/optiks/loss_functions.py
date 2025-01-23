from functools import reduce
import numpy as np
import torch
from optiks.utils import tensorInterp


def custom_loss(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    Creates custom loss function from supplied terms in 'params' with weightings in 'weights'. Calls each term and sums
    together to calculate total loss.

    Parameters
    ----------
    v : tensor
        K-space speed as a function of arc-length v(s) [cm^-1 / ms]
    T : tensor float
        Duration of the gradient waveform [ms]
    g : tensor
        Gradient waveform size Nx2 or Nx3 [G/cm]
    dt : float
        Gradient waveform time sampling [ms]
    smax : float
        Maximum allowed slew-rate [G/cm/ms]
    weights : dict
        Dictionary containing key and weight pairs for each term to be included in loss function. See specific loss term
        functions below for appropriate keys.
    rv : bool, optional
        Rotationally variant solution flag. Defaults to False.
    params : dict
        Dictionary containing key and value pairs for each term to be included in loss function. Value contains optional
        parameters for evaluating each term. See specific term functions below for appropriate keys and parameters. Also
        contains key 'terms' whose value is a list of function handles to term functions to be included.

    Returns
    -------
    loss : tensor float
        Total calculated loss with tree for backprop.
    terms : list
        List of evaluated terms (with weightings) in order of params['terms']. Tensor floats.

    Notes
    -----

    """
    terms = [None] * len(params['terms'])
    for i, func in enumerate(params['terms']):
        terms[i] = func(v, T, g, dt, smax, weights, rv, params)
    return sum(terms), terms


def time_min(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    Minimimum time (duration) loss function term.

    Parameters
    ----------
    T : tensor float
        Duration of the gradient waveform [ms]
    weights : dict
        Uses key 'time'
    """
    term = weights['time'] * T
    return term


def time_bound(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    Bounded time (duration) loss function term.

    Constrains time duration of gradient waveform to a maximum value by use of the log-barrier function.

    Parameters
    ----------
    T : tensor float
        Duration of the gradient waveform [ms]
    weights : dict
        Uses key 'time'
    params : dict
        Uses key 'bound', with value giving maximum allowed duration of waveform [ms].

    Notes
    -----
    Should not be used with time_min, use one or the other. If the initial solution waveform duration is greater than
    params['bound'] allows, the loss function will evaluate to Inf and gradient descent will not progress. In this case
    either increase the derate factor or allow for more time. The maximum time may be violated during gradient descent
    in some cases. If this occurs try running again with a smaller learning rate or increase weights['time'].
    """
    term = -weights['time'] * torch.log(torch.relu(params['bound'] - T))
    return term


def slew_lim(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    Slew-rate limit loss function term.

    Constrains the slew-rate of gradient waveform to a maximum value by use of the "leaky" log-barrier function (README)

    Parameters
    ----------
    smax : float
        Maximum allowed slew-rate [G/cm/ms]
    weights : dict
        Uses key 'slew'
    """
    if not rv:
        sr = torch.norm(torch.diff(g, dim=0), dim=1) / dt
    else:
        sr = torch.abs(torch.diff(g, dim=0) / dt)
    dx = torch.tensor(0.0002)
    sr[:3] = sr[:3] + smax/4
    term = weights['slew'] * (-torch.sum(torch.log(torch.relu(smax - sr[sr < (smax - dx)])))
                        + torch.sum(sr[sr >= (smax - dx)] / dx - torch.log(dx) + (1 - smax / dx))) / sr.detach().numel()
    return term


def pns_lim(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    PNS limit loss function term.

    Constrains the PNS threshold of gradient waveform as calculated using the IEC 60601-2-33 model to a maximum value by
    use of the "leaky" log-barrier function.

    Parameters
    ----------
    weights : dict
        Uses key 'pns'
    params : dict
        Uses key 'pns' with value list [Pthresh, r, c, a] the maximum allowed PNS threshold, rheobase, chronaxie time,
        and the effective coil length as in IEC 60601-2-33.

    Notes
    -----
    Use for GE systems. r, c, and a are hardware specific parameters acquired from config files.
    """
    dtu = dt * 1e-3
    Smin = params['pns'][1] / params['pns'][3]
    tp = torch.arange(0, dtu * (g.shape[0] - 2) + dtu / 10, dtu, dtype=torch.float64, device=g.device)

    # Fourier approach
    H = torch.cat((torch.zeros_like(tp), dtu * params['pns'][2] / (params['pns'][2] + tp) ** 2 / Smin))
    S = torch.diff(g.T * 0.01, dim=1) / dtu
    pd = S.shape[1] // 2
    S = torch.nn.functional.pad(S, (pd, pd))
    H = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(H)))
    S = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(S, dim=1)), dim=1)
    P = H * S

    stim = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(P, dim=1)), dim=1)[:, pd:-pd]

    stim = 100 * torch.norm(stim.squeeze(), dim=0)
    dp = torch.tensor(0.00005)
    term = weights['pns'] * (torch.sum(torch.relu(-torch.log(params['pns'][0] - stim[stim < (params['pns'][0] - dp)]) + 0.7))
        + torch.sum(stim[stim >= (params['pns'][0] - dp)] / dp - torch.log(dp) + (1 - params['pns'][0] / dp))) / stim.detach().numel()

    return term


def safemodel_lim(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    "SAFE" model PNS limit loss function term.

    Constrains the PNS threshold of gradient waveform as calculated using the "SAFE" model to a maximum value by use of
    the "leaky" log-barrier function.

    Parameters
    ----------
    weights : dict
        Uses key 'safemodel'
    params : dict
        Uses key 'safemodel' with value list [Pthresh, hardware] the maximum allowed PNS threshold, and the
        SimpleNamespace object defining SAFE model hardware parameters as used in PulseSeq.

    Notes
    -----
    Use for Seimens systems. SAFE model is calculated as in [refs].

    Should not be used with pns_lim, use one or the other.

    ATTENTION: The "SAFE" model is NOT rotationally independent, and furthermore its non-linear calculation makes it
    difficult to determine some limiting waveform orientation. For these reasons this term constrains PNS threshold as
    calculated by the "SAFE" model only for the SPECIFIC ORIENTATION DEFINED BY THIS TRAJECTORY. That is, rotating the
    resulting waveform CAN RESULT IN PNS VIOLATIONS as calculated by "SAFE". It is suggested that this risk may be
    reduced by using a conservative Pthresh, or by using the pns_lim function and evaluating with SAFE later to ensure
    the desired levels were met.

    This package has no dependency on PulseSeq. However, it is assumed that the convention for hardware PNS parameters
    developed there is used here as well.
    """
    hw = params['safemodel'][1]
    t_steps = torch.arange(0, g.shape[0] - 1, device=g.device)
    stim = 0
    ax = ['x', 'y', 'z']
    for i, fld in enumerate(ax[:g.shape[1]]):
        coil = getattr(hw, fld)

        alpha = dt / torch.tensor([coil.tau1 + dt,
                                   coil.tau2 + dt,
                                   coil.tau3 + dt], device=g.device)[np.newaxis, :]

        h = (1 - alpha) ** t_steps[:, np.newaxis]

        H = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(
            torch.cat((torch.zeros_like(h), h), dim=0),
            dim=0), dim=0), dim=0)
        S = torch.diff(g[:, i] * 0.01, dim=0)[:, np.newaxis] / (dt * 1e-3)
        S = alpha * torch.cat((S, torch.abs(S), S), dim=1)
        pd = S.shape[0] // 2
        S = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(
            torch.nn.functional.pad(S, (0, 0, pd, pd)),
            dim=0), dim=0), dim=0)
        P = H * S
        p = torch.real(torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(P, dim=0), dim=0), dim=0)[pd:-pd, :])
        stim += ((coil.a1 * torch.abs(p[:, 0]) + coil.a2 * p[:, 1] + coil.a3 * torch.abs(p[:, 2])) / coil.stim_limit * coil.g_scale * 100) ** 2

    stim = torch.sqrt(stim)
    dp = torch.tensor(0.00005)
    term = weights['pns'] * (torch.sum(torch.relu(-torch.log(params['safemodel'][0] - stim[stim < (params['safemodel'][0] - dp)]) + 0.7))
        + torch.sum(stim[stim >= (params['safemodel'][0] - dp)] / dp - torch.log(dp) + (1 - params['safemodel'][0] / dp))) / stim.detach().numel()
    return term


def freq_min(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    Minimizes power deposited by waveform into specified frequency bands. Used for mechanical resonance avoidance.

    Parameters
    ----------
    weights : dict
        Uses key 'frequency'
    params : dict
        Uses key 'frequency' with value list of frequncy band edge pairs [[low1, high1], [low2, high2], ...] in kHz.
    """
    nf = int(g.detach().numel() / 2 * 10)
    gf = dt * torch.fft.rfft(g, n=nf, dim=0)
    freq = torch.fft.rfftfreq(nf, d=dt)
    idx = np.argwhere(reduce(lambda ind, fed: np.logical_or(ind, np.logical_and(fed[0] <= freq, freq <= fed[1])),
                             params['frequency'], np.logical_and(params['frequency'][0][0] <= freq, freq <= params['frequency'][0][1])))

    term = weights['frequency'] * torch.norm(gf[idx])

    return term


def acoustic_min(v, T, g, dt, smax, weights, rv=False, params=None):
    t = torch.arange(0, 50//dt, device=g.device) * dt
    H = torch.cat((torch.zeros((t.numel() - 1, 2), device=g.device), dt * tensorInterp(params['acoustic'][0], params['acoustic'][1], t.detach()[:-1])))
    pd = (H.shape[0] - g.T.shape[1]) // 2
    G = torch.nn.functional.pad(g.T, (pd, pd)).T
    H = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(H[:-1], dim=0), dim=0), dim=0)
    G = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(G, dim=0), dim=0), dim=0)
    A = torch.abs(H * G)
    term = dt * weights['acoustic'] * torch.norm(A)**2
    return term


def acoustic_freq_min(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    Minimizes waveform power spectrum weighted by acoustic response function.

    Parameters
    ----------
    weights : dict
        Uses key 'acousticfreq'
    params : dict
        Uses key 'acousticfreq' with value list []
    """
    pd = g.shape[0]
    G = torch.nn.functional.pad(g.T, (pd, pd)).T
    freq = torch.fft.rfftfreq(G.shape[0], d=dt, device=g.device)
    G = dt * torch.fft.rfft(G, n=G.shape[0], dim=0)
    H = torch.zeros_like(G)
    H[freq < params['acousticfreq'][1][0], :] = params['acousticfreq'][0][0, :]
    H[freq > params['acousticfreq'][1][-1], :] = params['acousticfreq'][0][-1, :]
    freqidx = torch.logical_and(freq >= params['acousticfreq'][1][0], freq <= params['acousticfreq'][1][-1])
    larf_intp = tensorInterp(params['acousticfreq'][0], params['acousticfreq'][1], freq[freqidx])
    H[freqidx] = larf_intp
    A = torch.abs(H * G)
    term = dt * weights['acoustic'] * torch.norm(A)
    return term


def smooth_reg(v, T, g, dt, smax, weights, rv=False, params=None):
    """
    Performs minimum energy regularization on the first difference of k-space speed as a function of arc-length.

    Parameters
    ----------
    weights : dict
        Uses key 'regularization'
    v : tensor
        K-space speed as a function of arclength v(s) [cm^-1 / ms]
    """
    term = weights['regularization'] * torch.norm(torch.diff(v)) ** 2

    return term
