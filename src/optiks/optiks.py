import time
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("webagg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os import environ
import warnings
from tqdm import tqdm
from src.optiks.options import *
from src.optiks.utils import *
from src.optiks.loss_functions import custom_loss


def optiks(C, hwopts=HardwareOpts(), dsopts=DesignOpts(), svopts=SolverOpts()):
    """
    Given a k-space trajectory C(p), gradient and slew constraints, and a loss function definition. This function will
    return a gradient waveform that will meet these constraints while jointly minimizing total trajectory time and
    additional properties defined by the loss function which can be applied in the time domain. Some examples include
    minimizing power deposition in mechanical resonance bands, limiting PNS along readout, or reducing eddy currents.

    Parameters
    ----------
    C : ndarray
        The curve in k-space given in any parametrization [1/cm]. Accepts Nx2 or Nx3 real matrix.
    hwopts : HardwareOpts
        Hardware options for waveform design. Includes maximum gradient and slew rate, initial and final gradient
        values, and time sampling. See options.py
    dsopts : DesignOpts
        Design options for waveform design. Includes loss function terms and parameters, and rotationally variant flag.
        See options.py
    svopts : SolverOpts
        Solver options for waveform design. Includes arclength sampling, number of gradient descent steps, learning
        rate, and initial solution parameters. See options.py

    Returns
    -------
    Cnew : ndarray
    t : float
        Total time for trajectory [ms].
    g : ndarray
        Minimum loss gradient waveform for each required axis [G/cm]. Size M x D.
    s : ndarray
        Minimum loss slew rate waveform for each required axis [G/cm/ms]. Size M x D.
    ginit : ndarray
        Time optimal gradient waveform [G/cm]. Size M' x D.
    sinit : ndarray
        Time optimal slew rate waveform [G/cm/ms]. Size M' x D.
    g_last : ndarray
        Final descent step gradient waveform [G/cm]. Size M'' x D.

    Notes
    -----
    (c) Matthew McCready 2024.


    :param params:  Tuple of additional parameters necessary for loss function (e.g. mechanical resonance band frequency
                    edges, PNS filters, etc.). Default None.
    :param rv:      Rotationally variant flag. If true, the design is rotationally variant, optimized for orientation of
                    input trajectory C. Default False.
    :param initsol: Initialization for gradient descent. If None, the time optimal waveform is used with the input
                    derating factor. Default None.

    """
    # Unpacking inputs==================================================================================================
    g0 = hwopts.g0  # initial gradient (G/cm)
    gfin = hwopts.gfin  # final gradient (G/cm)
    gmax = hwopts.gmax  # maximum allowed gradient (G/cm)
    smax = hwopts.smax  # maximum allowed slew rate (G/cm/ms)
    dt = hwopts.dt  # time sampling for returned waveform (ms)
    weights = dsopts.weights  # list of weights for terms in custom loss function
    params = dsopts.params  # dictionary of terms and parameters for custom loss function
    rv = dsopts.rv  # flag for rotationally variant waveform design
    ds = svopts.ds  # arc-length sampling for design (cm^-1)
    maxiter = svopts.maxiter  # maximum allowed number of gradient descent steps
    count = svopts.count  # number of iterations to display progress after
    derate = svopts.derate  # factor to derate initial (time optimal) solution v(s) by before optimizing
    initsol = svopts.initsol  # custom initial waveform solution - replaces time optimal solution (G/cm)
    gamma = 4.258  # gyromagnetic ratio [Cycles/ms/G]

    # Setting up arclength parameterization and initializing solution===================================================

    # Represent the curve using spline with parametrization p
    # Miki uses an arbitrary parametrization, but I've had better success with p as an initial arclength calculation
    Cp = np.diff(C, axis=0)
    Cp = np.vstack((Cp, Cp[-1]))
    p = cumulative_trapezoid(np.linalg.norm(Cp, axis=1), axis=0, initial=0)
    Lp = p[-1]
    PP = CubicSpline(p, C)

    # Upsample curve (x10) for gradient accuracy
    dp = np.amin(np.diff(p)) / 10
    p = np.arange(0, Lp, dp)
    CC = PP(p)

    # Find length of the curve
    Cp = np.diff(CC, axis=0) / dp  # tangent curve as function of p
    Cp = np.vstack((Cp, Cp[-1]))
    s_of_p = cumulative_trapezoid(np.linalg.norm(Cp, axis=1), axis=0, initial=0) * dp  # arclength as function of p
    L = s_of_p[-1]  # Length of curve

    # Decide ds and compute st for the first point
    stt0 = gamma * smax  # Always assumes the first point is max slew
    st0 = stt0 * dt / 2  # Start at half the gradient for accuracy close to g=0
    s0 = st0 * dt
    if ds is None:
        ds = s0 / 1.5  # Smaller step size for numerical accuracy

    s = np.arange(0, L, ds)
    s_half = np.arange(0, L, ds / 2)  # for RK integration

    if g0 is None:
        g0 = 0  # assume start at gradient of 0

    p_of_s_half = interp1d(s_of_p, p, kind='cubic')(s_half)  # for RK integration
    p_of_s = p_of_s_half[::2]

    # Get initial solution (init) from time optimal method, as well as forbidden line curve (phi), and curvature (k)
    # Use s0/5 for ds in init solution better accuracy
    init, phi, k, s_half_init = initSolution(C, g0, gfin, gmax, smax, dt, s0/5, rv=rv)
    s_half_init = np.arange(0, L, s0/10)
    s_init = np.arange(0, L, s0/5)

    # interpolate initial solution back to ds and ds_half sampling
    st0 = init[0]  # fastest reachable initial point
    init_interp = interp1d(s_init, init, kind='linear')(s[s <= s_init[-1]])  # interpolating to optimization sampling
    init = np.hstack((init_interp, init[-1]*np.ones_like(s[s > s_init[-1]])))
    phi_interp = interp1d(s_half_init, phi, kind='linear')(s_half[s_half <= s_half_init[-1]])
    phi = np.hstack((phi_interp, phi[-1]*np.ones_like(s_half[s_half > s_half_init[-1]])))
    phi = phi[::2] + 1e-6  # avoid numerical issues when inverting sigmoid
    phi[0] = st0 + 1e-6  # Ensure initial value is used

    # If a solution was passed as an argument use this as initial v(s)
    if not initsol is None:
        initv_of_t = np.linalg.norm(np.diff(initsol, axis=0), axis=1) / dt
        inits_of_t = cumulative_trapezoid(initv_of_t * dt, axis=0, initial=0)
        init = interp1d(inits_of_t, initv_of_t, kind='linear')(s[s <= inits_of_t[-1]])
        init = np.hstack((init, init[-1]*np.ones_like(s[s > inits_of_t[-1]])))
        init[0] = st0

    # Preparing optimization variable and moving variables to Torch=====================================================

    environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # make GPU ID match that of nvitop
    device = torch.device(get_free_gpu() if torch.cuda.is_available() else 'cpu')  # automatically choose emptiest GPU

    weights['time'] = weights['time'] / np.trapz(ds / init)  # normalizing timing weight by time optimal duration

    # Initializing optimization variable (nu):
    # nu is passed through a sigmoid as v(s) = vmax(s) * sigmoid(nu(s))
    # we must derate the time optimal solution or initialize from zeros
    if derate is None:
        nu = torch.zeros(init.size, device=device)  # using constant initialization
    else:
        nu = derate * init
        nu = torch.tensor(np.log(nu / (phi - nu)), device=device)  # using de-rated time-optimal solution
    nu.requires_grad = True

    # Moving variables to device
    phi = torch.tensor(phi, device=device)
    s = torch.tensor(s, device=device)
    p_of_s = torch.tensor(p_of_s, device=device)
    CC = torch.tensor(CC, device=device)
    p = torch.tensor(p, device=device)
    zro = torch.tensor(0, device=device)  # padding used later
    g0 = torch.tensor(g0*np.ones((1, C.shape[1])), device=device)
    if not gfin is None:
        gfin = torch.tensor(gfin*np.ones((1, C.shape[1])), device=device)
    if 'acoustic' in params.keys():
        params['acoustic'][0] = params['acoustic'][0].to(device=device)
        params['acoustic'][1] = params['acoustic'][1].to(device=device)
    elif 'acousticfreq' in params.keys():
        params['acousticfreq'][0] = params['acousticfreq'][0].to(device=device)
        params['acousticfreq'][1] = params['acousticfreq'][1].to(device=device)

    # setting length of time-domain vector. Changing length with constant time sampling during optimization leads to
    # memory leak issue in PyTorch. Workaround is to choose a length s.t. sampling will always be smaller than dt.
    nscale = 1.5 * np.trapz(ds / init) if 'bound' not in params.keys() else 1.1 * params['bound']
    tsamp = torch.arange(0, 1, dt / nscale, dtype=torch.float64, device=device)  # time sampling used later
    if tsamp.numel() % 2 != 0:
        tsamp = tsamp[:-1]

    # initializing optimizer, minimum loss, and loss storage
    optimizer = torch.optim.AdamW([nu], lr=1e-4)
    lossvec = np.zeros(maxiter // count)
    lossterms = np.zeros((maxiter // count, len(params['terms'])))
    minloss = np.inf

    # Performing gradient descent for maxiter steps=====================================================================

    for i in tqdm(range(maxiter)):
        optimizer.zero_grad()

        # Address velocity constraint
        v = phi * (1 / (1 + torch.exp(-nu)))  # scale to velocity constraint 1

        # Move to time domain and get gradient waveform
        t_of_s = torch.cumulative_trapezoid(1 / v * ds)
        t_of_s = torch.hstack((zro, t_of_s))  # Time as function of arc-length
        t = tsamp * t_of_s[-1]  # evenly spaced time
        dt_temp = t[1].detach().cpu()  # time sampling on this iteration.
        # Note: use of a set time sampling varies size of t each iteration. This causes memory leakage due to some
        # internal PyTorch error. Solution not found, so sampling is allowed to change and length of t is held constant.
        s_of_t = tensorInterp(s, t_of_s, t)  # arc-length as function of evenly spaced time
        s_of_t = torch.clip(s_of_t, max=s[-1]*0.999)
        p_of_t = tensorInterp(p_of_s, s, s_of_t)  # initial parameter as function of evenly spaced time
        p_of_t = torch.clip(p_of_t, max=p[-1]*0.999)
        Cnew = tensorInterp(CC, p, p_of_t)  # trajectory as function of evenly spaced time
        g = torch.diff(Cnew, dim=0) / gamma / dt_temp  # time domain gradient
        if gfin is None:
            g = torch.vstack((g0, g))  # pad initial value constraint
        else:
            g = torch.vstack((g0, g, gfin))  # pad initial and final value constraints

        loss, terms = custom_loss(v, t_of_s[-1], g, dt_temp, smax, weights, rv=rv, params=params)  # calculate loss
        if loss < minloss:
            minloss = loss
            best = nu
        loss.backward()  # backprop
        optimizer.step()  # GD step
        if i % count == 0:  # record loss statistics
            lossvec[i // count] = loss.detach().cpu().numpy()
            lossterms[i // count] = torch.tensor(terms).detach().cpu()

    # Collect waveforms=================================================================================================

    # Get g(t) in dt sampling for final iteration (g_last)
    t_of_s = torch.cumulative_trapezoid(1 / v * ds)
    t_of_s = torch.hstack((zro, t_of_s))  # Time as function of arc-length
    t = torch.arange(0, 1, dt / t_of_s[-1].detach(), dtype=torch.float64, device=device) * t_of_s[-1]
    s_of_t = tensorInterp(s, t_of_s, t)
    p_of_t = tensorInterp(p_of_s, s, s_of_t)
    Cnew = tensorInterp(CC, p, p_of_t)
    g_last = torch.diff(Cnew, dim=0) / gamma / dt

    # Calculate time domain waveforms for minimum loss (and time optimal) solution and their power spectra
    v = phi * (1 / (1 + torch.exp(-best)))  # scale to velocity constraint 1
    t_of_s = torch.cumulative_trapezoid(1 / v * ds)
    t_of_s = torch.hstack((zro, t_of_s))  # Time as function of arc-length
    t = torch.arange(0, 1, dt/t_of_s[-1].detach(), dtype=torch.float64, device=device) * t_of_s[-1]
    s_of_t = tensorInterp(s, t_of_s, t)
    p_of_t = tensorInterp(p_of_s, s, s_of_t)
    Cnew = tensorInterp(CC, p, p_of_t)
    g = torch.diff(Cnew, dim=0) / gamma / dt
    gf = torch.fft.rfft(g, dim=0, n=g.detach().shape[0] * 10) * dt
    freq = torch.fft.rfftfreq(g.detach().shape[0] * 10, d=dt)
    ktrue = gamma * cumulative_trapezoid(g.detach().cpu().numpy(), axis=0, initial=0) * dt

    init = torch.tensor(init, device=device)
    t_of_s = torch.cumulative_trapezoid(1 / init * ds)
    t_of_s = torch.hstack((zro, t_of_s))
    t_init = torch.arange(0, 1, dt / t_of_s[-1].detach(), dtype=torch.float64, device=device) * t_of_s[-1]
    s_of_t = tensorInterp(s, t_of_s, t_init)
    p_of_t = tensorInterp(p_of_s, s, s_of_t)
    Cinit = tensorInterp(CC, p, p_of_t)
    g_init = torch.diff(Cinit, axis=0) / gamma / dt
    gf_init = torch.fft.rfft(g_init, dim=0, n=g_init.detach().shape[0] * 10) * dt
    freq_init = torch.fft.rfftfreq(g_init.detach().shape[0] * 10, d=dt)

    tlim = torch.maximum(t_init[-1], t[-1]).detach().cpu()

    # Plotting results==================================================================================================
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))

    # Plot loss statistics
    ax[0, 0].semilogy(np.arange(i//count+1) * count, lossvec[:i//count+1])
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].set_xlabel('Iteration')
    ax[0, 1].semilogy(np.arange(i//count+1) * count, lossterms[:i//count+1, :], label=weights.keys())
    ax[0, 1].set_ylabel('Loss')
    ax[0, 1].set_xlabel('Iteration')
    ax[0, 1].legend()

    # Plot time domain waveforms
    dir = ['X', 'Y', 'Z']
    for i in range(C.shape[1]):
        ax[1, i].plot(t_init[:-1].detach().cpu(), g_init[:, i].detach().cpu(), label="Unsafe")
        ax[1, i].plot(t[:-1].detach().cpu(), g[:, i].detach().cpu(), label="Optimized")
        ax[1, i].legend()
        ax[1, i].set_xlabel('Time (ms)')
        ax[1, i].set_ylabel(dir[i] + '-Gradient Amplitude (G/cm)')
        ax[1, i].set_xlim(left=0, right=tlim)

    # Plot waveform slew-rates
    sr = np.abs(np.diff(g.detach().cpu().numpy(), axis=0)) / dt
    for i in range(C.shape[1]):
        ax[2, 0].plot(t[:-2].detach().cpu(), sr[:, i], label=dir[i])
    if not rv:
        ax[2, 0].plot(t[:-2].detach().cpu(), np.linalg.norm(sr, axis=1), label="Mag")
    ax[2, 0].set_xlabel('Time (ms)')
    ax[2, 0].set_ylabel('Slew-Rate (G/cm/ms)')
    ax[2, 0].legend()
    ax[2, 0].set_xlim(left=0, right=tlim - 2 * dt)

    # Plot resulting trajectory
    if C.shape[1] == 2:
        ax[2, 1].plot(ktrue[:, 0], ktrue[:, 1], label='result')
        ax[2, 1].plot(Cnew[:, 0].detach().cpu(), Cnew[:, 1].detach().cpu(), ls='--', color='black', label='target')
        ax[2, 1].axis('equal')
        ax[2, 1].set_xlabel('k$_x$ (cm$^{-1}$)')
        ax[2, 1].set_ylabel('k$_y$ (cm$^{-1}$)')
        ax[2, 1].legend()
        plt.grid()
    else:
        ax[2, 1].remove()
        ax[2, 1] = fig.add_subplot(4, 3, 8, projection='3d')
        ax[2, 1].plot(ktrue[:, 0], ktrue[:, 1], ktrue[:, 2], label='result')
        ax[2, 1].plot(Cnew[:, 0].detach().cpu(), Cnew[:, 1].detach().cpu(), Cnew[:, 2].detach().cpu(), ls='--',
                      color='black', label='target')
        ax[2, 1].set_aspect('auto')
        ax[2, 1].set_xlabel('k$_x$ (cm$^{-1}$)')
        ax[2, 1].set_ylabel('k$_y$ (cm$^{-1}$)')
        ax[2, 1].set_zlabel('k$_z$ (cm$^{-1}$)')
        ax[2, 1].legend()
        plt.grid()

    # Plot design parameters
    if 'frequency' in params.keys():
        fig, axf = plt.subplots(1, C.shape[1], figsize=(15, 5))
        for i in range(C.shape[1]):
            axf[i].plot(freq_init.detach().cpu().numpy(), np.abs(gf_init[:, i].detach().cpu().numpy()), label="Unsafe")
            axf[i].plot(freq.detach().cpu().numpy(), np.abs(gf[:, i].detach().cpu().numpy()), label="Optimized")
            h = np.diff(axf[0].get_ylim())[0]
            axf[i].set_xlim((0.4, 8))
            for j in range(len(params['frequency'])):
                rect = patches.Rectangle((params['frequency'][j][0], 0), params['frequency'][j][1] - params['frequency'][j][0], h, edgecolor='r', facecolor='r', alpha=0.3)
                axf[i].add_patch(rect)
            axf[i].set_ylim((0, axf[0].get_ylim()[1]))
            axf[i].legend()
            axf[i].set_xlabel('Frequency (kHz)')
            axf[i].set_ylabel(dir[i] + '-Gradient Power (a.u.)')

    if 'pns' in params.keys():
        fig, axp = plt.subplots(1, 1, figsize=(7.5, 5))
        dtu = dt * 1e-3
        Smin = params['pns'][1] / params['pns'][3]
        tp = torch.arange(0, dtu * (g_init.shape[0] - 2), dtu, dtype=torch.float64, device=device)
        h = torch.flip(dtu * params['pns'][2] / (params['pns'][2] + tp) ** 2 / Smin, dims=[0])
        S = torch.diff(g_init.T * 0.01, dim=1)[:, np.newaxis, :] / dtu
        stim_all = torch.nn.functional.conv1d(S, h[np.newaxis, np.newaxis, :], padding=S.shape[2])[:, :,
                   :S.shape[2]]
        stim_all = torch.norm(stim_all.squeeze(), dim=0)
        axp.plot(t_init[:-2].detach().cpu(), stim_all.detach().cpu(), label="unsafe")
        tp = torch.arange(0, dtu * (g.shape[0] - 2), dtu, dtype=torch.float64, device=device)
        h = torch.flip(dtu * params['pns'][2] / (params['pns'][2] + tp) ** 2 / Smin, dims=[0])
        S = torch.diff(g.T * 0.01, dim=1)[:, np.newaxis, :] / dtu
        stim_all = torch.nn.functional.conv1d(S, h[np.newaxis, np.newaxis, :], padding=S.shape[2])[:, :,
                   :S.shape[2]]
        stim_all = torch.norm(stim_all.squeeze(), dim=0)
        axp.plot(t[:-2].detach().cpu(), stim_all.detach().cpu(), label="safe")
        axp.set_xlabel("Time (ms)")
        axp.set_ylabel("PNS Threshold")
        axp.legend()
        axp.set_xlim(left=0, right=tlim - 2 * dt)

    if 'safemodel' in params.keys():
        fig, axp = plt.subplots(1, 1, figsize=(7.5, 5))
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
                torch.nn.functional.pad(S, (0, 0, pd, pd if S.shape[0] % 2 == 0 else pd+1)),
                dim=0), dim=0), dim=0)
            P = H * S
            p = torch.real(torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(P, dim=0), dim=0), dim=0)[pd:-pd, :])
            stim_ax = (coil.a1 * torch.abs(p[:, 0]) + coil.a2 * p[:, 1] + coil.a3 * torch.abs(
                       p[:, 2])) / coil.stim_limit * coil.g_scale * 100
            axp.plot(t[:stim_ax.numel()].detach().cpu(), stim_ax.detach().cpu(), label='safe ' + fld)
            stim += ((coil.a1 * torch.abs(p[:, 0]) + coil.a2 * p[:, 1] + coil.a3 * torch.abs(
                p[:, 2])) / coil.stim_limit * coil.g_scale * 100) ** 2

        stim_all = torch.sqrt(stim).squeeze()
        axp.plot(t[:stim_all.numel()].detach().cpu(), stim_all.detach().cpu(), label="safe")
        axp.set_xlabel("Time (ms)")
        axp.set_ylabel("PNS Threshold")
        axp.legend()
        axp.set_xlim(left=0, right=tlim - 2 * dt)

    if 'acoustic' in params.keys():
        fig, axa = plt.subplots(1, C.shape[1], figsize=(15, 5))
        t_H = torch.arange(0, 50/dt, device=device) * dt
        H = torch.cat((torch.zeros((t_H.numel() - 1, 2), device=t_H.device),
                       dt * tensorInterp(params['acoustic'][0], params['acoustic'][1], t_H.detach()[:-1])))
        pd = (H.shape[0] - g_init.T.shape[1]) // 2
        G = torch.nn.functional.pad(g_init[:, :-1].T, (pd, pd)).T
        H = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(H, dim=0), dim=0), dim=0)
        G = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(G, dim=0), dim=0), dim=0)
        A = H[:G.shape[0], :] * G
        A = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(A, dim=0), dim=0), dim=0)[pd:, :]
        for i in range(C.shape[1]):
            axa[i].plot(torch.abs(A[:, i].detach().cpu()), label="Unsafe")
        pd = (H.shape[0] - g.T.shape[1]) // 2
        G = torch.nn.functional.pad(g[:, :-1].T, (pd, pd)).T
        G = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(G, dim=0), dim=0), dim=0)
        A = H[:G.shape[0], :] * G
        A = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(A, dim=0), dim=0), dim=0)[pd:, :]
        for i in range(C.shape[1]):
            axa[i].plot(torch.abs(A[:, i].detach().cpu()), label="Optimized")
        axa[0].set_ylabel("X-Gradient Acoustics (A.U.)")
        axa[1].set_ylabel("Y-Gradient Acoustics (A.U.)")
        axa[0].set_xlabel("Time (ms)")
        axa[1].set_xlabel("Time (ms)")
        axa[0].legend()
        axa[1].legend()

    if 'acousticfreq' in params.keys():
        fig, axf = plt.subplots(1, C.shape[1], figsize=(15, 5))
        for i in range(C.shape[1]):
            axfr = axf[i].twinx()
            ln3 = axfr.plot(params['acousticfreq'][1].cpu(), torch.abs(params['acousticfreq'][0][:,i]).cpu(), label="ARF",
                            color='tab:green')
            ln1 = axf[i].plot(freq_init.detach().cpu().numpy(), np.abs(gf_init[:, i].detach().cpu().numpy()),
                              label="Loud")
            ln2 = axf[i].plot(freq.detach().cpu().numpy(), np.abs(gf[:, i].detach().cpu().numpy()), label="Optimized")
            lns = ln1 + ln2 + ln3
            axf[i].legend(lns, [l.get_label() for l in lns])
            axf[i].set_xlabel('Frequency (kHz)')
            axf[i].set_ylabel(dir[i] + '-Gradient Power (a.u.)')
            axfr.set_ylabel(dir[i] + '-Gradient ARF (a.u.)')
            axf[i].set_xlim((0.3, params['acousticfreq'][1][-1].cpu()))
            axf[i].set_ylim((0, axf[0].get_ylim()[1]))
    plt.show()

    Cinit = tensorInterp(CC, p, p_of_t)  # Get time optimal trajectory
    ginit = torch.diff(Cinit, dim=0) / gamma / dt  # Get time optimal gradient
    s = torch.diff(g, dim=0) / dt  # Get optimized gradient slew-rate
    sinit = torch.diff(ginit, dim=0) / dt  # Get time optimal gradient slew-rate
    return (Cnew.detach().cpu().numpy(), t.detach().cpu().numpy(), g.detach().cpu().numpy(), s.detach().cpu().numpy(),
            ginit.detach().cpu().numpy(), sinit.detach().cpu().numpy(), g_last.detach().cpu().numpy())


def initSolution(C, g0=None, gfin=None, gmax=4, smax=15, dt=4e-3, ds=None, rv=False):
    """
    Compute the velocity in arclength to meet gradient and slew constraints.

    Given a k-space trajectory `C(p)` and specified gradient and slew constraints,
    this function computes a velocity in arclength that meets the constraints
    while minimizing the time to traverse the trajectory.

    Parameters
    ----------
    C : ndarray
        The curve in k-space, provided in any parametrization [1/cm].
        Accepts a Nx2 array for 2D trajectories or a Nx3 array for 3D trajectories.
    g0 : float, optional
        Initial gradient amplitude. If not specified, defaults to `0`.
    gfin : float, optional
        Gradient value at the end of the trajectory. If not achievable,
        the result will be the largest possible amplitude. If not specified,
        defaults to maximizing the gradient.
    gmax : float, optional
        Maximum gradient [G/cm]. Defaults to `3.9`.
    smax : float, optional
        Maximum slew rate [G/cm/ms]. Defaults to `14.5`.
    dt : float, optional
        Sampling time interval [ms]. Defaults to `4e-3`.
    ds : float, optional
        Step size for ODE integration. If not specified, a default value is used.

    Returns
    -------
    k : ndarray
        Exact k-space corresponding to the gradient `g`. This function reparametrizes `C`,
        then computes its derivative. Note that numerical errors in the derivative may
        lead to slight deviations.
    phi : ndarray
        Geometry constraints on the amplitude as a function of arclength.
    st : ndarray
        Derivative of arclength with respect to time as a function of arclength
        (the solution).

    Notes
    -----
    This function is based on the original MATLAB implementation by Michael Lustig (2005).
    Python implementation and modifications by Matthew A. McCready (2024).
    """

    gamma = 4.258

    # Represent the curve using spline with parametrization p
    Cp = np.diff(C, axis=0)
    Cp = np.vstack((Cp, Cp[-1]))
    p = cumulative_trapezoid(np.linalg.norm(Cp, axis=1), axis=0, initial=0)
    Lp = p[-1]
    PP = CubicSpline(p, C)

    # Interpolate curve for gradient accuracy
    dp = np.amin(np.diff(p)) / 10
    p = np.arange(0, Lp, dp)
    CC = PP(p)

    # Find length of the curve
    Cp = np.diff(CC, axis=0) / dp
    Cp = np.vstack((Cp, Cp[-1]))
    s_of_p = cumulative_trapezoid(np.linalg.norm(Cp, axis=1), axis=0, initial=0) * dp
    L = s_of_p[-1]

    # Decide ds and compute st for the first point
    stt0 = gamma * smax  # Always assumes the first point is max slew
    st0 = stt0 * dt / 6  # Start at fraction of the gradient for accuracy close to g=0
    s0 = 3 * st0 * dt
    if ds is None:
        ds = s0 / 1.5  # Smaller step size for numerical accuracy

    s = np.arange(0, L, ds)
    s_half = np.arange(0, L, ds / 2)

    if g0 is None:
        g0 = 0

    p_of_s_half = interp1d(s_of_p, p, kind='cubic')(s_half)
    p_of_s = p_of_s_half[::2]

    sta = np.zeros_like(s, dtype=float)
    sta[0] = np.amin((np.amax(g0 * gamma + st0), gamma * gmax))

    # Compute constraints (forbidden line curve)
    phi, k, Cprime = sdotMax(PP, p_of_s_half, s_half, gmax, smax, ds, rv)
    if rv:
        k = np.vstack((k, k[-1], k[-1]))  # Extend for the Runge-Kutte method
    else:
        k = np.hstack((k, k[-1], k[-1]))

    print('Solve ODE forward...')

    # Solve ODE forward
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
        for n in range(2, len(s) + 1):
            dstds = RungeKutte(s[n - 1], ds, sta[n - 2], Cprime[(2 * n - 4):(2 * n - 1)],
                               k[(2 * n - 4):(2 * n - 1)], smax, L, rv=rv)
            tmpst = sta[n - 2] + dstds

            if np.isnan(tmpst):
                sta[n - 1] = phi[2 * n - 2]
            else:
                sta[n - 1] = min(tmpst, phi[2 * n - 2])

        print("Time elapsed: ", time.time() - start)
        stb = np.zeros_like(s)

        if gfin is None:
            stb[-1] = sta[-1]
        else:
            stb[-1] = min(max(gfin * gamma, st0), gamma * gmax)

        # Solve ODE backwards
        print('Solve ODE backwards...')
        for n in range(len(s) - 1, 0, -1):
            dstds = RungeKutte(s[n - 1], ds, stb[n], Cprime[2 * n:max(2 * n - 3, 0) or None:-1],
                               k[2 * n:max(2 * n - 3, 0) or None:-1], smax, L, rv=rv)

            tmpst = stb[n] + dstds

            if np.isnan(tmpst):
                stb[n - 1] = phi[n * 2 - 2]
            else:
                stb[n - 1] = min(tmpst, phi[n * 2 - 2])
        print("Total time elapsed: ", time.time() - start, "\n")

    # Take the minimum of the curves
    st = np.min([sta, stb], axis=0)

    # Compute time
    t_of_s = cumulative_trapezoid(1 / st * ds)
    t_of_s = np.hstack((0, t_of_s))
    t = np.arange(0, t_of_s[-1] + 1e-10, dt)

    s_of_t = np.clip(interp1d(t_of_s, s, kind='cubic')(t), 0, s[-1])
    p_of_t = interp1d(s, p_of_s, kind='cubic')(s_of_t)

    C = PP(p_of_t)

    g = np.diff(C, axis=0) / gamma / dt
    g = np.vstack((g, 2 * g[-1, :] - g[-2, :]))
    s = np.linalg.norm(np.diff(g, axis=0), axis=1) / dt
    if np.amax(s) > smax:
        print('WARNING: MAXIMUM SLEW RATE VIOLATED IN TIME-OPTIMAL SOLUTION')
        print('LIMIT: ', smax, '\nMAX REACHED: ', np.amax(s), '\n')

    return st, phi, k, s_half


def sdotMax(PP, p_of_s, s, gmax, smax, ds, rv=False):
    """
    This function calculates the upper bound for the time parametrization `sdot`
    (a non-scaled maximum gradient constraint) as a function of arclength `s`,
    based on the provided k-space curve and constraints.

    Parameters
    ----------
    PP : spline
        Spline polynomial representing the k-space curve.
    p_of_s : ndarray
        Function representing the parametrization as a function of arclength.
    s : ndarray
        Arclength
    gmax : float
        Maximum gradient amplitude [G/cm].
    smax : float
        Maximum slew rate [G/cm/ms].
    rv : bool
        Flag indicating whether to use the rotationally variant solution.

    Returns
    -------
    sdot : ndarray
        Maximum `sdot` (first derivative of `s`) as a function of arclength `s`.
    k : ndarray
        Curvature as a function of arclength `s` and the length of the curve `L`.
    Cs : ndarray
        Derivative (tangent) of trajectory with respect to arclength 's'.

    Notes
    -----
    This function is based on the original MATLAB implementation by Michael Lustig (2005).
    Python implementation by Matthew A. McCready (2024).
    """

    gamma = 4.258

    s = s.flatten()
    dp_p = p_of_s[1:] - p_of_s[:-1]
    dp_p = np.append(dp_p, dp_p[-1])
    dp_m = p_of_s - np.append(p_of_s[0], p_of_s[:-1])
    dp_m[0] = dp_m[1]
    ds_p = s[1:] - s[:-1]
    ds_p = np.append(ds_p, ds_p[-1])
    ds_m = s - np.append(s[0], s[:-1])
    ds_m[0] = ds_m[1]

    Cs_p = (PP(p_of_s + dp_p) - PP(p_of_s)) / ds_p[:, np.newaxis]
    Cs_m = (PP(p_of_s) - PP(p_of_s - dp_m)) / ds_m[:, np.newaxis]
    Cs = Cs_p / 2 + Cs_m / 2
    Css = (Cs_p - Cs_m) / (ds_m[:, np.newaxis] / 2 + ds_p[:, np.newaxis] / 2)

    if not rv:
        k = np.linalg.norm(Css, axis=1)
        # Fix edge numerical problems
        k[-1] = k[-2]
        k[0] = k[1]
        #k = cur
        # Calc I constraint curve (maximum gradient)
        sdot1 = gamma * gmax * np.ones_like(s)

        # Calc II constraint curve (curve curvature dependent)
        sdot2 = np.sqrt(gamma * smax / (np.abs(k) + np.finfo(float).eps))

        # Calc total constraint
        sdot = np.minimum.reduce([sdot1, sdot2], axis=0)
        Cs = []
    else:
        eps = 5e-16
        k = np.abs(Css)
        Cs = np.abs(Cs)
        sdot = gamma * gmax * np.min(1/(Cs+eps), axis=1)

    return sdot, k, Cs


def RungeKutte(s, ds, st, Cprime, k, smax, L, rv=False):
    """
    This function performs a single RK4 integration step for the nonlinear ODE
    described in [1], using arclength as the integration variable.

    Parameters
    ----------
    s : float
        Current arclength.
    ds : float
        Step size for the arclength.
    st : float
        Function representing k-space speed as a function of arclength.
    Cprime : ndarray
        K-space trajectory tangent curve parameterized by arclength.
    k : ndarray
        K-space trajectory curvature parameterized by arclength.
    smax : float
        Maximum allowed slew rate.
    L : float
        Total length of the k-space trajectory.
    rv : bool
        Flag indicating whether to use the rotationally variant solution.

    Returns
    -------
    res : float
        Result of the integration step.

    Notes
    -----
    This function is based on the original MATLAB implementation by Michael Lustig (2005).
    Python implementation by Matthew A. McCready (2024).
    """

    gamma = 4.258
    if not rv:
        k = np.abs(k)
        k1 = ds * 1 / st * np.sqrt(gamma**2 * smax**2 - k[0]**2 * st**4)
        k2 = ds * 1 / (st + k1 / 2) * np.sqrt(gamma**2 * smax**2 - k[1]**2 * (st + k1 / 2)**4)
        k3 = ds * 1 / (st + k2 / 2) * np.sqrt(gamma**2 * smax**2 - k[1]**2 * (st + k2 / 2)**4)
        k4 = ds * 1 / (st + k3) * np.sqrt(gamma**2 * smax**2 - k[2]**2 * (st + k3)**4)
    else:
        k1 = ds * 1 / st * np.min((-k[0, :] * st ** 2 + gamma * smax) / Cprime[0, :])
        k2 = ds * 1 / (st + k1 / 2) * np.min((-k[1, :] * (st + k1 / 2) ** 2 + gamma * smax) / Cprime[1, :])
        k3 = ds * 1 / (st + k2 / 2) * np.min((-k[1, :] * (st + k2 / 2) ** 2 + gamma * smax) / Cprime[1, :])
        k4 = ds * 1 / (st + k3 / 2) * np.min((-k[2, :] * (st + k3 / 2) ** 2 + gamma * smax) / Cprime[2, :])

    res = k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

    return res
