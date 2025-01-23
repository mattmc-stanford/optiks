import numpy as np
from optiks.optiks import optiks
from optiks.loss_functions import *
from optiks.options import *
from optiks.utils import spiralTraj
from scipy.io import loadmat
import torch

"""
This script designs a 2mm, 22cm FOV, R=3 spiral for the GE PREMIER system minimizing acoustic power as weighted by a
prior measured acoustic response function.
"""

# Designing desired trajectory (Spiral)=================================================================================
gfin = None
fov = 22 / 3
res = 0.2
C = spiralTraj(fov, res, npoints=7e3)
rew = True
if rew:
    kmax = 1 / (2 * res)
    dt = 4e-3  # In ms
    theta = np.angle(C[-1])[0]
    rot2d = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T_rew = 0.5
    t_rew = np.arange(dt, T_rew+dt, dt)
    C_rew = kmax/2*(1 + np.cos(np.pi*t_rew/T_rew) + 1j*np.sin(np.pi*t_rew/T_rew))
    C_rew = np.hstack((np.real(C_rew[:, np.newaxis]), np.imag(C_rew[:, np.newaxis])))
    C_rew = C_rew @ rot2d.T
    C_rew = C_rew[:, 0] + 1j * C_rew[:, 1]
    C = np.vstack((C, C_rew[:, np.newaxis]))
    gfin = 0
C_m = np.hstack((np.real(C), np.imag(C))).astype(float)

# Setting hardware options==============================================================================================
hw = HardwareOpts(gfin=gfin, gmax=7, smax=13.5, dt=4e-3)

# Setting design options================================================================================================
Pthresh = 100*0.72
system = "PREMIER"
if system == "MAGNUS":
    r = 52.2
    c = 611e-6
    alpha = 0.324
elif system == "UHP":
    r = 26.5
    c = 359e-6
    alpha = 0.37
else:
    r = 23.4
    c = 334e-6
    alpha = 0.333
ATF = loadmat('acoustic_response_functions/larf_PREMIER.mat')
params = [ATF['larf'], ATF['freqs']]
params[1] = torch.tensor(params[1]/1e3).flatten()
params[0] = torch.tensor(params[0][:, :-1])
params = {'terms': [time_bound, slew_lim, acoustic_freq_min],
          'bound': 27,
          'acousticfreq': params,
          'pns': [Pthresh, r, c, alpha]}
weights = {'time': 5e2,
           'slew': 9e2,
           'acoustic': 1e5}
des = DesignOpts(params=params, weights=weights)

# Setting solver options================================================================================================
sv = SolverOpts(ds=5e-5, maxiter=15000, count=50, derate=0.85)

# Designing gradient waveforms==========================================================================================
C_v, t_sf, g_sf, s_sf, g_usf, s_usf, g_last = optimGradientRIV(C_m, hwopts=hw, dsopts=des, svopts=sv)
