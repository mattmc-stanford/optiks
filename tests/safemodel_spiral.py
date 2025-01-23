import numpy as np
from types import SimpleNamespace
from optiks.options import *
from optiks.loss_functions import *
from optiks.optiks import optiks
from optiks.utils import spiralTraj

"""
This script designs a 2mm, 22cm FOV, R=3 spiral with a maximum pns threshold of 100% as calculated using the SAFE model.
The hardware used for the SAFE model is a fictional example taken from PulseSeq.
"""

# Setting hardware options==============================================================================================
hwOpts = HardwareOpts(gfin=0, gmax=10, smax=19.5, dt=4e-3)

# Designing desired trajectory (Rosette)================================================================================
gfin = None
fov = 22 / 3
res = 0.2
C = spiralTraj(fov, res, npoints=7e3)
C_m = np.hstack((np.real(C), np.imag(C))).astype(float)

# Setting design options================================================================================================
Pthresh = 100*1.0

hw = SimpleNamespace()
hw.name = 'MP_GPA_EXAMPLE'
hw.checksum = '1234567890'
hw.dependency = ''

hw.x = SimpleNamespace()
hw.x.tau1 = 0.20  # ms
hw.x.tau2 = 0.03  # ms
hw.x.tau3 = 3.00  # ms
hw.x.a1 = 0.40
hw.x.a2 = 0.10
hw.x.a3 = 0.50
hw.x.stim_limit = 30.0  # T/m/s
hw.x.stim_thresh = 24.0  # T/m/s
hw.x.g_scale = 0.35  # 1

hw.y = SimpleNamespace()
hw.y.tau1 = 1.50  # ms
hw.y.tau2 = 2.50  # ms
hw.y.tau3 = 0.15  # ms
hw.y.a1 = 0.55
hw.y.a2 = 0.15
hw.y.a3 = 0.30
hw.y.stim_limit = 15.0  # T/m/s
hw.y.stim_thresh = 12.0  # T/m/s
hw.y.g_scale = 0.31  # 1

hw.z = SimpleNamespace()
hw.z.tau1 = 2.00  # ms
hw.z.tau2 = 0.12  # ms
hw.z.tau3 = 1.00  # ms
hw.z.a1 = 0.42
hw.z.a2 = 0.40
hw.z.a3 = 0.18
hw.z.stim_limit = 25.0  # T/m/s
hw.z.stim_thresh = 20.0  # T/m/s
hw.z.g_scale = 0.25  # 1

params = {'terms': [time_min, slew_lim, safemodel_lim],
          'safemodel': [Pthresh, hw]}
weights = {'time': 1e4,
           'slew': 7e2,
           'pns': 5e1}
des = DesignOpts(params=params, weights=weights)

# Setting solver options================================================================================================
sv = SolverOpts(ds=5e-5, maxiter=25000, count=50, derate=0.8)

# Designing gradient waveforms==========================================================================================
C_v, t_sf, g_sf, s_sf, g_usf, s_usf, g_last = optiks(C_m, hwopts=hwOpts, dsopts=des, svopts=sv)