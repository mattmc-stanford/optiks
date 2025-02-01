import numpy as np
from optiks.optiks import optiks
from optiks.loss_functions import *
from optiks.utils import spiralTraj
from optiks.options import *

"""
This script designs a 3mm, 24cm FOV, R=2 spiral for the GE 3T UHP system minimizing power deposited in known mechanical
resonance bands, with a maximum duration of 16.5ms.
"""

# Designing desired trajectory (Spiral)=================================================================================
C = spiralTraj(12, 0.3)
C_m = np.hstack((np.real(C), np.imag(C))).astype(float)

# Setting hardware options==============================================================================================
hw = HardwareOpts(gmax=10, smax=19.7)

# Setting design options================================================================================================
system = "UHP"

if system == "UHP":
    r = 26.5
    c = 359e-6
    alpha = 0.37
    fedges = [[0.51, 0.575], [0.96, 1.06], [1.14, 1.26], [1.4, 1.56], [1.72, 1.9]]  # UHP
elif system == "PREMIER":
    fedges = [[0.560, 0.620], [0.96, 1.310], [1.860, 1.950]]  # Premier
    fedges = [[0.550, 0.630], [0.96, 1.310], [1.850, 1.960], [4, np.inf]]  # Premier with band-limiting
    r = 23.4
    c = 334e-6
    alpha = 0.333
else:
    r = 23.4
    c = 334e-6
    alpha = 0.333

params = {'terms': [time_bound, slew_lim, freq_min],
          'bound': 16.5,
          'frequency': fedges}
weights = {'time': 1e0,
           'slew': 1e1,
           'frequency': 4e3}

des = DesignOpts(params=params, weights=weights)


# Setting solver options================================================================================================
sv = SolverOpts(ds=5e-5, maxiter=20000, count=50)

# Designing gradient waveforms==========================================================================================
C_v, t_sf, g_sf, s_sf, g_usf, s_usf, g_last = optiks(C_m, hwopts=hw, dsopts=des, svopts=sv)
