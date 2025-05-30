from optiks.optiks import optiks
from optiks.loss_functions import *
import matplotlib
matplotlib.use("webagg")
from optiks.options import *
from optiks.utils import rosetteTraj

"""
This script designs a 1mm, 7 leaf rosette for the GE 3T UHP system with a maximum PNS threshold of 80% as calculated
using the IEC 60601-2-33 model.
"""

# Setting hardware options==============================================================================================
hw = HardwareOpts(gfin=0, gmax=10, smax=19.5, dt=4e-3)

# Designing desired trajectory (Rosette)================================================================================
res = 0.1  # cm^-1
C = rosetteTraj(res, n1=7, n2=5, npoints=800)
C_m = np.hstack((np.real(C), np.imag(C))).astype(float)

# Setting design options================================================================================================
Pthresh = 100*0.8
system = "UHP"
if system == "UHP":
    r = 26.5
    c = 359e-6
    alpha = 0.37
else:
    r = 23.4
    c = 334e-6
    alpha = 0.333
params = {'terms': [time_min, slew_lim, pns_lim],
          'pns': [Pthresh, r, c, alpha]}
weights = {'time': 1e4,
           'slew': 7e2,
           'pns': 5e1}
des = DesignOpts(params=params, weights=weights)

# Setting solver options================================================================================================
sv = SolverOpts(ds=5e-5, maxiter=25000, count=50, derate=0.9)

# Designing gradient waveforms==========================================================================================
C_v, t_sf, g_sf, s_sf, g_usf, s_usf, g_last = optiks(C_m, hwopts=hw, dsopts=des, svopts=sv)
