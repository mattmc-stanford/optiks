from dataclasses import dataclass
from numpy import ndarray


@dataclass
class HardwareOpts:
    """
    Class defining input options for gradient hardware.

    Parameters
    ----------
    g0 : float, optional
        Initial gradient amplitude (set None for g0 = 0) [G/cm]. Default None.
    gfin : float, optional
        Gradient amplitude at the end of the trajectory [G/cm]. If not possible, the result would be the largest
        possible amplitude (set None for optimal). Default None.
    gmax : float, optional
        Maximum gradient amplitude [G/cm]. Default 4 G/cm.
    smax : float, optional
        Maximum slew rate [G/cm/ms]. Default 15 G/cm/ms.
    dt : float, optional
        Sampling time interval for returned waveform [ms]. Default 4e-3 (4us).
    """
    g0: float = None  # initial gradient (G/cm)
    gfin: float = None  # final gradient (G/cm)
    gmax: float = 4  # maximum allowed gradient (G/cm)
    smax: float = 15  # maximum allowed slew rate (G/cm/ms)
    dt: float = 4e-3  # time sampling for returned waveform (ms)


@dataclass
class DesignOpts:
    """
    Class defining input options for design goals.

    Parameters
    ----------
    weights : dict
        Dictionary of weights corresponding to loss function terms. See notes for existing examples.
    params : dict
        Dictionary of optional parameters passed to loss function terms. See notes for existing examples.
    rv : bool
        Rotationally variant flag. If true, the design is rotationally variant, optimized for orientation of input
        trajectory C. Default False.

    Notes
    -----
    Concerning params attribute keys
    Mandatory key:
        'terms' - a list of function handles to loss function terms defined in loss_functions.py
    Optional keys: (one for each corresponding term used containing necessary parameters)
        'pns'       - a list containing [Pthresh, r, c, alpha] the maximum allowed PNS threshold, rheobase, chronaxie
                      time, and the effective coil length as in IEC 60601-2-33.
        'frequency' - a list containing mechanical resonance band edges [[low1, high1], [low2, high2], ...]
        'acoustic'  - a list containing [ARF, ARF_frequencies] the acoustic response function in the frequency domain,
                      and the frequencies it is evaluated at.
        'safemodel' - a list containing [Pthresh, hardware] where hardware is a SimpleNamespace as in PulseSeq for a
                      Siemens system.

        An optional key should be added for passing parameters to each new custom loss function term you define.
    """
    weights: dict = None  # dictionary of weights for terms in custom loss function
    params: dict = None  # dictionary of terms and parameters for custom loss function
    rv: bool = False  # flag for rotationally variant waveform design


@dataclass
class SolverOpts:
    """
    Class defining input options for solver settings.
    Parameters
    ----------
    ds : float, optional
        Arclength paramaterization sampling. Set None for default value (s[0] / 1.5).
    maxiter : int, optional
    count : int, optional
        Number of iterations to store loss details after. Default 50.
    derate : float, optional
        Factor by which to derate the initializing solution, must be in (0, 1). Default 0.8.
    initsol : ndarray, optional
        Initialization for gradient descent. If None, the time optimal waveform is used with the input derating factor.
        Default None.
    lr : float, optional
        Initial learning rate for AdamW gradient descent optimizer. Default 1e-4.
    """
    ds: float = None  # arc-length sampling for design (cm^-1)
    maxiter: int = 4000  # maximum allowed number of gradient descent steps
    count: int = 50  # number of iterations to display progress after
    derate: float = 0.8  # factor to derate initial (time optimal) solution v(s) by before optimizing
    initsol: ndarray = None  # custom initial waveform solution - replaces time optimal solution (G/cm)
    lr: float = 1e-4  # initial learning rate for AdamW optimizer
