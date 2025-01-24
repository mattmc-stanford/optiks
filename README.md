# OPTIKS

Optimizied (gradient) Properties through Timing In K-Space.
Please cite [TBD] when using this package.

This package is intended for the design of arbitrary k-space trajectory gradient waveforms in MRI.

## Installation / Setup

From the command line clone the optiks repo:
```
git clone https://github.com/mattmc-stanford/optiks.git
```

Then install optiks as a package. If you wish to just use optiks and not perform any customization run the command:
```
pip install .
```

If instead you wish to create custom loss functions or otherwise edit optiks and use the edited package run the command:
```
pip install -e .
```

## Organization

Diagrams and overview here.

## Example Scripts

Examples can be found under:
https://github.com/mattmc-stanford/optiks/tree/main/tests

## Debugging

Coming soon.

## References
_Paper coming soon_

ISMRM Abstract (Under Review):

_Optimized gradient Properties through Timing In K-Space (OPTIKS), M. A. McCready, X. Cao, C. Liao, K. Setsompop, J. M. Pauly, and A. B. Kerr. Proc. Intl. Soc. Mag. Res. Med., 2025_

Background Literature:

M. Lustig, S.-J. Kim, and J. M. Pauly, “A fast method for designing time-optimal gradient waveforms for arbitrary k-space trajectories,” IEEE
Transactions on Medical Imaging, vol. 27, no. 6, pp. 866–873, 2008.

The Fastest Arbitrary k-space Trajectories, S. Vaziri and M. Lustig. Abstract No 2284. Proc. Intl. Soc. Mag. Res. Med. 20, 2012
https://people.eecs.berkeley.edu/~mlustig/tOptGrad_ISMRM12.pdf

SAFE-Model - A New Method for Predicting Peripheral Nerve Stimulations in MRI, F.X. Herbank and M. Gebhardt. Abstract No 2007. Proc. Intl. Soc. Mag. Res. Med. 8, 2000, Denver, Colorado, USA
https://cds.ismrm.org/ismrm-2000/PDF7/2007.PDF

IEC
“Medical electrical equipment - part 2–33: particular requirements for the basic safety and essential performance of magnetic resonance equipment
for medical diagnosis,” standard, International Electrotechnical Commission, Aug. 2022.

PulseSeq
https://github.com/pulseq/pulseq