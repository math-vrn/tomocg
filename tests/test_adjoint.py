import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np
import tomocg as pt

if __name__ == "__main__":

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 128  # number of angles (rotations)
    center = n/2  # rotation center
    theta = np.linspace(0, np.pi, ntheta).astype('float32')  # angles
    niter = 64  # tomography iterations
    pnz = 32  # number of slice partitions for simultaneous processing in tomography
    # Load object
    beta = dxchange.read_tiff('data/beta-chip-128.tiff')
    delta = dxchange.read_tiff('data/delta-chip-128.tiff')
    u0 = delta+1j*beta

    # Class gpu solver
    with pt.SolverTomo(theta, ntheta, nz, n, pnz, center) as slv:
        # generate data
        data = slv.fwd_tomo_batch(u0)
        # adjoint test
        u1 = slv.adj_tomo_batch(data)
        print('Adjoint test: ', np.sum(data*np.conj(data)),
              '=?', np.sum(u0*np.conj(u1)))
