import cupy as cp
import dxchange
import numpy as np
import tomocg as pt

if __name__ == "__main__":

    # Model parameters
    n = 256  # object size in x,y
    nz = 256  # object size in z
    ntheta = 256//2  # number of angles (rotations)
    center = n/2  # rotation center
    theta = np.linspace(0, np.pi, ntheta).astype('float32')  # angles

    pnz = 32  # number of slice partitions for simultaneous processing in tomography
    # Load object
    beta = dxchange.read_tiff('data/beta-chip-256.tiff')
    delta = dxchange.read_tiff('data/delta-chip-256.tiff')
    u0 = delta+1j*beta
    ngpus = 1

    with pt.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as slv:
        # generate data
        data = slv.fwd_tomo_batch(u0)
        # initial guess
        u = np.zeros([nz, n, n], dtype='complex64')
        u = slv.cg_tomo_batch(data, u, 64)
        # save results
        dxchange.write_tiff(u.real,  'rec/delta', overwrite=True)
        dxchange.write_tiff(u.imag,  'rec/beta', overwrite=True)
        dxchange.write_tiff(data.real,  'datar/r', overwrite=True)
        dxchange.write_tiff(data.imag,  'datar/i', overwrite=True)
        