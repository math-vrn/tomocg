import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np

import ptychotomo as pt

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])

    cp.cuda.Device(igpu).use()  # gpu id to use
    # use cuda managed memory in cupy
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # Model parameters
    n = 256  # object size n x,y
    nz = 256  # object size in z
    ntheta = 3*n//4  # number of angles (rotations)
    voxelsize = 1e-6  # object voxel size
    energy = 8.8  # xray energy
    maxint = 0.1  # maximal probe intensity
    prbsize = 16  # probe size
    prbshift = 8  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [64, 64]  # detector size
    noise = True  # apply discrete Poisson noise

    # Reconstrucion parameters
    model = 'poisson'  # minimization funcitonal (poisson,gaussian)
    alpha = 3*1e-7  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 1024  # tomography iterations
    niter = 100  # ADMM iterations
    ptheta = 32  # number of angular partitions for simultaneous processing in ptychography
    pnz = 256  # number of slice partitions for simultaneous processing in tomography

    # Load a 3D object
    beta = dxchange.read_tiff('data/beta-chip-256.tiff')
    delta = -dxchange.read_tiff('data/delta-chip-256.tiff')
    print(beta.shape)
    obj = cp.array(delta+1j*beta)

    # init probe, angles, scanner
    prb = cp.array(pt.probe(prbsize, maxint))
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
    scan = cp.array(pt.scanner3(theta, obj.shape, prbshift,
                                prbshift, prbsize, spiral=0, randscan=True, save=False))
    # Class gpu solver
    slv = pt.Solver(prb, scan, theta, det, voxelsize,
                    energy, ntheta, nz, n, ptheta, pnz)

    def signal_handler(sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    data = slv.fwd_tomo_batch(obj)/slv.coeftomo
    data.real[data.real<0]=0
    data.imag[data.imag<0]=0
    # data[data < 0] = 0
    # print(np.max(data))
    # print(np.min(data))
    # data[data<0]=0
    # Compute data
    # data = np.abs(slv.exptomo(slv.fwd_tomo_batch(obj))).get()

    maxinta = np.array([0.5, 0.1, 0.05,0.01,0.005,0.001])
    print(np.max(data))
    for k in range(len(maxinta)):
        maxint = maxinta[k]
        print(np.shape(data*maxint))
        for id in range(len(maxinta)):
            if (noise == True):  # Apply Poisson noise
                datanoise = cp.array(np.random.poisson(data.real.get()*maxint*10000000).astype(
                    'float32'))+1j*cp.array(np.random.poisson(data.imag.get()*maxint*10000000).astype('float32'))
            init = obj*0
            rec = slv.cg_tomo0(datanoise, init, titer)
            # print(cp.linalg.norm(rec))
            # print(np.max(datanoise))
            # # Save result
            name = 'maxint' + str(maxint)+'/rec_n'+str(n)+'ntheta' + str(ntheta)+'id'+str(id)
            dxchange.write_tiff(rec.real.get(),  'tomo_rec_noisy/'+name+'re',overwrite=True)
            dxchange.write_tiff(rec.imag.get(),  'tomo_rec_noisy/'+name+'im',overwrite=True)
            dxchange.write_tiff(datanoise.real.get(),  'data_noisy/'+name+'re',overwrite=True)
            dxchange.write_tiff(datanoise.imag.get(),  'data_noisy/'+name+'im',overwrite=True)
