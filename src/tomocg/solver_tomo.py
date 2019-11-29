"""Module for tomography."""

import cupy as cp
import numpy as np
from tomocg.radonusfft import radonusfft


class SolverTomo(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    pnz : int
        The number of slice partitions to process together
        simultaneously.
    """

    def __init__(self, theta, ntheta, nz, n, pnz, center):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(ntheta, pnz, n, center, theta.ctypes.data)
        self.nz = nz
        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo(self, u):
        """Radon transform (R)"""
        res = cp.zeros([self.ntheta, self.pnz, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u.data.ptr)        
        return res

    def adj_tomo(self, data):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.pnz, self.n, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays        
        self.adj(res.data.ptr, data.data.ptr)
        return res

    def line_search(self, minf, gamma, Ru, Rd, gu, gd):
        """Line search for the step sizes gamma"""
        while(minf(Ru, gu)-minf(Ru+gamma*Rd, gu+gamma*gd) < 0):
            gamma *= 0.5
        return gamma
    
    def fwd_tomo_batch(self, u):
        """Batch of Tomography transform (R)"""
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            # Radon transform
            res_gpu = self.fwd_tomo(u_gpu)
            # copy result to cpu
            res[:, ids] = res_gpu.get()
        return res

    def adj_tomo_batch(self, data):
        """Batch of adjoint Tomography transform (R*)"""
        res = np.zeros([self.nz, self.n, self.n], dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            data_gpu = cp.array(data[:, ids])

            # Adjoint Radon transform
            res_gpu = self.adj_tomo(data_gpu)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res
    
    # Forward operator for regularization (J)
    def fwd_reg(self, u):
        res = np.zeros([3, *u.shape], dtype='complex64')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        res *= 2/np.sqrt(3)  # normalization
        return res

    # Adjoint operator for regularization (J^*)
    def adj_reg(self, gr):
        res = np.zeros(gr.shape[1:], dtype='complex64')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        res *= -2/np.sqrt(3)  # normalization
        return res

    def solve_reg(self, u, mu, tau, alpha):
        z = self.fwd_reg(u)+mu/tau
        # Soft-thresholding
        za = cp.sqrt(cp.real(cp.sum(z*cp.conj(z), 0)))
        z[:, za <= alpha/tau] = 0
        z[:, za > alpha/tau] -= alpha/tau * \
            z[:, za > alpha/tau]/(za[za > alpha/tau])
        return z
    # Conjugate gradients tomography (for 1 slice partition)
    def cg_tomo(self, xi0, u, titer):
        """CG solver for ||Ru-xi0||_2"""
        # minimization functional
        def minf(Ru):
            f = cp.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo(u)
            grad = self.adj_tomo(Ru-xi0) / \
                (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo(d)
            gamma = 0.5*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (np.mod(i, 1) == -1):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru)))
        return u
  
    # Conjugate gradients tomography (by slices partitions)
    def cg_tomo_batch(self, xi0, init, titer):
        """CG solver for rho||Ru-xi0||_2 by z-slice partitions"""
        u = init.copy()

        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            u_gpu = cp.array(u[ids])
            xi0_gpu = cp.array(xi0[:, ids])
            # reconstruct
            u_gpu = self.cg_tomo(xi0_gpu, u_gpu, titer)
            u[ids] = u_gpu.get()
        return u

    # Conjugate gradients tomography (for all slices by batching fwd and adj operators)
    def cg_tomo_batch_all(self, xi0, u, titer):
        """CG solver for ||Ru-xi0||_2"""
        # minimization functional
        def minf(Ru):
            f = cp.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo_batch(u)
            grad = self.adj_tomo_batch(Ru-xi0) / \
                (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo_batch(d)
            gamma = 0.5*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (np.mod(i, 1) == -1):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru)))
        return u

# Conjugate gradients tomography (for all slices)
    def cg_tomo_batch_ext(self, xi0, xi1, u, tau,  titer):
        """CG solver for ||Ru-xi0||_2"""
        # minimization functional
        def minf(Ru, gu):
            f = np.linalg.norm(Ru-xi0)**2+tau*np.linalg.norm(gu-xi1)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo_batch(u)
            gu = self.fwd_reg(u)
            grad = self.adj_tomo_batch(Ru-xi0) / \
                (self.ntheta * self.n/2)+\
                tau*self.adj_reg(gu-xi1)
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo_batch(d)
            gd = self.fwd_reg(d)
            gamma = 0.5*self.line_search(minf, 1, Ru, Rd, gu, gd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (np.mod(i, 1) == -1):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru, gu)))
        return u
