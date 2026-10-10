import multiprocessing as mp

import numpy as np


def mhd_grid(dat):
    shape = dat.shape
    tvar = np.zeros(shape[:-1] + (shape[-1] + 1,))
    tvar[..., 1:-1] = (dat[..., 1:] + dat[..., :-1]) / 2
    tvar[..., 0] = dat[..., 0]
    tvar[..., -1] = dat[..., -1]
    return tvar


def calc_coeff(delta, dtau_min=1e-8):
    # if dtau is big enough
    edt0 = np.exp(-delta)
    cc0 = (1 - edt0) / delta
    # if dtau is too small, do a taylor expansion up to 4 terms
    edt1 = 1 - delta + delta**2 / 2 - delta**3 / 6
    cc1 = 1 - delta / 2 + delta**2 / 6
    # get overall values
    edt = edt0 * (delta > dtau_min) + edt1 * (delta <= dtau_min)
    cc = cc0 * (delta > dtau_min) + cc1 * (delta <= dtau_min)
    # calc coefficients
    Ac = 1 - cc  # coeff for local B
    Bc = cc - edt  # coeff for upwind B
    return Ac, Bc


class Solver:
    def __init__(self, kappa, rho, S, z=None, tau=None, nmu=4, mu=None):
        self.transform = False
        if kappa.ndim == 3:
            self.transform = True
            self.nw, self.nbin, self.nz = kappa.shape
            kappa = kappa.reshape(self.nw * self.nbin, self.nz)
        self.kappa = kappa
        self.rho = rho
        if S.ndim == 3:
            S = S.reshape(self.nw * self.nbin, self.nz)
        self.S = S

        if mu is not None:
            self.mus, self.wmus = [mu], [1.0]
        else:
            self.mus, self.wmus = np.polynomial.legendre.leggauss(2 * nmu)

        self.z = z
        if tau is None:
            if self.z is None:
                raise ValueError("Either 'z' or 'tau' must be provided.")
            self.tau = compute_tau(self.z, self.kappa, self.rho)
        else:
            self.tau = np.repeat(tau[np.newaxis, :], kappa.shape[0], axis=0)

    @property
    def I(self):
        if self.transform:
            return self.intensity.reshape(self.nw, self.nbin, self.nz)
        return self.intensity

    @property
    def J(self):
        """Return the mean intensity integrated over angle."""
        J = np.sum(self.intensity * self.wmus, axis=-1) / 2
        if self.transform:
            return J.reshape(self.nw, self.nbin, self.nz)
        return J

    @property
    def F(self):
        """Return the radiative flux integrated over angle."""
        F = 2 * np.pi * np.sum(self.intensity * self.mus * self.wmus, axis=-1)
        if self.transform:
            return F.reshape(self.nw, self.nbin, self.nz)
        return F

    def solve_rte(self):
        """
        Parameters:
        -----------
        nmu : int
            number of mu angles for which to solve RTE (per hemisphere)
        """

        Nlam, nz = self.kappa.shape
        I = np.empty((Nlam, nz, len(self.mus)))

        dtau = self.tau[:, 1:] - self.tau[:, :-1]

        # Batched by ray direction: all mu > 0 (outward) share one ascending pass,
        # all mu < 0 (inward) one descending pass. Per-(lam, mu) flops run in the
        # same order as the old per-mu loop, so results are bit-identical; the
        # z-loop runs twice instead of 2*nmu times with 4x wider inner ops.
        mus = np.asarray(self.mus)
        for sign, loc in ((1.0, -1), (-1.0, 0)):
            sel = np.flatnonzero(mus * sign > 0.0)
            if sel.size == 0:
                continue
            if sign > 0.0:  # outward: S in the optically thick region
                I[:, loc, sel] = self.S[:, loc][:, np.newaxis]
            else:  # inward: nothing comes in from outside
                I[:, loc, sel] = 0.0
            amu = np.abs(mus[sel])[np.newaxis, :]  # [1, n_sel]
            # delta/[Ac, Bc] per (lam, mu): [Nlam, nz-1, n_sel].
            delta = dtau[:, :, np.newaxis] / amu[:, np.newaxis, :]
            Ac, Bc = calc_coeff(delta)
            E = np.exp(-delta)
            S = self.S[:, :, np.newaxis]
            if sign > 0.0:  # outward: sweep from the bottom up
                for i in range(nz - 1):
                    i_up, i_loc = -i - 1, -i - 2
                    I[:, i_loc, sel] = (
                        I[:, i_up, sel] * E[:, i_up, :]
                        + Ac[:, i_up, :] * S[:, i_loc, :]
                        + Bc[:, i_up, :] * S[:, i_up, :]
                    )
            else:  # inward: sweep from the top down
                for i in range(nz - 1):
                    I[:, i + 1, sel] = (
                        I[:, i, sel] * E[:, i, :] + Ac[:, i, :] * S[:, i + 1, :] + Bc[:, i, :] * S[:, i, :]
                    )

        self.intensity = I

    def get_Q(self, tau0=0.1):
        if self.z is None:
            return None
        QJ = 4 * np.pi * self.kappa * self.rho * (self.J - self.S)
        QF = mhd_grid((self.F[:, 1:] - self.F[:, :-1]) / (self.z[1:] - self.z[:-1]))
        Q = np.exp(-self.tau / tau0) * QJ + (1 - np.exp(-self.tau / tau0)) * QF

        return Q


class ParallelSolver:
    def __init__(self, kappa, rho, S, z=None, tau=None, nmu=4, mu=None):
        self.natmos = rho.shape[0]
        self.args = [
            Solver(kappa=kappa[ida], rho=rho[ida], z=z, tau=tau, S=S[ida], nmu=nmu, mu=mu) for ida in range(self.natmos)
        ]

        if mu is not None:
            self.mus, self.wmus = [mu], [1.0]
        else:
            self.mus, self.wmus = np.polynomial.legendre.leggauss(2 * nmu)

    @property
    def I(self):
        return self.intensity

    @property
    def J(self):
        return np.sum(self.intensity * self.wmus, axis=-1) / 2

    @property
    def F(self):
        return 2 * np.pi * np.sum(self.intensity * self.mus * self.wmus, axis=-1)

    def solve_rte(self, ncpus):
        with mp.Pool(processes=ncpus) as pool:
            results = pool.map(_solve_rte, self.args, chunksize=self.natmos // ncpus + 1)

        self.intensity = np.asarray(results)


def _solve_rte(solver):
    solver.solve_rte()
    I = solver.I
    del solver
    return I


def compute_tau(z, kappa, rho):
    dz = z[1:] - z[:-1]
    chi = kappa * rho
    tau = np.zeros_like(kappa)
    tau[..., 1:] = np.cumsum((chi[..., 1:] + chi[..., :-1]) * dz / 2, axis=-1)

    return tau
