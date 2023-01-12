"""References:
    [1] J. Nocedal and S. J. Wright, Numerical Optimization (Springer, New York, 1999).
"""

import warnings
from copy import deepcopy
from typing import TypeVar

import numpy as np
from ase.optimize.optimize import Optimizer
from numpy.linalg import eigh

B = TypeVar("B", bound="BFGSState")


def update_hessian(
    coords: np.ndarray,
    coords_previous: np.ndarray,
    gradient: np.ndarray,
    gradient_previous: np.ndarray,
    hessian: np.ndarray,
) -> np.ndarray:
    """Update hessian by BFGS algorithm

    REMARK: `gradient` currently means negative gradient, i.e., force

    Args:
        coords: current state (e.g. positions)
        coords_previous: state before last update (e.g. prev. positions)
        gradient: current gradient (e.g. forces)
        gradient_previous: last gradient
        hessian: current hessian

    Returns:
        new_hessian: update hessian

    Reference:
        Eq. (8.19) in [1]
    """
    dx = coords - coords_previous
    df = gradient - gradient_previous

    a = dx @ df
    dg = hessian @ dx
    b = dx @ dg
    new_hessian = hessian - np.outer(df, df) / a + np.outer(dg, dg) / b

    return new_hessian


def predict_step(gradient: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """take gradient and hessian to predict step

    Reference:
        Eq. (8.18) in [1]

    Remark:
        Line search is omitted, i.e., a_k = 1.
    """
    omega, V = eigh(hessian)
    return (V @ (gradient @ V) / np.fabs(omega)).reshape((-1, 3))


class BFGSState:
    """dataclass handling the state of an BFGS algorithm incl. initial cond."""

    def __init__(
        self,
        coords: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> None:
        """initialize state representing the BFGS step

        Args:
            coords: coordinates, e.g., positions
            gradient: the gradient of objective function, e.g.
            hessian: the Hessian
        """
        self.coords = coords
        self.gradient = gradient
        self.hessian = hessian
        self.step = None

    def update(self, coords, gradient) -> None:
        """Update the state using `update_hessian`"""
        self.hessian = update_hessian(
            coords=coords,
            coords_previous=self.coords,
            gradient=gradient,
            gradient_previous=self.gradient,
            hessian=self.hessian,
        )
        self.step = predict_step(gradient, self.hessian)
        self.coords = coords
        self.gradient = gradient

    def copy(self) -> B:
        """return a (deep) copy of the object"""
        return deepcopy(self)


class BFGSOptimizer:
    """BFGS optimizer

    Use:
        Call with .update(coords=..., gradients=...) to update internal BFGS state,
        use .residual to check residual gradient
    """
    default_configuration = {"alpha": 70.0}

    def __init__(
        self,
        coords: np.ndarray,
        gradient: np.ndarray,
        alpha: float = None,
        hessian: np.ndarray = None,
        residual: str = "max_atom",
    ) -> None:
        """initialize with a coords (e.g. positions), gradient (e.g. forces),
        optionally hessian

        Args:
            coords: current state (e.g. positions)
            coords_previous: state before last update (e.g. prev. positions)
            gradient: current gradient (e.g. forces)
            gradient_previous: last gradient
            hessian: current hessian
            residual: ('max_atom' or 'norm') of residual gradient
        """
        ndim = np.size(coords)
        if hessian is not None:
            assert np.shape(hessian) == (ndim, ndim), (np.shape(hessian), ndim)
        elif alpha is not None:
            hessian = np.eye(ndim) / alpha
        else:
            hessian = np.eye(ndim) / self.default_configuration["alpha"]

        self.state = BFGSState(
            coords=coords, gradient=gradient, hessian=hessian
        )
        self._residual = residual

    def update(self, coords, gradient) -> None:
        """Take coords and gradient to update state"""
        self.state.update(coords=coords, gradient=gradient)

    def residual(self) -> float:
        """return residual force as norm of gradient"""
        if self._residual.lower() == "norm":
            return np.linalg.norm(self.state.gradient)
        elif self._residual.lower() == "max_atom":
            return np.linalg.norm(self.state.gradient, axis=1)
        else:
            msg = f"residual measure {self._residual} not implemented"
            raise ValueError(msg)


# fmt: off
class BFGS(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'alpha': 70.0}

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None):
        """BFGS optimizer.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.
        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep

        if self.maxstep > 1.0:
            warnings.warn('You are using a *very* large value for '
                          'the maximum step size: %.1f Å' % maxstep)

        if alpha is None:
            self.alpha = self.defaults['alpha']
        else:
            self.alpha = alpha

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

    def todict(self):
        d = Optimizer.todict(self)
        if hasattr(self, 'maxstep'):
            d.update(maxstep=self.maxstep)
        return d

    def initialize(self):
        # initial hessian
        self.H0 = np.eye(3 * len(self.atoms)) * self.alpha

        self.H = None
        self.r0 = None
        self.f0 = None

    def read(self):
        self.H, self.r0, self.f0, self.maxstep = self.load()

    def step(self, f=None):
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        f = f.reshape(-1)
        self.update(r.flat, f, self.r0, self.f0)
        omega, V = eigh(self.H)

        # FUTURE: Log this properly
        # # check for negative eigenvalues of the hessian
        # if any(omega < 0):
        #     n_negative = len(omega[omega < 0])
        #     msg = '\n** BFGS Hessian has {} negative eigenvalues.'.format(
        #         n_negative
        #     )
        #     print(msg, flush=True)
        #     if self.logfile is not None:
        #         self.logfile.write(msg)
        #         self.logfile.flush()

        dr = (V @ (f @ V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)
        atoms.set_positions(r + dr)
        self.r0 = r.flat.copy()
        self.f0 = f.copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            # FUTURE: Log this properly
            # msg = '\n** scale step by {:.3f} to be shorter than {}'.format(
            #     scale, self.maxstep
            # )
            # print(msg, flush=True)

            dr *= scale

        return dr

    def update(self, r, f, r0, f0):
        if self.H is None:
            self.H = self.H0
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = dr @ df
        dg = self.H @ dr
        b = dr @ dg
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        if isinstance(traj, str):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        self.H = None
        atoms = traj[0]
        r0 = atoms.get_positions().ravel()
        f0 = atoms.get_forces().ravel()
        for atoms in traj:
            r = atoms.get_positions().ravel()
            f = atoms.get_forces().ravel()
            self.update(r, f, r0, f0)
            r0 = r
            f0 = f

        self.r0 = r0
        self.f0 = f0


class oldBFGS(BFGS):
    def determine_step(self, dr, steplengths):
        """Old BFGS behaviour for scaling step lengths

        This keeps the behaviour of truncating individual steps. Some might
        depend of this as some absurd kind of stimulated annealing to find the
        global minimum.
        """
        dr /= np.maximum(steplengths / self.maxstep, 1.0).reshape(-1, 1)
        return dr
