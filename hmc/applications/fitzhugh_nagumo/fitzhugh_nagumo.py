from typing import Callable, Optional, Tuple

import numba as nb
import numpy as np
import scipy.linalg as spla
import scipy.stats as spst
from scipy.integrate import odeint


def generate_data(
        state: np.ndarray,
        t: np.ndarray,
        sigma: float,
        a: float=0.2,
        b: float=0.2,
        c: float=3.0,
        rtol: float=None,
        atol: float=None,
        hmax: float=0.0,
        hmin: float=0.0,
        mxstep: int=0
) -> np.ndarray:
    """Generate random observations from the Fitzhugh-Nagumo model.

    Args:
        state: The current state of the system.
        t: The current time of the system.
        sigma: The noise level of the system.
        a: Parameter of the Fitzhugh-Nagumo model.
        b: Parameter of the Fitzhugh-Nagumo model.
        c: Parameter of the Fitzhugh-Nagumo model.
        rtol: Relative error tolerance of the numerical integrator.
        atol: Absolute error tolerance of the numerical integrator.
        hmax: Maximum integration step-size.
        mxstep: Maximum number of internal integration steps.

    Returns:
        y: Noise-corrupted observations of the Fitzhugh-Nagumo model.

    """
    y = odeint(fn_dynamics, state, t, (a, b, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    y += np.random.normal(0.0, sigma, size=y.shape)
    return y

@nb.njit
def fn_dynamics(state: np.ndarray, t: float, *args: Tuple) -> np.ndarray:
    """Definition of the dynamics for the Fitzhugh-Nagumo differential equation
    model.

    Args:
        state: The current state of the system.
        t: The current time of the system.
        args: Parameters of the Fitzhugh-Nagumo model.

    Returns:
        out: The time derivative of the state.

    """
    a, b, c = args
    v, r = state[0], state[1]
    ds = np.array([c * (v - np.power(v, 3.0) / 3.0 + r), -(v - a + b * r) / c])
    return ds

@nb.njit
def fn_sensitivity(state: np.ndarray, t: float, *args: Tuple) -> np.ndarray:
    """The sensitivity of the states of the Fitzhugh-Nagumo model with respect to
    the model parameters at each time.

    Args:
        state: The expanded state of the system, including states representing
            the sensitivity of the state with respect to the parameters.
        t: The current time of the system.
        args: Parameters of the Fitzhugh-Nagumo model.

    Returns:
        out: The time derivative of the expanded state.

    """
    # Parameters of the Fitzhugh-Nagumo model.
    a, b, c = args
    # Compute the dynamics of the Fitzhugh-Nagumo differential equation as we
    # progress through time.
    s = state[:2]
    ds = fn_dynamics(s, t, a, b, c)
    v, r = s[0], s[1]
    # Compute the state sensitivities.
    csq = np.square(c)
    fx = np.array([[c - c*np.square(v), c], [-1.0 / c,  -b / c]])
    fo = np.array([[0.0, 0.0, v - np.power(v, 3.0) / 3.0 + r],
                   [1.0 / c, -r / c, (v - a + b*r) / csq]])

    # Here is the layout of the sensitivities with the index of these
    # quantities in the output of the ODE solver shown in parentheses.
    #
    # Index      Interpretation
    # --------------------------------------------------
    # 0 (2)      Sensitivity of `v` with respect to `a`.
    # 1 (3)      Sensitivity of `r` with respect to `a`.
    # 2 (4)      Sensitivity of `v` with respect to `b`.
    # 3 (5)      Sensitivity of `r` with respect to `b`.
    # 4 (6)      Sensitivity of `v` with respect to `c`.
    # 5 (7)      Sensitivity of `r` with respect to `c`.
    # --------------------------------------------------
    va, ra, vb, rb, vc, rc = state[2:8]
    vsq = np.square(v)
    vsq_m_one = vsq - 1.0
    neg_b_div_c = -b/c
    dS = np.array([
        (-c*vsq_m_one)*va + c*ra,
        -va/c + neg_b_div_c*ra + 1/c,
        (-c*vsq_m_one)*vb + c*rb,
        -vb/c + neg_b_div_c*rb- r/c,
        (-c*vsq_m_one)*vc + (c)*rc + v - v**3/3 + r,
        -vc/c + neg_b_div_c*rc + (v - a + r*b)/csq
    ])
    # Stack the state dynamics and the sensitivity dynamics.
    ret = np.hstack((ds, dS))
    return ret

@nb.njit
def fn_higher_sensitivity(state: np.ndarray, t: float, *args: Tuple) -> np.ndarray:
    # Unpack variables and computed repeated quantities.
    a, b, c, correct = args
    v, r = state[:2]
    va, ra, vb, rb, vc, rc = state[2:8]
    vaa, raa, vab, rab, vac, rac, vbb, rbb, vbc, rbc, vcc, rcc = state[8:20]
    two_c_v = 2*c*v
    vsq = np.square(v)
    vsq_m_one = vsq - 1.0
    neg_b_div_c = -b/c
    csq = np.square(c)

    # Decide whether or not to break HMC by invalidating the symmetry of
    # partial derivatives.
    if correct:
        d1 = -vsq_m_one*va - two_c_v*va*vc - c*vsq_m_one*vac + ra + c*rac
        d2 = -vsq_m_one*vb - two_c_v*vc*vb - c*vsq_m_one*vbc + rb + c*rbc
        d3 = -vsq_m_one*vc - two_c_v*vc*vc - c*vsq_m_one*vcc + 2*rc + c*rcc + vc - vsq*vc
    else:
        d1 = -two_c_v*vc*va + 1 - vsq*va - c*vsq_m_one*vac + ra + c*rac
        d2 = -two_c_v*vc*vb + 1 - vsq*vb - c*vsq_m_one*vbc + rb + c*rbc
        d3 = -two_c_v*vc*vc + 1 - vsq*vc - c*vsq_m_one*vcc + 1 - vsq*vc + 2*rc + c*rcc

    # Index        Interpretation
    # ------------------------------------------------------------------------
    # 0 (8)        Sensitivity of `v` with respect to `a` with respect to `a`.
    # 1 (9)        Sensitivity of `r` with respect to `a` with respect to `a`.
    # 2 (10)       Sensitivity of `v` with respect to `a` with respect to `b`.
    # 3 (11)       Sensitivity of `r` with respect to `a` with respect to `b`.
    # 4 (12)       Sensitivity of `v` with respect to `a` with respect to `c`.
    # 5 (13)       Sensitivity of `r` with respect to `a` with respect to `c`.
    # 6 (14)       Sensitivity of `v` with respect to `b` with respect to `b`.
    # 7 (15)       Sensitivity of `r` with respect to `b` with respect to `b`.
    # 8 (16)       Sensitivity of `v` with respect to `b` with respect to `c`.
    # 9 (17)       Sensitivity of `r` with respect to `b` with respect to `c`.
    # 10 (18)      Sensitivity of `v` with respect to `c` with respect to `c`.
    # 11 (19)      Sensitivity of `r` with respect to `c` with respect to `c`.
    # ------------------------------------------------------------------------
    dsens = np.array([
        -two_c_v*va*va - c*vsq_m_one*vaa + c*raa,
        -vaa/c - b/c*raa,
        -two_c_v*vb*va - c*vsq_m_one*vab + c*rab,
        -vab/c - ra/c - b/c*rab,
        d1,
        va/csq - vac/c + b/csq*ra - b/c*rac - 1/csq,
        -two_c_v*vb*vb - c*vsq_m_one*vbb + c*rbb,
        -vbb/c - rb/c - b/c*rbb - rb/c,
        d2,
        vb/csq - vbc/c + b/csq*rb - b/c*rbc - rc/c + r/csq,
        d3,
        vc/csq - vcc/c + vc/csq + b/csq*rc - b/c*rcc + b/csq*rc - (2*(v - a + r*b))/c**3
    ])
    # Stack the time derivative of the sensitivity dynamics.
    sens = fn_sensitivity(state, t, a, b, c)
    ret = np.hstack((sens, dsens))
    return ret

def six_to_nine(a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """Helper function to stack matrices with redundancy."""
    return np.array([
        [a, b, c],
        [b, d, e],
        [c, e, f]
    ])

def posterior_factory(
        state: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        sigma: float,
        rtol: float=None,
        atol: float=None,
        hmax: float=0.0,
        hmin: float=0.0,
        mxstep: int=0,
        correct: bool=True
) -> Tuple[Callable]:
    """Factory function that yields functions to compute the log-posterior of the
    Fitzhugh-Nagumo model, the gradient of the log-posterior, the Fisher
    information metric and the gradient of the Fisher information metric.

    Args:
        state: Initial state of the Fitzhugh-Nagumo dynamics.
        y: Observations from the Fitzhugh-Nagumo model.
        t: Time points at which observations from the Fitzhugh-Nagumo model
            were collected.
        sigma: Standard deviation of noise.
        rtol: Relative error tolerance of the numerical integrator.
        atol: Absolute error tolerance of the numerical integrator.
        hmax: Maximum integration step-size.
        hmin: Minimum integration step-size.
        mxstep: Maximum number of internal integration steps.
        correct: Whether or not to invalidate the detailed balance of HMC by
            breaking the symmetry of partial derivatives.

    Returns:
        log_posterior: Function to compute the log-posterior.
        grad_log_posterior: Function to compute the gradient of the log-posterior.
        metric: Function to compute the Fisher information metric.
        grad_metric: Function to compute the gradient of the Fisher information
            metric.
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, the Fisher information metric, and the derivatives of
            the Fisher information metric.

    """
    # Precompute assumed noise variance.
    sigmasq = np.square(sigma)
    # Precompute identity matrix.
    Id = np.eye(3)
    # Precompute augmented state with and without higher order sensitivities.
    aug = np.hstack((state, np.zeros(6)))
    augh = np.hstack((state, np.zeros(18)))

    def _odeint(dynamics, state, t, params):
        return odeint(dynamics, state, t, params, rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)

    def _log_posterior_helper(sens: np.ndarray, a: float, b: float, c: float) -> float:
        yh = sens[:, :2]
        ll = -0.5*np.sum(np.square((y - yh) / sigma)) \
            - np.log(sigma) - 0.5*np.log(2*np.pi)
        lpa = -0.5*a**2
        lpb = -0.5*b**2
        lpc = -0.5*c**2
        lp = ll + lpa + lpb + lpc
        return lp

    def _grad_log_posterior_helper(sens: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        yh = sens[:, :2]
        dsa, dsb, dsc = sens[:, 2:4], sens[:, 4:6], sens[:, 6:8]
        r = (y - yh) / sigmasq
        da = np.sum(r*dsa) - a
        db = np.sum(r*dsb) - b
        dc = np.sum(r*dsc) - c
        return np.array([da, db, dc])

    def _metric_helper(sens: np.ndarray) -> np.ndarray:
        dsa, dsb, dsc = sens[:, 2:4], sens[:, 4:6], sens[:, 6:8]
        Ga = dsa[:, 0]@dsa[:, 0] + dsa[:, 1]@dsa[:, 1]
        Gb = dsa[:, 0]@dsb[:, 0] + dsa[:, 1]@dsb[:, 1]
        Gc = dsa[:, 0]@dsc[:, 0] + dsa[:, 1]@dsc[:, 1]
        Gd = dsb[:, 0]@dsb[:, 0] + dsb[:, 1]@dsb[:, 1]
        Ge = dsb[:, 0]@dsc[:, 0] + dsb[:, 1]@dsc[:, 1]
        Gf = dsc[:, 0]@dsc[:, 0] + dsc[:, 1]@dsc[:, 1]
        G = six_to_nine(Ga, Gb, Gc, Gd, Ge, Gf)
        G /= sigmasq
        # Add in the negative Hessian of the log-prior.
        G += Id
        return G

    def _grad_metric_helper(sens: np.ndarray) -> np.ndarray:
        S, dS = sens[:, 2:8], sens[:, 8:]
        dG = np.zeros((3, 3, 3))

        # Derivatives with respect to `a`.
        dGaa = 2*dS[:, 0]@S[:, 0] + 2*dS[:, 1]@S[:, 1]
        dGab = dS[:, 0]@S[:, 2] + S[:, 0]@dS[:, 2] + dS[:, 1]@S[:, 3] + S[:, 1]@dS[:, 3]
        dGac = dS[:, 0]@S[:, 4] + S[:, 0]@dS[:, 4] + dS[:, 1]@S[:, 5] + S[:, 1]@dS[:, 5]
        dGad = 2*dS[:, 2]@S[:, 2] + 2*dS[:, 3]@S[:, 3]
        dGae = dS[:, 2]@S[:, 4] + S[:, 2]@dS[:, 4] + dS[:, 3]@S[:, 5] + S[:, 3]@dS[:, 5]
        dGaf = 2*dS[:, 4]@S[:, 4] + 2*dS[:, 5]@S[:, 5]
        dGa = six_to_nine(dGaa, dGab, dGac, dGad, dGae, dGaf)

        # Derivatives with respect to `b`.
        dGba = 2*dS[:, 2]@S[:, 0] + 2*dS[:, 3]@S[:, 1]
        dGbb = dS[:, 2]@S[:, 2] + S[:, 0]@dS[:, 6] + dS[:, 3]@S[:, 3] + S[:, 1]@dS[:, 7]
        dGbc = dS[:, 2]@S[:, 4] + S[:, 0]@dS[:, 8] + dS[:, 3]@S[:, 5] + S[:, 1]@dS[:, 9]
        dGbd = 2*dS[:, 6]@S[:, 2] + 2*dS[:, 7]@S[:, 3]
        dGbe = dS[:, 6]@S[:, 4] + S[:, 2]@dS[:, 8] + dS[:, 7]@S[:, 5] + S[:, 3]@dS[:, 9]
        dGbf = 2*dS[:, 8]@S[:, 4] + 2*dS[:, 9]@S[:, 5]
        dGb = six_to_nine(dGba, dGbb, dGbc, dGbd, dGbe, dGbf)

        # Derivatives with respect to `c`.
        dGca = 2*dS[:, 4]@S[:, 0] + 2*dS[:, 5]@S[:, 1]
        dGcb = dS[:, 4]@S[:, 2] + S[:, 0]@dS[:, 8] + dS[:, 5]@S[:, 3] + S[:, 1]@dS[:, 9]
        dGcc = dS[:, 4]@S[:, 4] + S[:, 0]@dS[:, 10] + dS[:, 5]@S[:, 5] + S[:, 1]@dS[:, 11]
        dGcd = 2*dS[:, 8]@S[:, 2] + 2*dS[:, 9]@S[:, 3]
        dGce = dS[:, 8]@S[:, 4] + S[:, 2]@dS[:, 10] + dS[:, 9]@S[:, 5] + S[:, 3]@dS[:, 11]
        dGcf = 2*dS[:, 10]@S[:, 4] + 2*dS[:, 11]@S[:, 5]
        dGc = six_to_nine(dGca, dGcb, dGcc, dGcd, dGce, dGcf)

        # Stack the component matrices.
        dG = np.array([dGa, dGb, dGc]).swapaxes(0, -1)
        dG /= sigmasq
        return dG

    def _log_posterior(a: float, b: float, c: float) -> float:
        """Posterior distribution of the Fitzhugh-Nagumo differential equation model.
        The model places a normal likelihood over observations from an
        integrated trajectory of a Fitzhugh-Nagumo ODE. There is also a
        standard normal prior over the model parameters.

        Args:
            a: Parameter of the Fitzhugh-Nagumo model.
            b: Parameter of the Fitzhugh-Nagumo model.
            c: Parameter of the Fitzhugh-Nagumo model.

        Returns:
            lp: The log-posterior of the Fitzhugh-Nagumo model.

        """
        yh = _odeint(fn_dynamics, state, t, (a, b, c))
        lp = _log_posterior_helper(yh, a, b, c)
        return lp

    def _grad_log_posterior(a: float, b: float, c: float) -> np.ndarray:
        """The gradient of the log-posterior of the Fitzhugh-Nagumo model with respect
        to the model parameters.

        Args:
            a: Parameter of the Fitzhugh-Nagumo model.
            b: Parameter of the Fitzhugh-Nagumo model.
            c: Parameter of the Fitzhugh-Nagumo model.

        Returns:
            glp: The gradient of the log-posterior with respect to the model
                parameters.

        """
        sens = _odeint(fn_sensitivity, aug, t, (a, b, c))
        glp = _grad_log_posterior_helper(sens, a, b, c)
        return glp

    def _metric(a: float, b: float, c: float) -> np.ndarray:
        """The Fisher information metric of the Fitzhugh-Nagumo model. The sensitivity
        differential equation allows us to propogate the derivatives of the
        trajectory states into the metric.

        Args:
            a: Parameter of the Fitzhugh-Nagumo model.
            b: Parameter of the Fitzhugh-Nagumo model.
            c: Parameter of the Fitzhugh-Nagumo model.

        Returns:
            G: The Fisher information metric of the Fitzhugh-Nagumo model.

        """
        sens = _odeint(fn_sensitivity, aug, t, (a, b, c))
        G = _metric_helper(sens)
        return G

    def _grad_metric(a: float, b: float, c: float) -> np.ndarray:
        """The gradient of the Fisher information metric of the Fitzhugh-Nagumo model
        with respect to the model parameters.

        Args:
            a: Parameter of the Fitzhugh-Nagumo model.
            b: Parameter of the Fitzhugh-Nagumo model.
            c: Parameter of the Fitzhugh-Nagumo model.

        Returns:
            dG: The gradient of the Fisher information metric of the
                Fitzhugh-Nagumo model.

        """
        sens = _odeint(fn_higher_sensitivity, augh, t, (a, b, c, correct))
        dG = _grad_metric_helper(sens)
        return dG

    def _euclidean_auxiliaries(a: float, b: float, c: float) -> Tuple[np.ndarray]:
        """Function to compute the log-posterior and the gradient of the log-posterior.

        Args:
            a: Parameter of the Fitzhugh-Nagumo model.
            b: Parameter of the Fitzhugh-Nagumo model.
            c: Parameter of the Fitzhugh-Nagumo model.

        Returns:
            lp: The log-posterior of the Fitzhugh-Nagumo posterior.
            glp: The gradient of the log-posterior of the Fitzhugh-Nagumo
                posterior.

        """
        sens = _odeint(fn_sensitivity, aug, t, (a, b, c))
        lp = _log_posterior_helper(sens, a, b, c)
        glp = _grad_log_posterior_helper(sens, a, b, c)
        return lp, glp

    def _riemannian_auxiliaries(a: float, b: float, c: float) -> Tuple[np.ndarray]:
        """Function to compute the log-posterior, the gradient of the log-posterior,
        the Fisher information metric and the derivatives of the Fisher
        information metric of the Fitzhugh-Nagumo model with respect to model
        parameters.

        Args:
            a: Parameter of the Fitzhugh-Nagumo model.
            b: Parameter of the Fitzhugh-Nagumo model.
            c: Parameter of the Fitzhugh-Nagumo model.

        Returns:
            lp: The log-posterior of the Fitzhugh-Nagumo posterior.
            glp: The gradient of the log-posterior of the Fitzhugh-Nagumo
                posterior.
            G: The Fisher information metric of the Fitzhugh-Nagumo distribution.
            dG: The gradient of the Fisher information metric with respect to the
                model parameters.

        """
        sens = _odeint(fn_higher_sensitivity, augh, t, (a, b, c, correct))
        lp = _log_posterior_helper(sens, a, b, c)
        glp = _grad_log_posterior_helper(sens, a, b, c)
        G = _metric_helper(sens)
        dG = _grad_metric_helper(sens)
        return lp, glp, G, dG

    def log_posterior_and_metric(q):
        a, b, c = q
        sens = _odeint(fn_sensitivity, aug, t, (a, b, c))
        lp = _log_posterior_helper(sens, a, b, c)
        G = _metric_helper(sens)
        return lp, G

    # Convert functions that whose inputs were `a`, `b`, and `c` to take a
    # single vector-valued argument representing the concatenation of those
    # variables.
    log_posterior = lambda q: _log_posterior(q[0], q[1], q[2])
    grad_log_posterior = lambda q: _grad_log_posterior(q[0], q[1], q[2])
    metric = lambda q: _metric(q[0], q[1], q[2])
    grad_metric = lambda q: _grad_metric(q[0], q[1], q[2])
    euclidean_auxiliaries = lambda q: _euclidean_auxiliaries(q[0], q[1], q[2])
    riemannian_auxiliaries = lambda q: _riemannian_auxiliaries(q[0], q[1], q[2])

    return (
        log_posterior,
        grad_log_posterior,
        metric,
        log_posterior_and_metric,
        grad_metric,
        euclidean_auxiliaries,
        riemannian_auxiliaries
    )


def main():
    import time

    from hmc.linalg import solve_psd

    # Integrator parameters.
    rtol = None
    atol = None
    hmax = 0.0
    hmin = 0.0
    mxstep = 0
    # Generate observations from the Fitzhugh-Nagumo model.
    a = 0.2
    b = 0.2
    c = 3.0
    sigma = 0.5

    state = np.array([-1.0, 1.0])
    t = np.linspace(0.0, 10.0, 200)
    start = time.time()
    y = generate_data(state, t, sigma, a, b, c, rtol, atol, hmax, hmin, mxstep=mxstep)
    elapsed = time.time() - start
    print('data generation elapsed: {:.4f}'.format(elapsed))

    # Compute the sensitivities of the states to finite difference perturbations.
    _fn_sensitivity = lambda t, state, *args: fn_sensitivity(state, t, *args)
    aug = np.hstack((state, np.zeros(6)))
    start = time.time()
    sens = odeint(fn_sensitivity, aug, t, (a, b, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    elapsed = time.time() - start
    print('state sensitivities elapsed: {:.4f}'.format(elapsed))

    delta = 1e-5
    yh = odeint(fn_dynamics, state, t, (a + 0.5*delta, b, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    yl = odeint(fn_dynamics, state, t, (a - 0.5*delta, b, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    fda = (yh - yl) / delta
    err = np.linalg.norm(sens[:, 2:4] - fda)
    print('state sensitivity `a`: {:.8f}'.format(err))

    yh = odeint(fn_dynamics, state, t, (a, b + 0.5*delta, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    yl = odeint(fn_dynamics, state, t, (a, b - 0.5*delta, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    fdb = (yh - yl) / delta
    err = np.linalg.norm(sens[:, 4:6] - fdb)
    print('state sensitivity `b`: {:.8f}'.format(err))

    yh = odeint(fn_dynamics, state, t, (a, b, c + 0.5*delta), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    yl = odeint(fn_dynamics, state, t, (a, b, c - 0.5*delta), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    fdc = (yh - yl) / delta
    err = np.linalg.norm(sens[:, 6:8] - fdc)
    print('state sensitivity `c`: {:.8f}'.format(err))

    # Check sensitivity of sensitivity with respect to `a`.
    augh = np.hstack((state, np.zeros(18)))
    sens = odeint(fn_higher_sensitivity, augh, t, (a, b, c, True), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    sh = odeint(fn_sensitivity, aug, t, (a + 0.5*delta, b, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    sl = odeint(fn_sensitivity, aug, t, (a - 0.5*delta, b, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    fds = (sh - sl) / delta
    print("""
    sensitivity of `v` with respect to `a` with respect to `a`: {:.8f}
    sensitivity of `v` with respect to `a` with respect to `b`: {:.8f}
    sensitivity of `v` with respect to `a` with respect to `c`: {:.8f}
    sensitivity of `r` with respect to `a` with respect to `a`: {:.8f}
    sensitivity of `r` with respect to `a` with respect to `b`: {:.8f}
    sensitivity of `r` with respect to `a` with respect to `c`: {:.8f}
    """.format(
        np.linalg.norm(fds[:, 2] - sens[:, 8]),
        np.linalg.norm(fds[:, 4] - sens[:, 10]),
        np.linalg.norm(fds[:, 6] - sens[:, 12]),
        np.linalg.norm(fds[:, 3] - sens[:, 9]),
        np.linalg.norm(fds[:, 5] - sens[:, 11]),
        np.linalg.norm(fds[:, 7] - sens[:, 13])
    ))

    # Check sensitivity of sensitivity with respect to `b`.
    sh = odeint(fn_sensitivity, aug, t, (a, b + 0.5*delta, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    sl = odeint(fn_sensitivity, aug, t, (a, b - 0.5*delta, c), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    fds = (sh - sl) / delta
    print("""
    sensitivity of `v` with respect to `a` with respect to `b`: {:.8f}
    sensitivity of `v` with respect to `b` with respect to `b`: {:.8f}
    sensitivity of `v` with respect to `b` with respect to `c`: {:.8f}
    sensitivity of `r` with respect to `a` with respect to `b`: {:.8f}
    sensitivity of `r` with respect to `b` with respect to `b`: {:.8f}
    sensitivity of `r` with respect to `b` with respect to `c`: {:.8f}
    """.format(
        np.linalg.norm(fds[:, 2] - sens[:, 10]),
        np.linalg.norm(fds[:, 4] - sens[:, 14]),
        np.linalg.norm(fds[:, 6] - sens[:, 16]),
        np.linalg.norm(fds[:, 3] - sens[:, 11]),
        np.linalg.norm(fds[:, 5] - sens[:, 15]),
        np.linalg.norm(fds[:, 7] - sens[:, 17])
    ))

    # Check sensitivity of sensitivity with respect to `c`.
    sh = odeint(fn_sensitivity, aug, t, (a, b, c + 0.5*delta), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    sl = odeint(fn_sensitivity, aug, t, (a, b, c - 0.5*delta), rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
    fds = (sh - sl) / delta
    print("""
    sensitivity of `v` with respect to `a` with respect to `c`: {:.8f}
    sensitivity of `v` with respect to `b` with respect to `c`: {:.8f}
    sensitivity of `v` with respect to `c` with respect to `c`: {:.8f}
    sensitivity of `r` with respect to `a` with respect to `c`: {:.8f}
    sensitivity of `r` with respect to `b` with respect to `c`: {:.8f}
    sensitivity of `r` with respect to `c` with respect to `c`: {:.8f}
    """.format(
        np.linalg.norm(fds[:, 2] - sens[:, 12]),
        np.linalg.norm(fds[:, 4] - sens[:, 16]),
        np.linalg.norm(fds[:, 6] - sens[:, 18]),
        np.linalg.norm(fds[:, 3] - sens[:, 13]),
        np.linalg.norm(fds[:, 5] - sens[:, 17]),
        np.linalg.norm(fds[:, 7] - sens[:, 19])
    ))

    # Check the gradient of the log-posterior against finite differences.
    (
        log_posterior,
        grad_log_posterior,
        metric,
        log_posterior_and_metric,
        grad_metric,
        euclidean_auxiliaries,
        riemannian_auxiliaries
    ) = posterior_factory(
        state,
        y,
        t,
        sigma,
        rtol=rtol,
        atol=atol,
        hmax=hmax,
        hmin=hmin,
        mxstep=mxstep
    )
    a, b, c = 0.1, 0.5, 2.0
    q = np.array([a, b, c])
    u = np.random.normal(size=q.shape)
    g = grad_log_posterior(q)@u
    delta = 1e-5
    fd = (log_posterior(q + 0.5*delta*u) - log_posterior(q - 0.5*delta*u)) / delta
    assert np.allclose(g, fd)

    err = g - fd
    rerr = err / np.linalg.norm(fd)
    print('log-posterior gradient abs. error: {:.5f} - rel. error: {:.5f}'.format(err, rerr))

    # Natural gradient method to check the usefulness of the metric.
    for i in range(20):
        lp, glp, G, dG = riemannian_auxiliaries(q)
        ng = solve_psd(G, glp)
        q += ng
        diff = lp - log_posterior(np.array([0.2, 0.2, 3.0]))
        print('iter. {} - difference of log-posteriors: {:.5f}'.format(i+1, diff))

    # Check gradient of metric.
    dG = grad_metric(q)
    delta = 1e-5
    u = np.array([1.0, 0.0, 0.0])
    fd = (metric(q + 0.5*delta*u) - metric(q - 0.5*delta*u)) / delta
    aerr = np.linalg.norm(fd - dG[..., 0])
    rerr = aerr / np.linalg.norm(fd)
    print('metric gradient `a` abs. error: {:.5f} - rel. error: {:.5f}'.format(aerr, rerr))
    u = np.array([0.0, 1.0, 0.0])
    fd = (metric(q + 0.5*delta*u) - metric(q - 0.5*delta*u)) / delta
    aerr = np.linalg.norm(fd - dG[..., 1])
    rerr = aerr / np.linalg.norm(fd)
    print('metric gradient `b` abs. error: {:.5f} - rel. error: {:.5f}'.format(aerr, rerr))
    u = np.array([0.0, 0.0, 1.0])
    fd = (metric(q + 0.5*delta*u) - metric(q - 0.5*delta*u)) / delta
    aerr = np.linalg.norm(fd - dG[..., 2])
    rerr = aerr / np.linalg.norm(fd)
    print('metric gradient `c` abs. error: {:.5f} - rel. error: {:.5f}'.format(aerr, rerr))


if __name__ == '__main__':
    main()
