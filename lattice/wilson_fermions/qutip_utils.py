import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import warnings

################################################################################################
# 1. Set up plotting capabilities for qutip states
################################################################################################

def plot_operator(operator, part='real'):
    """Plots a qutip Operator)"""

    if part == 'real':
        if np.any(np.iscomplex(operator)):
            warnings.warn('Operator has nonzero imaginary part')
        plt.matshow(operator.data.todense().real)

    elif part == 'imag':
        plt.matshow(operator.data.todense().imag)
        if np.any(np.isreal(operator)):
            warnings.warn('Operator has nonzero real part')

    elif part == 'both':
        plt.matshow(operator.data.todense().real)
        plt.colorbar();
        plt.matshow(operator.data.todense().imag)

    plt.colorbar();


def plot_evolution(observable, result, **kwargs):
    """Plots the time evoluton of the given observable (given as qutip Qobj)"""
    obs = [qt.expect(observable, state) for state in result.states]

    plt.plot(result.times, obs, **kwargs)
    plt.fill_between(result.times, obs, alpha=0.2)
    plt.xlabel('Time')


def plot_overlap(phi, result, **kwargs):
    """Plots the time evoluton of the given observable (given as qutip Qobj)"""
    obs = [abs(phi.overlap(psi_t)) ** 2 for psi_t in result.states]

    plt.plot(result.times, obs, **kwargs)
    plt.fill_between(result.times, obs, alpha=0.2)
    plt.xlabel('Time t')
    plt.ylabel(r'Probability   $|\langle \phi | e^{-iHt} | \psi_0 \rangle|^2$')