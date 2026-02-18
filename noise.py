"""
noise.py — Quantum noise channels via Kraus operators.

Real quantum hardware is noisy. This module implements the most
physically relevant single-qubit noise models using the Kraus formalism.

A quantum channel E acts on a density matrix as:
    ρ → E(ρ) = Σ_k K_k ρ K_k†
where Kraus operators satisfy: Σ_k K_k†K_k = I (trace-preserving).
"""

import numpy as np
from typing import List
from .qubit import Qubit


class NoiseChannel:
    """Base class for quantum noise channels."""

    def apply(self, qubit: Qubit) -> Qubit:
        raise NotImplementedError

    def _apply_kraus(self, rho: np.ndarray, kraus_ops: List[np.ndarray]) -> np.ndarray:
        """Apply Kraus operator sum: ρ → Σ_k K_k ρ K_k†"""
        new_rho = sum(K @ rho @ K.conj().T for K in kraus_ops)
        return new_rho


class AmplitudeDamping(NoiseChannel):
    """
    Amplitude Damping Channel — models energy loss (T1 relaxation).
    
    Physical interpretation: spontaneous emission of a photon.
    A qubit in |1⟩ (excited state) decays to |0⟩ (ground state).
    
    γ = 1 - exp(-t/T1) is the damping parameter (0 ≤ γ ≤ 1).
    
    Kraus operators:
        K0 = [[1, 0], [0, √(1-γ)]]   (no decay)
        K1 = [[0, √γ], [0, 0]]        (decay event)
    """

    def __init__(self, gamma: float):
        assert 0 <= gamma <= 1, "Damping parameter γ must be in [0, 1]"
        self.gamma = gamma

    def apply(self, qubit: Qubit) -> Qubit:
        g = self.gamma
        K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
        new_rho = self._apply_kraus(qubit.rho, [K0, K1])
        return Qubit.from_density_matrix(new_rho)


class PhaseDamping(NoiseChannel):
    """
    Phase Damping (Dephasing) Channel — models T2 decoherence.
    
    Physical interpretation: random phase kicks destroy quantum coherence
    without changing energy. The off-diagonal elements (coherences) decay.
    
    λ = decoherence probability (0 ≤ λ ≤ 1).
    
    This is the fundamental channel that turns quantum superpositions
    into classical probability distributions.
    """

    def __init__(self, lam: float):
        assert 0 <= lam <= 1, "Dephasing parameter λ must be in [0, 1]"
        self.lam = lam

    def apply(self, qubit: Qubit) -> Qubit:
        l = self.lam
        K0 = np.array([[1, 0], [0, np.sqrt(1 - l)]], dtype=complex)
        K1 = np.array([[0, 0], [0, np.sqrt(l)]], dtype=complex)
        new_rho = self._apply_kraus(qubit.rho, [K0, K1])
        return Qubit.from_density_matrix(new_rho)


class Depolarizing(NoiseChannel):
    """
    Depolarizing Channel — symmetric noise model.
    
    Physical interpretation: with probability p, the qubit undergoes
    a random Pauli error (X, Y, or Z with equal probability p/3 each).
    With probability 1-p, nothing happens.
    
    ρ → (1 - p)ρ + (p/3)(XρX + YρY + ZρZ)
    
    Maps any state toward the maximally mixed state I/2.
    Standard benchmark noise model for quantum error correction.
    """

    def __init__(self, p: float):
        assert 0 <= p <= 1, "Error probability p must be in [0, 1]"
        self.p = p

    def apply(self, qubit: Qubit) -> Qubit:
        p = self.p
        rho = qubit.rho
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        new_rho = (
            (1 - p) * rho
            + (p / 3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)
        )
        return Qubit.from_density_matrix(new_rho)


class BitFlip(NoiseChannel):
    """
    Bit Flip Channel — classical-like noise.
    With probability p: X error (|0⟩↔|1⟩).
    With probability 1-p: no error.
    """

    def __init__(self, p: float):
        assert 0 <= p <= 1
        self.p = p

    def apply(self, qubit: Qubit) -> Qubit:
        p = self.p
        K0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
        K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=complex)
        new_rho = self._apply_kraus(qubit.rho, [K0, K1])
        return Qubit.from_density_matrix(new_rho)


class PhaseFlip(NoiseChannel):
    """
    Phase Flip Channel — quantum-specific noise.
    With probability p: Z error (phase flip).
    With probability 1-p: no error.
    """

    def __init__(self, p: float):
        assert 0 <= p <= 1
        self.p = p

    def apply(self, qubit: Qubit) -> Qubit:
        p = self.p
        K0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
        K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
        new_rho = self._apply_kraus(qubit.rho, [K0, K1])
        return Qubit.from_density_matrix(new_rho)
