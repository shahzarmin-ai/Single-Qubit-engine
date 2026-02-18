"""
qubit.py — Core quantum state representation using density matrices.
Supports both pure and mixed states, enabling noise simulation.
"""

import numpy as np
from typing import Tuple, Optional


class Qubit:
    """
    A single qubit represented as a density matrix ρ (2×2 complex Hermitian matrix).
    
    Pure state |ψ⟩ = α|0⟩ + β|1⟩  →  ρ = |ψ⟩⟨ψ|
    Mixed state: convex combination of pure state density matrices.
    
    Advantages over state-vector representation:
    - Can represent mixed states (statistical mixtures, decoherence)
    - Naturally handles noise channels
    - Purity tr(ρ²) ∈ (0.5, 1] distinguishes pure from mixed states
    """

    def __init__(self, alpha: complex = 1.0, beta: complex = 0.0):
        """
        Initialize qubit from amplitudes α|0⟩ + β|1⟩.
        Automatically normalizes the input vector.
        """
        state = np.array([alpha, beta], dtype=complex)
        norm = np.linalg.norm(state)
        if norm < 1e-10:
            raise ValueError("State vector cannot be zero.")
        state /= norm
        # ρ = |ψ⟩⟨ψ|
        self._rho = np.outer(state, state.conj())

    @classmethod
    def from_density_matrix(cls, rho: np.ndarray) -> "Qubit":
        """Create a Qubit directly from a density matrix."""
        q = cls.__new__(cls)
        rho = np.array(rho, dtype=complex)
        _validate_density_matrix(rho)
        q._rho = rho.copy()
        return q

    @classmethod
    def zero(cls) -> "Qubit":
        """|0⟩ computational basis state."""
        return cls(1.0, 0.0)

    @classmethod
    def one(cls) -> "Qubit":
        """|1⟩ computational basis state."""
        return cls(0.0, 1.0)

    @classmethod
    def plus(cls) -> "Qubit":
        """|+⟩ = (|0⟩ + |1⟩)/√2  — X-basis eigenstate."""
        return cls(1 / np.sqrt(2), 1 / np.sqrt(2))

    @classmethod
    def minus(cls) -> "Qubit":
        """|-⟩ = (|0⟩ - |1⟩)/√2  — X-basis eigenstate."""
        return cls(1 / np.sqrt(2), -1 / np.sqrt(2))

    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> "Qubit":
        """
        Create a pure state from Bloch sphere angles.
        θ ∈ [0, π], φ ∈ [0, 2π)
        |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        """
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        return cls(alpha, beta)

    @property
    def rho(self) -> np.ndarray:
        """The 2×2 density matrix."""
        return self._rho.copy()

    @property
    def purity(self) -> float:
        """tr(ρ²) ∈ (0.5, 1]. Pure state = 1, maximally mixed = 0.5."""
        return float(np.real(np.trace(self._rho @ self._rho)))

    @property
    def is_pure(self) -> bool:
        return abs(self.purity - 1.0) < 1e-6

    @property
    def bloch_vector(self) -> Tuple[float, float, float]:
        """
        Compute (x, y, z) Bloch vector from Pauli expectation values.
        ⟨σ_x⟩ = tr(ρ X), ⟨σ_y⟩ = tr(ρ Y), ⟨σ_z⟩ = tr(ρ Z)
        Mixed states have |r| < 1 (inside the sphere).
        """
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        x = float(np.real(np.trace(self._rho @ X)))
        y = float(np.real(np.trace(self._rho @ Y)))
        z = float(np.real(np.trace(self._rho @ Z)))
        return (x, y, z)

    @property
    def prob_zero(self) -> float:
        """Probability of measuring |0⟩. P(0) = ρ_{00}."""
        return float(np.real(self._rho[0, 0]))

    @property
    def prob_one(self) -> float:
        """Probability of measuring |1⟩. P(1) = ρ_{11}."""
        return float(np.real(self._rho[1, 1]))

    def measure(self) -> Tuple[int, "Qubit"]:
        """
        Projective measurement in the computational basis.
        Returns (outcome, post_measurement_state).
        
        Born's rule: P(0) = ⟨0|ρ|0⟩ = ρ_{00}
        Post-measurement collapse: ρ → |outcome⟩⟨outcome|
        """
        outcome = int(np.random.choice([0, 1], p=[self.prob_zero, self.prob_one]))
        collapsed = Qubit.zero() if outcome == 0 else Qubit.one()
        return outcome, collapsed

    def apply(self, gate: np.ndarray) -> "Qubit":
        """
        Apply a unitary gate U: ρ → U ρ U†
        Returns a new Qubit (immutable transformation).
        """
        U = np.array(gate, dtype=complex)
        new_rho = U @ self._rho @ U.conj().T
        return Qubit.from_density_matrix(new_rho)

    def von_neumann_entropy(self) -> float:
        """
        S(ρ) = -tr(ρ log ρ) — entropy of the quantum state.
        Pure states: S = 0. Maximally mixed: S = log(2) ≈ 0.693.
        """
        eigenvalues = np.linalg.eigvalsh(self._rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))

    def fidelity(self, other: "Qubit") -> float:
        """
        F(ρ, σ) = tr(√(√ρ σ √ρ))²
        For pure states simplifies to |⟨ψ|φ⟩|².
        """
        sqrt_rho = _matrix_sqrt(self._rho)
        product = sqrt_rho @ other._rho @ sqrt_rho
        sqrt_product = _matrix_sqrt(product)
        return float(np.real(np.trace(sqrt_product)) ** 2)

    def __repr__(self) -> str:
        x, y, z = self.bloch_vector
        return (
            f"Qubit(purity={self.purity:.4f}, "
            f"P(0)={self.prob_zero:.4f}, P(1)={self.prob_one:.4f}, "
            f"bloch=({x:.3f}, {y:.3f}, {z:.3f}))"
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _validate_density_matrix(rho: np.ndarray) -> None:
    assert rho.shape == (2, 2), "Density matrix must be 2×2"
    assert np.allclose(rho, rho.conj().T, atol=1e-8), "ρ must be Hermitian"
    assert abs(np.trace(rho) - 1.0) < 1e-8, "tr(ρ) must equal 1"
    eigenvalues = np.linalg.eigvalsh(rho)
    assert np.all(eigenvalues >= -1e-8), "ρ must be positive semidefinite"


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Compute matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T
