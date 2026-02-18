"""
gates.py — Quantum gate library.
All gates are 2×2 unitary matrices (U†U = I).
Physical interpretation: each gate is a rotation on the Bloch Sphere.
"""

import numpy as np
from typing import Optional


# ── Single-Qubit Gates ────────────────────────────────────────────────────────

class Gates:
    """
    Standard single-qubit gate library.
    All gates are unitary: U†U = UU† = I
    """

    # Computational basis states
    KET_0 = np.array([[1], [0]], dtype=complex)
    KET_1 = np.array([[0], [1]], dtype=complex)

    @staticmethod
    def I() -> np.ndarray:
        """Identity gate — no operation."""
        return np.eye(2, dtype=complex)

    @staticmethod
    def X() -> np.ndarray:
        """
        Pauli-X (NOT gate) — π rotation around X-axis.
        Flips |0⟩ ↔ |1⟩. Classical analog: bit flip.
        """
        return np.array([[0, 1], [1, 0]], dtype=complex)

    @staticmethod
    def Y() -> np.ndarray:
        """
        Pauli-Y — π rotation around Y-axis.
        Combines bit flip and phase flip.
        """
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    @staticmethod
    def Z() -> np.ndarray:
        """
        Pauli-Z (phase flip) — π rotation around Z-axis.
        Maps |0⟩→|0⟩, |1⟩→-|1⟩. No classical analog.
        """
        return np.array([[1, 0], [0, -1]], dtype=complex)

    @staticmethod
    def H() -> np.ndarray:
        """
        Hadamard gate — π rotation around (X+Z)/√2 axis.
        Creates equal superposition: H|0⟩ = |+⟩, H|1⟩ = |-⟩.
        H² = I (self-inverse).
        """
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def S() -> np.ndarray:
        """
        Phase gate (S gate) — π/2 rotation around Z-axis.
        Maps |1⟩ → i|1⟩. S = √Z.
        """
        return np.array([[1, 0], [0, 1j]], dtype=complex)

    @staticmethod
    def T() -> np.ndarray:
        """
        T gate (π/8 gate) — π/4 rotation around Z-axis.
        T = √S = Z^(1/4). Critical for universal quantum computation.
        """
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """
        Rotation around X-axis by angle θ.
        Rx(θ) = exp(-iθX/2) = I·cos(θ/2) - iX·sin(θ/2)
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """
        Rotation around Y-axis by angle θ.
        Ry(θ) = exp(-iθY/2) = I·cos(θ/2) - iY·sin(θ/2)
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """
        Rotation around Z-axis by angle θ.
        Rz(θ) = exp(-iθZ/2) = diag(e^(-iθ/2), e^(iθ/2))
        """
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=complex,
        )

    @staticmethod
    def P(phi: float) -> np.ndarray:
        """
        Phase shift gate — adds phase e^(iφ) to |1⟩ component.
        P(π/2) = S,  P(π/4) = T,  P(π) = Z
        """
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

    @staticmethod
    def U(theta: float, phi: float, lam: float) -> np.ndarray:
        """
        General single-qubit unitary gate (IBM convention).
        Any single-qubit gate can be expressed in this form.
        U(θ,φ,λ) = Rz(φ) Ry(θ) Rz(λ)
        """
        ct = np.cos(theta / 2)
        st = np.sin(theta / 2)
        return np.array(
            [
                [ct, -np.exp(1j * lam) * st],
                [np.exp(1j * phi) * st, np.exp(1j * (phi + lam)) * ct],
            ],
            dtype=complex,
        )

    @staticmethod
    def is_unitary(gate: np.ndarray, atol: float = 1e-8) -> bool:
        """Verify U†U = I."""
        product = gate.conj().T @ gate
        return np.allclose(product, np.eye(2), atol=atol)
