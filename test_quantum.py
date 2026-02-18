"""
tests/test_quantum.py — Comprehensive test suite.

Tests verify:
- Mathematical correctness (unitarity, normalization, Hermiticity)
- Physical properties (Born's rule, purity bounds, entropy)
- Known quantum identities (H²=I, X²=I, gate compositions)
- Noise channel properties (trace preservation, purity reduction)
- Circuit correctness (Bell state creation, teleportation setup)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import Qubit, Gates, QuantumCircuit, TwoQubitSystem
from src.noise import AmplitudeDamping, PhaseDamping, Depolarizing, BitFlip, PhaseFlip


# ── Qubit Tests ───────────────────────────────────────────────────────────────

class TestQubit:
    def test_zero_state_normalization(self):
        q = Qubit.zero()
        assert abs(q.prob_zero - 1.0) < 1e-10
        assert abs(q.prob_one - 0.0) < 1e-10

    def test_plus_state_equal_superposition(self):
        q = Qubit.plus()
        assert abs(q.prob_zero - 0.5) < 1e-10
        assert abs(q.prob_one - 0.5) < 1e-10

    def test_pure_state_purity_is_one(self):
        q = Qubit.from_bloch(np.pi / 3, np.pi / 4)
        assert abs(q.purity - 1.0) < 1e-8

    def test_density_matrix_hermitian(self):
        q = Qubit(0.6 + 0.1j, 0.8 - 0.2j)
        rho = q.rho
        assert np.allclose(rho, rho.conj().T, atol=1e-10)

    def test_density_matrix_trace_one(self):
        q = Qubit(0.3, 0.7j)
        assert abs(np.trace(q.rho) - 1.0) < 1e-10

    def test_density_matrix_positive_semidefinite(self):
        q = Qubit.from_bloch(1.2, 2.5)
        eigenvalues = np.linalg.eigvalsh(q.rho)
        assert np.all(eigenvalues >= -1e-8)

    def test_bloch_vector_unit_length_pure_state(self):
        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            for phi in [0, np.pi / 3, np.pi]:
                q = Qubit.from_bloch(theta, phi)
                x, y, z = q.bloch_vector
                length = np.sqrt(x**2 + y**2 + z**2)
                assert abs(length - 1.0) < 1e-6, f"Bloch vector length {length} != 1 for θ={theta}, φ={phi}"

    def test_zero_bloch_north_pole(self):
        q = Qubit.zero()
        x, y, z = q.bloch_vector
        assert abs(z - 1.0) < 1e-8

    def test_one_bloch_south_pole(self):
        q = Qubit.one()
        x, y, z = q.bloch_vector
        assert abs(z + 1.0) < 1e-8

    def test_pure_state_entropy_is_zero(self):
        q = Qubit.plus()
        assert q.von_neumann_entropy() < 1e-8

    def test_fidelity_with_self_is_one(self):
        q = Qubit.from_bloch(0.7, 1.2)
        assert abs(q.fidelity(q) - 1.0) < 1e-8

    def test_fidelity_orthogonal_states_is_zero(self):
        q0 = Qubit.zero()
        q1 = Qubit.one()
        assert abs(q0.fidelity(q1) - 0.0) < 1e-8

    def test_measurement_collapses_to_basis(self):
        q = Qubit.plus()
        for _ in range(20):
            outcome, post = q.measure()
            assert outcome in (0, 1)
            if outcome == 0:
                assert abs(post.prob_zero - 1.0) < 1e-8
            else:
                assert abs(post.prob_one - 1.0) < 1e-8

    def test_born_rule_statistics(self):
        """Statistical test: |+⟩ measurements should be ~50/50."""
        q = Qubit.plus()
        outcomes = [q.measure()[0] for _ in range(1000)]
        p_zero = sum(1 for o in outcomes if o == 0) / 1000
        assert 0.4 < p_zero < 0.6, f"Born rule violation: P(0)={p_zero}"

    def test_normalization_preserved_after_apply(self):
        q = Qubit.zero()
        for gate in [Gates.H(), Gates.X(), Gates.Y(), Gates.Z(), Gates.S(), Gates.T()]:
            q2 = q.apply(gate)
            assert abs(np.trace(q2.rho) - 1.0) < 1e-10


# ── Gate Tests ────────────────────────────────────────────────────────────────

class TestGates:
    def test_all_standard_gates_are_unitary(self):
        gates = [
            Gates.I(), Gates.X(), Gates.Y(), Gates.Z(),
            Gates.H(), Gates.S(), Gates.T(),
            Gates.Rx(0.5), Gates.Ry(1.2), Gates.Rz(2.1),
            Gates.P(0.8), Gates.U(0.3, 0.7, 1.1),
        ]
        for gate in gates:
            assert Gates.is_unitary(gate), f"Gate is not unitary:\n{gate}"

    def test_hadamard_squared_is_identity(self):
        H = Gates.H()
        assert np.allclose(H @ H, np.eye(2), atol=1e-10)

    def test_pauli_squared_is_identity(self):
        for gate in [Gates.X(), Gates.Y(), Gates.Z()]:
            assert np.allclose(gate @ gate, np.eye(2), atol=1e-10)

    def test_s_squared_is_z(self):
        S = Gates.S()
        Z = Gates.Z()
        assert np.allclose(S @ S, Z, atol=1e-10)

    def test_t_squared_is_s(self):
        T = Gates.T()
        S = Gates.S()
        assert np.allclose(T @ T, S, atol=1e-10)

    def test_rx_zero_is_identity(self):
        assert np.allclose(Gates.Rx(0), np.eye(2), atol=1e-10)

    def test_rx_pi_is_ix(self):
        """Rx(π) = -iX (up to global phase, physically equivalent to X)."""
        Rx_pi = Gates.Rx(np.pi)
        X = Gates.X()
        # Should map |0⟩→|1⟩ and |1⟩→|0⟩
        assert abs(abs(Rx_pi[0, 1]) - 1.0) < 1e-8
        assert abs(abs(Rx_pi[1, 0]) - 1.0) < 1e-8

    def test_hadamard_maps_zero_to_plus(self):
        q = Qubit.zero().apply(Gates.H())
        assert abs(q.prob_zero - 0.5) < 1e-8
        assert abs(q.prob_one - 0.5) < 1e-8

    def test_x_gate_flips_state(self):
        q = Qubit.zero().apply(Gates.X())
        assert abs(q.prob_one - 1.0) < 1e-8


# ── Noise Channel Tests ───────────────────────────────────────────────────────

class TestNoise:
    def test_noise_preserves_trace(self):
        q = Qubit.plus()
        channels = [
            AmplitudeDamping(0.3),
            PhaseDamping(0.5),
            Depolarizing(0.2),
            BitFlip(0.15),
            PhaseFlip(0.1),
        ]
        for channel in channels:
            q_noisy = channel.apply(q)
            assert abs(np.trace(q_noisy.rho) - 1.0) < 1e-10

    def test_noise_reduces_purity(self):
        """Noise should reduce or maintain purity (never increase for pure → noisy)."""
        q = Qubit.plus()
        channels = [
            AmplitudeDamping(0.5),
            PhaseDamping(0.5),
            Depolarizing(0.5),
        ]
        for channel in channels:
            q_noisy = channel.apply(q)
            assert q_noisy.purity <= q.purity + 1e-8

    def test_amplitude_damping_full_decay(self):
        """γ=1: |1⟩ always decays to |0⟩."""
        q = Qubit.one()
        channel = AmplitudeDamping(gamma=1.0)
        q_decayed = channel.apply(q)
        assert abs(q_decayed.prob_zero - 1.0) < 1e-8

    def test_amplitude_damping_preserves_ground(self):
        """Amplitude damping leaves |0⟩ unchanged."""
        q = Qubit.zero()
        channel = AmplitudeDamping(gamma=0.7)
        q_after = channel.apply(q)
        assert abs(q_after.prob_zero - 1.0) < 1e-8

    def test_depolarizing_full_noise_is_maximally_mixed(self):
        """p=0.75: full depolarizing maps any state to I/2."""
        q = Qubit.zero()
        channel = Depolarizing(p=0.75)
        q_noisy = channel.apply(q)
        assert abs(q_noisy.purity - 0.5) < 1e-8

    def test_phase_damping_kills_coherence(self):
        """λ=1: phase damping destroys all off-diagonal elements."""
        q = Qubit.plus()
        channel = PhaseDamping(lam=1.0)
        q_dephased = channel.apply(q)
        rho = q_dephased.rho
        assert abs(rho[0, 1]) < 1e-8
        assert abs(rho[1, 0]) < 1e-8


# ── Two-Qubit & Circuit Tests ─────────────────────────────────────────────────

class TestTwoQubit:
    def test_initial_state_not_entangled(self):
        sys = TwoQubitSystem()
        assert not sys.is_entangled

    def test_bell_state_is_entangled(self):
        for i in range(4):
            bell = TwoQubitSystem.bell_state(i)
            assert bell.is_entangled

    def test_bell_state_max_entanglement(self):
        bell = TwoQubitSystem.bell_state(0)
        S = bell.entanglement_entropy
        assert abs(S - np.log(2)) < 1e-6

    def test_cnot_creates_entanglement_from_superposition(self):
        """H on qubit 0 + CNOT → Bell state."""
        sys = TwoQubitSystem()
        sys = sys.apply_single(Gates.H(), 0)
        sys = sys.apply_cnot(0, 1)
        assert sys.is_entangled

    def test_circuit_bell_state(self):
        circuit = QuantumCircuit(qubits=2)
        circuit.h(0).cnot(0, 1)
        result = circuit.run()
        final = result["final_state"]
        assert final.is_entangled

    def test_circuit_records_history(self):
        circuit = QuantumCircuit(qubits=1)
        circuit.h(0).x(0).z(0)
        result = circuit.run()
        assert len(result["history"]) == 4  # init + 3 gates

    def test_circuit_single_qubit_full_rotation(self):
        """Rx(2π) should return qubit to original state (up to global phase)."""
        circuit = QuantumCircuit(qubits=1)
        circuit.rx(2 * np.pi, 0)
        result = circuit.run()
        q = result["final_state"]
        assert abs(q.prob_zero - 1.0) < 1e-6

    def test_measurement_in_circuit(self):
        circuit = QuantumCircuit(qubits=1)
        circuit.h(0).measure(0)
        result = circuit.run()
        assert len(result["measurements"]) == 1
        assert result["measurements"][0]["outcome"] in (0, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
