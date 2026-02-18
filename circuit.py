"""
circuit.py — Quantum circuit framework.

Provides:
- TwoQubitSystem: 4D Hilbert space for entanglement simulation
- QuantumCircuit: Declarative circuit builder with gate chaining
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .qubit import Qubit
from .gates import Gates


# ── Two-Qubit System ──────────────────────────────────────────────────────────

class TwoQubitSystem:
    """
    Two-qubit system in a 4D Hilbert space: C² ⊗ C²
    
    Basis states: |00⟩, |01⟩, |10⟩, |11⟩
    State vector: |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
    
    Enables simulation of:
    - Quantum entanglement (Bell states)
    - CNOT and controlled operations
    - Quantum teleportation protocol
    - Superdense coding
    """

    def __init__(self, q0: Optional[Qubit] = None, q1: Optional[Qubit] = None):
        """
        Initialize from two individual qubits via tensor product.
        |ψ⟩ = |q0⟩ ⊗ |q1⟩
        """
        if q0 is None:
            q0 = Qubit.zero()
        if q1 is None:
            q1 = Qubit.zero()
        # State vector from density matrices
        self._rho = np.kron(q0.rho, q1.rho)

    @classmethod
    def from_statevector(cls, state: np.ndarray) -> "TwoQubitSystem":
        """Create system from a 4-component normalized state vector."""
        state = np.array(state, dtype=complex)
        norm = np.linalg.norm(state)
        state /= norm
        system = cls.__new__(cls)
        system._rho = np.outer(state, state.conj())
        return system

    @classmethod
    def bell_state(cls, index: int = 0) -> "TwoQubitSystem":
        """
        Create one of the four maximally entangled Bell states.
        |Φ+⟩ = (|00⟩ + |11⟩)/√2   [index=0]
        |Φ-⟩ = (|00⟩ - |11⟩)/√2   [index=1]
        |Ψ+⟩ = (|01⟩ + |10⟩)/√2   [index=2]
        |Ψ-⟩ = (|01⟩ - |10⟩)/√2   [index=3]
        """
        bell_states = [
            np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),   # Φ+
            np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),  # Φ-
            np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),   # Ψ+
            np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),  # Ψ-
        ]
        return cls.from_statevector(bell_states[index])

    @property
    def rho(self) -> np.ndarray:
        return self._rho.copy()

    @property
    def entanglement_entropy(self) -> float:
        """
        Von Neumann entropy of the reduced density matrix.
        S = 0 for separable states, S = log(2) for maximally entangled.
        Computed via partial trace over qubit 1.
        """
        rho_reduced = self._partial_trace_q1()
        eigenvalues = np.linalg.eigvalsh(rho_reduced)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))

    @property
    def is_entangled(self) -> bool:
        """A state is entangled if its entanglement entropy > 0."""
        return self.entanglement_entropy > 1e-6

    def apply_single(self, gate: np.ndarray, qubit_index: int) -> "TwoQubitSystem":
        """
        Apply single-qubit gate to qubit 0 or 1.
        Full operator: U ⊗ I (qubit 0) or I ⊗ U (qubit 1)
        """
        I = np.eye(2, dtype=complex)
        if qubit_index == 0:
            full_gate = np.kron(gate, I)
        else:
            full_gate = np.kron(I, gate)
        new_rho = full_gate @ self._rho @ full_gate.conj().T
        system = TwoQubitSystem.__new__(TwoQubitSystem)
        system._rho = new_rho
        return system

    def apply_cnot(self, control: int = 0, target: int = 1) -> "TwoQubitSystem":
        """
        Controlled-NOT gate.
        Flips target qubit if and only if control qubit is |1⟩.
        
        CNOT is the archetypal entangling gate:
        CNOT · (H ⊗ I)|00⟩ = |Φ+⟩  (Bell state creation)
        """
        if control == 0 and target == 1:
            CNOT = np.array(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex
            )
        else:  # control=1, target=0
            CNOT = np.array(
                [[1, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0]], dtype=complex
            )
        new_rho = CNOT @ self._rho @ CNOT.conj().T
        system = TwoQubitSystem.__new__(TwoQubitSystem)
        system._rho = new_rho
        return system

    def measure_qubit(self, qubit_index: int) -> Tuple[int, "TwoQubitSystem"]:
        """
        Measure one qubit. Returns (outcome, post_measurement_system).
        Non-local effect: if entangled, measuring one qubit instantly
        determines the reduced state of the other (EPR correlations).
        """
        probs = np.real(np.diag(self._rho))
        if qubit_index == 0:
            p0 = float(probs[0] + probs[1])
            p1 = float(probs[2] + probs[3])
        else:
            p0 = float(probs[0] + probs[2])
            p1 = float(probs[1] + probs[3])
        
        p0, p1 = max(p0, 0), max(p1, 0)
        total = p0 + p1
        outcome = int(np.random.choice([0, 1], p=[p0 / total, p1 / total]))
        
        # Post-measurement projection
        if qubit_index == 0:
            proj = np.diag([1, 1, 0, 0] if outcome == 0 else [0, 0, 1, 1]).astype(complex)
            prob = p0 if outcome == 0 else p1
        else:
            proj = np.diag([1, 0, 1, 0] if outcome == 0 else [0, 1, 0, 1]).astype(complex)
            prob = p0 if outcome == 0 else p1
        
        new_rho = proj @ self._rho @ proj / prob
        system = TwoQubitSystem.__new__(TwoQubitSystem)
        system._rho = new_rho
        return outcome, system

    def reduced_state(self, qubit_index: int) -> Qubit:
        """
        Obtain the reduced state of one qubit by tracing out the other.
        ρ_0 = tr_1(ρ),  ρ_1 = tr_0(ρ)
        Mixed if entangled — you cannot assign a pure state to one qubit
        of an entangled pair.
        """
        if qubit_index == 0:
            rho_reduced = self._partial_trace_q1()
        else:
            rho_reduced = self._partial_trace_q0()
        return Qubit.from_density_matrix(rho_reduced)

    def _partial_trace_q1(self) -> np.ndarray:
        """Partial trace over qubit 1: ρ_0 = tr_1(ρ)"""
        rho = self._rho.reshape(2, 2, 2, 2)
        return np.einsum('ijik->jk', rho)

    def _partial_trace_q0(self) -> np.ndarray:
        """Partial trace over qubit 0: ρ_1 = tr_0(ρ)"""
        rho = self._rho.reshape(2, 2, 2, 2)
        return np.einsum('ijkj->ik', rho)

    def statevector_probabilities(self) -> np.ndarray:
        """Return measurement probabilities for |00⟩, |01⟩, |10⟩, |11⟩."""
        return np.real(np.diag(self._rho))


# ── Quantum Circuit ───────────────────────────────────────────────────────────

class Operation:
    """A single operation in a circuit."""
    def __init__(self, name: str, gate: Optional[np.ndarray], qubit: int,
                 is_cnot: bool = False, control: int = 0, target: int = 1,
                 is_measure: bool = False):
        self.name = name
        self.gate = gate
        self.qubit = qubit
        self.is_cnot = is_cnot
        self.control = control
        self.target = target
        self.is_measure = is_measure


class QuantumCircuit:
    """
    Declarative quantum circuit builder.
    
    Supports 1 or 2 qubits. Operations are recorded and applied
    sequentially when run() is called.
    
    Example:
        circuit = QuantumCircuit(qubits=2)
        circuit.h(0).cnot(0, 1)  # Create Bell state
        result = circuit.run()
    """

    def __init__(self, qubits: int = 1,
                 initial_state: Optional[Union[Qubit, TwoQubitSystem]] = None):
        assert qubits in (1, 2), "Currently supports 1 or 2 qubits"
        self.num_qubits = qubits
        self._initial_state = initial_state
        self._operations: List[Operation] = []
        self.history: List[dict] = []

    # ── Gate methods (chainable) ──────────────────────────────────────────────

    def h(self, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation("H", Gates.H(), qubit))
        return self

    def x(self, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation("X", Gates.X(), qubit))
        return self

    def y(self, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation("Y", Gates.Y(), qubit))
        return self

    def z(self, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation("Z", Gates.Z(), qubit))
        return self

    def s(self, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation("S", Gates.S(), qubit))
        return self

    def t(self, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation("T", Gates.T(), qubit))
        return self

    def rx(self, theta: float, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation(f"Rx({theta:.3f})", Gates.Rx(theta), qubit))
        return self

    def ry(self, theta: float, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation(f"Ry({theta:.3f})", Gates.Ry(theta), qubit))
        return self

    def rz(self, theta: float, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation(f"Rz({theta:.3f})", Gates.Rz(theta), qubit))
        return self

    def cnot(self, control: int = 0, target: int = 1) -> "QuantumCircuit":
        assert self.num_qubits == 2, "CNOT requires a 2-qubit circuit"
        self._operations.append(
            Operation("CNOT", None, -1, is_cnot=True, control=control, target=target)
        )
        return self

    def measure(self, qubit: int = 0) -> "QuantumCircuit":
        self._operations.append(Operation("M", None, qubit, is_measure=True))
        return self

    def reset(self) -> "QuantumCircuit":
        """Clear all operations."""
        self._operations = []
        self.history = []
        return self

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute all operations in sequence.
        Returns a result dictionary with final state and measurement outcomes.
        """
        self.history = []
        measurements = []

        # Initialize state
        if self._initial_state is not None:
            state = self._initial_state
        elif self.num_qubits == 1:
            state = Qubit.zero()
        else:
            state = TwoQubitSystem()

        # Record initial state
        self._record_snapshot(state, "init", -1)

        for op in self._operations:
            if op.is_measure:
                if self.num_qubits == 1:
                    outcome, state = state.measure()
                else:
                    outcome, state = state.measure_qubit(op.qubit)
                measurements.append({"qubit": op.qubit, "outcome": outcome})
                self._record_snapshot(state, "M", op.qubit, outcome=outcome)
            elif op.is_cnot:
                state = state.apply_cnot(op.control, op.target)
                self._record_snapshot(state, "CNOT", -1)
            else:
                if self.num_qubits == 1:
                    state = state.apply(op.gate)
                else:
                    state = state.apply_single(op.gate, op.qubit)
                self._record_snapshot(state, op.name, op.qubit)

        return {
            "final_state": state,
            "measurements": measurements,
            "history": self.history,
            "num_operations": len(self._operations),
        }

    def _record_snapshot(self, state, op_name: str, qubit: int, outcome: int = None):
        """Record state snapshot after each operation."""
        entry = {"operation": op_name, "qubit": qubit}
        if self.num_qubits == 1:
            entry["bloch_vector"] = state.bloch_vector
            entry["purity"] = state.purity
            entry["prob_zero"] = state.prob_zero
            entry["prob_one"] = state.prob_one
        else:
            entry["entanglement_entropy"] = state.entanglement_entropy
            entry["is_entangled"] = state.is_entangled
            entry["probabilities"] = state.statevector_probabilities().tolist()
        if outcome is not None:
            entry["outcome"] = outcome
        self.history.append(entry)

    def __repr__(self) -> str:
        ops = " → ".join(
            f"CNOT({o.control},{o.target})" if o.is_cnot
            else f"M[q{o.qubit}]" if o.is_measure
            else f"{o.name}[q{o.qubit}]"
            for o in self._operations
        )
        return f"QuantumCircuit({self.num_qubits}q): {ops if ops else '(empty)'}"
