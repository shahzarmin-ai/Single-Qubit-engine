# ⚛ Quantum State Mapper

A high-fidelity Python simulation of single and two-qubit quantum dynamics, complete with an interactive 3D Bloch Sphere visualizer.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-required-orange)
![Tests](https://img.shields.io/badge/tests-42%20passing-brightgreen)

---

## Features

| Feature | Description |
|---|---|
| **Density Matrix Formalism** | Represents qubits as ρ (not just state vectors), enabling mixed state simulation |
| **Full Gate Library** | H, X, Y, Z, S, T, S†, T†, Rx, Ry, Rz, P, U — all verified unitary |
| **5 Noise Channels** | Amplitude Damping, Phase Damping, Depolarizing, Bit Flip, Phase Flip |
| **Two-Qubit System** | Tensor product Hilbert space with CNOT, entanglement entropy, partial trace |
| **Bell States** | All 4 maximally entangled Bell states with EPR correlation simulation |
| **Quantum Circuit API** | Chainable declarative circuit builder with history tracking |
| **42 Unit Tests** | Full test coverage verifying unitarity, Born's rule, noise, entanglement |
| **Interactive Visualizer** | Drag-to-rotate 3D Bloch Sphere with live state readout |

---

## Installation

```bash
git clone https://github.com/yourusername/quantum-state-mapper
cd quantum-state-mapper
pip install numpy pytest
```

---

## Quick Start

```python
from src import Qubit, Gates, QuantumCircuit, TwoQubitSystem
from src.noise import Depolarizing

# Create a qubit and apply gates
q = Qubit.zero()
q = q.apply(Gates.H())     # Hadamard → |+⟩ superposition
q = q.apply(Gates.T())     # T gate phase rotation

print(q)
# Qubit(purity=1.0000, P(0)=0.5000, P(1)=0.5000, bloch=(0.707, 0.707, 0.000))

# Measure (Born's rule collapse)
outcome, post_state = q.measure()
print(f"Measured: |{outcome}⟩")

# Apply noise
noisy = Depolarizing(p=0.1).apply(q)
print(f"Purity after noise: {noisy.purity:.4f}")  # < 1.0 → mixed state
```

### Bell State Entanglement

```python
# Create an entangled Bell pair: |Φ+⟩ = (|00⟩ + |11⟩)/√2
circuit = QuantumCircuit(qubits=2)
circuit.h(0).cnot(0, 1)

result = circuit.run()
system = result["final_state"]

print(f"Entangled: {system.is_entangled}")              # True
print(f"Entropy: {system.entanglement_entropy:.4f}")    # 0.6931 = log(2)

# Measure qubit 0 — instantly determines qubit 1 (EPR correlations)
outcome, post = system.measure_qubit(0)
q1_state = post.reduced_state(1)
print(f"Qubit 0: |{outcome}⟩  →  Qubit 1 P(0): {q1_state.prob_zero:.0f}")
```

### Noise and Decoherence

```python
from src.noise import AmplitudeDamping, PhaseDamping

# T1 relaxation: excited state decays to ground state
q = Qubit.one()
decayed = AmplitudeDamping(gamma=0.5).apply(q)
print(f"After decay: P(0)={decayed.prob_zero:.4f}")  # > 0.5

# T2 dephasing: destroys quantum coherence off-diagonal elements
q = Qubit.plus()
dephased = PhaseDamping(lam=0.9).apply(q)
print(f"Coherence: {abs(dephased.rho[0,1]):.4f}")   # → 0
print(f"Entropy: {dephased.von_neumann_entropy():.4f}")  # → log(2)
```

---

## Interactive Visualizer

Open `visualizer.html` in any browser — no server required.

- **Drag** the Bloch sphere to rotate the 3D view
- **Apply gates** and watch the state vector rotate in real time
- **Add noise** to watch the state shrink inside the sphere (purity < 1)
- **Measure** to trigger wavefunction collapse with visual flash

---

## Running Tests

```bash
pytest tests/ -v
```

All 42 tests verify mathematical correctness:
- Gate unitarity (U†U = I for all gates)
- Known identities (H²=I, X²=I, S²=Z, T²=S)
- Density matrix axioms (Hermitian, trace=1, positive semidefinite)
- Born's rule statistical accuracy (1000-sample test)
- Noise trace preservation and purity reduction
- Bell state maximum entanglement entropy = log(2)

---

## Physics Reference

### Why Density Matrices?

State vectors |ψ⟩ can only represent *pure* states. Real qubits interact with their environment and become *mixed states* — statistical mixtures of pure states. The density matrix ρ handles both:

- **Pure state**: ρ = |ψ⟩⟨ψ|,  tr(ρ²) = 1
- **Mixed state**: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|,  tr(ρ²) < 1

### Noise Channels (Kraus Formalism)

Each noise channel E is a completely positive trace-preserving (CPTP) map:

```
ρ → E(ρ) = Σₖ Kₖ ρ Kₖ†     where  Σₖ Kₖ†Kₖ = I
```

| Channel | Physical Process | Effect on Bloch sphere |
|---|---|---|
| Amplitude Damping | T1 energy relaxation | Shrinks toward |0⟩ |
| Phase Damping | T2 dephasing | Shrinks toward Z axis |
| Depolarizing | Random Pauli errors | Shrinks toward origin |
| Bit Flip | X error with prob p | Shrinks along X/Y plane |
| Phase Flip | Z error with prob p | Shrinks along Z axis |

### Entanglement Entropy

For a bipartite system, entanglement is measured by the Von Neumann entropy of the reduced density matrix:

```
S(ρ_A) = -tr(ρ_A log ρ_A)
```

- Separable states: S = 0
- Maximally entangled Bell states: S = log(2) ≈ 0.693 nats

---

## Real-World Applications

**Quantum Algorithm Prototyping** — Classical simulation is the standard first step before deploying on QPU hardware (IBM Quantum, IonQ, Rigetti). Shor's and Grover's algorithms are developed this way.

**Quantum Error Correction** — The noise models implemented here are the fundamental building blocks of surface codes and stabilizer codes used in fault-tolerant quantum computing.

**NMR / MRI Technology** — The Bloch sphere precession model directly describes spin dynamics in Nuclear Magnetic Resonance, the physics behind MRI scanners.

**Quantum Cryptography (QKD)** — The measurement collapse logic demonstrates why eavesdropping on BB84 protocol quantum keys is physically detectable.

**Quantum Sensing** — Rotation gate precision is the basis for quantum magnetometers and atomic clocks that surpass classical measurement limits.

---

## Project Structure

```
quantum-state-mapper/
├── src/
│   ├── __init__.py      # Public API
│   ├── qubit.py         # Density matrix qubit representation
│   ├── gates.py         # Full single-qubit gate library
│   ├── noise.py         # 5 Kraus-operator noise channels
│   └── circuit.py       # TwoQubitSystem + QuantumCircuit
├── tests/
│   └── test_quantum.py  # 42 unit tests
├── visualizer.html      # Interactive Bloch sphere (no dependencies)
└── README.md
```

---

## License

MIT