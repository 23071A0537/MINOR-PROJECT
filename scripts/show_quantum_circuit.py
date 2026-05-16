"""Render the VQC circuit used in this project.

This mirrors the circuit definition used in:
- VQC/vqc_v7_phase1_training.py
- VQC/vqc_v7_precompute_features.py

Circuit = ZZFeatureMap + RealAmplitudes (linear entanglement).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap


def build_circuit(n_qubits: int = 8, ra_reps: int = 2, zz_reps: int = 2) -> tuple[QuantumCircuit, int]:
    """Build the project's VQC circuit and return (circuit, num_trainable_params)."""
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=zz_reps, entanglement="linear")
    va = RealAmplitudes(n_qubits, reps=ra_reps, entanglement="linear")

    qc = QuantumCircuit(n_qubits)
    qc.compose(fm, inplace=True)
    qc.compose(va, inplace=True)
    return qc, va.num_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the VQC quantum circuit used in this project.")
    parser.add_argument("--qubits", type=int, default=8, help="Number of qubits (default: 8)")
    parser.add_argument("--ra-reps", type=int, default=2, help="RealAmplitudes repetitions (default: 2)")
    parser.add_argument("--zz-reps", type=int, default=2, help="ZZFeatureMap repetitions (default: 2)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts") / "plots" / "vqc_circuit.png",
        help="Path for PNG output (default: artifacts/plots/vqc_circuit.png)",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Decompose circuit to show individual gates (RY, CNOT, ZZ, etc.)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc, n_params = build_circuit(
        n_qubits=args.qubits,
        ra_reps=args.ra_reps,
        zz_reps=args.zz_reps,
    )

    print("VQC circuit summary")
    print(f"- qubits: {args.qubits}")
    print(f"- ZZFeatureMap reps: {args.zz_reps}")
    print(f"- RealAmplitudes reps: {args.ra_reps}")
    print(f"- trainable parameters: {n_params}")
    print("\nASCII circuit diagram:\n")
    
    # Decompose if requested to show actual gates
    if args.decompose:
        qc_display = qc.decompose()
        print("[Decomposed - showing individual gates (RY, CNOT, ZZ, etc.)]\n")
    else:
        qc_display = qc
        print("[Blueprint blocks - use --decompose for individual gates]\n")
    
    print(qc_display.draw(output="text", fold=120))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig = qc_display.draw(output="mpl", fold=120)
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"\nSaved image to: {args.out}")
    except Exception as exc:
        print("\nCould not render PNG via matplotlib drawer.")
        print("Install/verify matplotlib support if you need image output.")
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()
