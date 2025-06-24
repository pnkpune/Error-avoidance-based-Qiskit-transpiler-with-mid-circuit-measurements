# Complete Python Script for the Ancilla-Assisted Decoherence Watchdog
#
# This script contains the full implementation of the custom transpiler pass,
# the benchmark circuit, and the simulation protocol to reproduce the 
# findings of this report.
#
# To Run:
# 1. Ensure you have the necessary packages installed:
#    pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib
# 2. Execute the script from your terminal: python watchdog_benchmark.py

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import ASAPScheduleAnalysis
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import CnotGate, Measure, Delay
from qiskit.quantum_info import hellinger_fidelity
from qiskit.visualization import plot_histogram

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

# --- Section 1: Custom Transpiler Pass Definition ---

class DecoherenceWatchdog(TransformationPass):
    """
    A transpiler pass to mitigate decoherence in the most vulnerable idle
    qubit by inserting an ancilla-based heralding gadget.
    """

    def __init__(self, target):
        """
        DecoherenceWatchdog initializer.

        Args:
            target (qiskit.transpiler.Target): The target backend description.
        """
        super().__init__()
        self.target = target
        self.t2_times = {}
        self.dt = self.target.dt if self.target.dt is not None else 1.0 # Fallback

        if self.target.qubit_properties:
            for i, q_props in enumerate(self.target.qubit_properties):
                if q_props.t2 is not None:
                    self.t2_times[i] = q_props.t2

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the DecoherenceWatchdog pass on `dag`.
        """
        # 1. Schedule the circuit to make idle times explicit
        if not self.target.has_calibration("delay"):
            return dag # Cannot schedule without delay calibrations
        
        instruction_durations = self.target.durations()
        scheduler = ASAPScheduleAnalysis(instruction_durations)
        scheduled_dag = scheduler(dag)

        # 2. Find all delay instructions
        idle_periods = []
        qubit_indices = {qubit: i for i, qubit in enumerate(scheduled_dag.qubits)}

        for node in scheduled_dag.op_nodes():
            if isinstance(node.op, Delay):
                if not node.qargs:
                    continue
                qubit = node.qargs
                qubit_idx = qubit_indices[qubit]
                duration_dt = node.op.duration
                duration_sec = duration_dt * self.dt
                idle_periods.append({
                    "node": node, "qubit_idx": qubit_idx, "duration_sec": duration_sec
                })

        if not idle_periods:
            return dag

        # 3. Calculate vulnerability and find the target
        max_vulnerability = -1.0
        target_idle_info = None

        for period in idle_periods:
            qubit_idx = period["qubit_idx"]
            t2 = self.t2_times.get(qubit_idx)
            if t2 is None or t2 <= 0:
                continue
            vulnerability = 1.0 - math.exp(-period["duration_sec"] / t2)
            if vulnerability > max_vulnerability:
                max_vulnerability = vulnerability
                target_idle_info = period

        VULNERABILITY_THRESHOLD = 0.01
        if target_idle_info is None or max_vulnerability < VULNERABILITY_THRESHOLD:
            return dag

        # 4. Reconstruct the DAG with the watchdog gadget
        target_node = target_idle_info["node"]
        data_qubit = target_node.qargs

        ancilla_qreg = QuantumRegister(1, "watchdog_ancilla")
        herald_creg = ClassicalRegister(1, "watchdog_herald")

        new_dag = DAGCircuit()
        new_dag.add_qreg(ancilla_qreg)
        new_dag.add_creg(herald_creg)
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        ancilla_qubit = new_dag.qubits[ancilla_qreg.index]
        herald_bit = new_dag.clbits[herald_creg.index]

        # 5. Copy operations, inserting the gadget at the target location
        for node in dag.topological_op_nodes():
            if node == target_node:
                new_dag.apply_operation_back(CnotGate(), qargs=(data_qubit, ancilla_qubit))
                new_dag.apply_operation_back(CnotGate(), qargs=(data_qubit, ancilla_qubit))
                new_dag.apply_operation_back(Measure(), qargs=(ancilla_qubit,), cargs=(herald_bit,))
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        
        return new_dag

# --- Section 2: Benchmarking and Simulation Logic ---

def create_benchmark_circuit():
    """Creates a 3-qubit circuit designed to have a long idle time."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.barrier()
    # Insert a long delay on qubit 0 and 2 to make them idle
    # Using an explicit delay is more direct for benchmarking
    qc.delay(2000, 0, "dt") 
    qc.delay(2000, 2, "dt")
    qc.barrier()
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.barrier()
    qc.measure(, )
    return qc

def post_select_results(counts, num_data_clbits):
    """Filters a counts dictionary based on a herald bit."""
    new_counts = {}
    total_shots = sum(counts.values())
    kept_shots = 0
    
    for outcome, num_shots in counts.items():
        # Assumes herald bit is the first bit, outcome is little-endian
        if outcome[num_data_clbits] == '0':
            post_selected_outcome = outcome[:num_data_clbits]
            new_counts[post_selected_outcome] = new_counts.get(post_selected_outcome, 0) + num_shots
            kept_shots += num_shots
            
    discard_fraction = (total_shots - kept_shots) / total_shots if total_shots > 0 else 0
    return new_counts, discard_fraction

def run_benchmark():
    """Executes the full benchmark protocol."""
    print("--- Starting Ancilla-Assisted Decoherence Watchdog Benchmark ---")

    # 1. Setup Environment
    print("1. Setting up simulation environment...")
    fake_backend = FakeManilaV2()
    noise_model = NoiseModel.from_backend(fake_backend)
    target = fake_backend.target()
    
    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=noise_model)
    
    shots = 8192
    qc_bench = create_benchmark_circuit()

    # 2. Ideal Run (Ground Truth)
    print("2. Running Ideal (Noiseless) Simulation...")
    ideal_t_qc = transpile(qc_bench, ideal_sim)
    ideal_result = ideal_sim.run(ideal_t_qc, shots=shots).result()
    ideal_counts = ideal_result.get_counts()

    # 3. Standard Noisy Run
    print("3. Running Standard Noisy Simulation (opt_level=3)...")
    std_t_qc = transpile(qc_bench, backend=fake_backend, optimization_level=3)
    std_result = noisy_sim.run(std_t_qc, shots=shots).result()
    std_counts = std_result.get_counts()

    # 4. Watchdog Noisy Run
    print("4. Running Watchdog-Enhanced Noisy Simulation...")
    watchdog_pass = DecoherenceWatchdog(target)
    # Custom pass manager needs scheduling before the watchdog
    pm = PassManager()
    watchdog_qc = pm.run(qc_bench)
    
    # Transpile the rest of the circuit for the backend
    watchdog_t_qc = transpile(watchdog_qc, backend=fake_backend, optimization_level=1)
    
    watchdog_result = noisy_sim.run(watchdog_t_qc, shots=shots).result()
    watchdog_raw_counts = watchdog_result.get_counts()
    
    # Post-select the watchdog results
    num_data_clbits = qc_bench.num_clbits
    watchdog_ps_counts, discarded_frac = post_select_results(watchdog_raw_counts, num_data_clbits)

    # 5. Analyze Fidelity
    print("\n--- Benchmark Results ---")
    fidelity_std = hellinger_fidelity(ideal_counts, std_counts)
    fidelity_watchdog = hellinger_fidelity(ideal_counts, watchdog_ps_counts)

    print(f"Standard Transpilation Fidelity: {fidelity_std:.4f}")
    print(f"Watchdog-Enhanced Fidelity:      {fidelity_watchdog:.4f}")
    print(f"Shot Discard Fraction (Watchdog): {discarded_frac:.2%}")
    
    # 6. Plot Histograms
    print("\nGenerating histograms...")
    legend =
    hist_data = [ideal_counts, std_counts, watchdog_ps_counts]
    
    fig = plot_histogram(hist_data, legend=legend, figsize=(12, 7),
                         title="Fidelity Comparison: Standard vs. Decoherence Watchdog")
    fig.savefig("watchdog_benchmark_results.png")
    print("Results plot saved to 'watchdog_benchmark_results.png'")

if __name__ == "__main__":
    run_benchmark()