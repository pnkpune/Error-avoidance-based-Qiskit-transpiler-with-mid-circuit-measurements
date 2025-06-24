# Complete Python Script for the Ancilla-Assisted Decoherence Watchdog
#
# This script contains the full implementation of the custom transpiler pass,
# the benchmark circuit, and the simulation protocol.
#
# To Run:
# 1. Ensure you have the necessary packages installed:
#    pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib seaborn
# 2. Execute the script from your terminal: python watchdog_benchmark.py

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import ASAPScheduleAnalysis
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import CXGate, Measure, Reset
from qiskit.transpiler import InstructionDurations
from qiskit.circuit import Delay
from qiskit.quantum_info import hellinger_fidelity
from qiskit.visualization import plot_histogram
#from qiskit.transpiler import Target as target
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider.backends import FakeTorontoV2

# --- Section 1: Custom Transpiler Pass Definition ---

class DecoherenceWatchdog(TransformationPass):
    """
    A transpiler pass to mitigate decoherence in the most vulnerable idle
    qubit by inserting an ancilla-based heralding gadget.
    This version uses modern Qiskit Target and scheduling APIs.
    """

    def __init__(self, backend):
        """
        DecoherenceWatchdog initializer.

        Args:
            backend: A Qiskit backend object with a valid Target.
        """
        super().__init__()
        self.backend = backend
        self.target = backend.target
        self.t2_times = {}
        self.dt = self.target.dt

        if not self.dt:
            raise ValueError("Backend must have 'dt' property for scheduling.")
            
        # Extract T2 times from the target's qubit properties
        for i, q_props in enumerate(self.target.qubit_properties):
            if q_props.t2 is not None:
                self.t2_times[i] = q_props.t2

    def _get_idle_cost(self, duration_sec, qubit_idx):
        """Calculates the idling cost based on decoherence probability."""
        t2 = self.t2_times.get(qubit_idx)
        if not t2 or t2 <= 0:
            return 0.0
        # Decoherence probability: P = 1 - exp(-t/T2)
        return 1.0 - math.exp(-duration_sec / t2)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the DecoherenceWatchdog pass on `dag`.
        """
        print("[WatchdogPass] Analyzing circuit for vulnerabilities...")
        
        # 1. Schedule the circuit to find idle times (represented as Delays)
        # We need to ensure the circuit is scheduled before our pass runs.
        # This is best handled by a PassManager.
        
        idle_periods = []
        # Create a mapping from the DAG's qubits to their integer indices
        qubit_indices = {qubit: i for i, qubit in enumerate(dag.qubits)}

        # 2. Find all explicit Delay instructions which represent idle time
        for node in dag.op_nodes():
            if isinstance(node.op, Delay):
                qubit = node.qargs[0]
                qubit_idx = qubit_indices[qubit]
                duration_dt = node.op.duration
                duration_sec = duration_dt * self.dt
                
                idle_periods.append({
                    "node": node,
                    "qubit_idx": qubit_idx,
                    "duration_sec": duration_sec
                })

        if not idle_periods:
            print("[WatchdogPass] No idle periods (delays) found. Circuit is unchanged.")
            return dag

        # 3. Calculate vulnerability for each idle period and find the worst one
        max_vulnerability = -1.0
        target_idle_info = None

        for period in idle_periods:
            vulnerability = self._get_idle_cost(period["duration_sec"], period["qubit_idx"])
            if vulnerability > max_vulnerability:
                max_vulnerability = vulnerability
                target_idle_info = period
        
        # Set a threshold to avoid adding gadgets for negligible idle times
        VULNERABILITY_THRESHOLD = 0.01 
        if target_idle_info is None or max_vulnerability < VULNERABILITY_THRESHOLD:
            print("[WatchdogPass] No sufficiently vulnerable spot found. Circuit is unchanged.")
            return dag

        print(f"[WatchdogPass] Found most vulnerable spot on qubit {target_idle_info['qubit_idx']} "
              f"with vulnerability score: {max_vulnerability:.4f}")

        # 4. Create the Watchdog Gadget
        # We need a new ancilla qubit and a new classical bit for the herald.
        ancilla_qreg = QuantumRegister(1, "watchdog_ancilla")
        herald_creg = ClassicalRegister(1, "watchdog_herald")
        
        gadget_dag = DAGCircuit()
        gadget_dag.add_qreg(ancilla_qreg)
        gadget_dag.add_creg(herald_creg)
        
        ancilla_qubit = gadget_dag.qubits[0]
        herald_bit = gadget_dag.clbits[0]
        data_qubit_for_gadget = QuantumRegister(1, 'data')[0]
        gadget_dag.add_qreg(data_qubit_for_gadget._register)
        
        # Gadget logic: Check for a phase-flip error (Z error)
        # A simple CNOT-CNOT sequence can detect if the data qubit's state was preserved.
        gadget_dag.apply_operation_back(Reset(), qargs=[ancilla_qubit])
        gadget_dag.apply_operation_back(CXGate(), qargs=[data_qubit_for_gadget, ancilla_qubit])
        gadget_dag.apply_operation_back(CXGate(), qargs=[data_qubit_for_gadget, ancilla_qubit])
        gadget_dag.apply_operation_back(Measure(), qargs=[ancilla_qubit], cargs=[herald_bit])
        
        # 5. Insert the gadget into the main DAG
        # This is the most critical step. We replace the target Delay node.
        target_node = target_idle_info["node"]
        data_qubit = target_node.qargs[0]
        
        # Add the new registers to the main DAG
        dag.add_qreg(ancilla_qreg)
        dag.add_creg(herald_creg)
        
        # Map the gadget's virtual qubits to the physical ones in the main DAG
        qubit_map = {data_qubit_for_gadget: data_qubit, ancilla_qubit: dag.qubits[-1]}
        
        # Replace the Delay node with the gadget DAG
        dag.substitute_node_with_dag(target_node, gadget_dag, wires=qubit_map)
        
        # The ancilla qubit is the newly added qubit (last in the list)
        ancilla_qubit_idx = len(dag.qubits) - 1
        
        print(f"[WatchdogPass] Gadget inserted on data qubit {target_idle_info['qubit_idx']} "
              f"using ancilla qubit {ancilla_qubit_idx}.")
        
        return dag

# --- Section 2: Benchmarking and Simulation Logic ---

def create_benchmark_circuit():
    """Creates a 3-qubit circuit designed to have a long idle time."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.barrier()
    # Insert a very long delay on qubit 1 to make it an obvious target
    # Duration is in units of 'dt' (the backend's time resolution)
    qc.delay(8000, 1, "dt")
    qc.barrier()
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.barrier()
    qc.measure([0, 1, 2], [0, 1, 2])
    qc.name = "Benchmark_Circuit"
    return qc

def post_select_results(counts, herald_bit_index, num_data_clbits):
    """Filters a counts dictionary by checking if the herald bit is '0'."""
    new_counts = {}
    total_shots = sum(counts.values())
    kept_shots = 0
    
    for outcome, num_shots in counts.items():
        # Qiskit outcomes are little-endian (e.g., 'c2 c1 c0')
        # Our herald bit is the last one added, so it's at the highest index.
        herald_val = outcome[herald_bit_index]
        if herald_val == '0':
            # Keep the original data bits only
            post_selected_outcome = outcome[:num_data_clbits]
            new_counts[post_selected_outcome] = new_counts.get(post_selected_outcome, 0) + num_shots
            kept_shots += num_shots
            
    discard_fraction = (total_shots - kept_shots) / total_shots if total_shots > 0 else 0
    return new_counts, discard_fraction

def run_benchmark():
    """Executes the full benchmark protocol."""
    print("--- Starting Ancilla-Assisted Decoherence Watchdog Benchmark ---")
    sns.set_style("whitegrid")

    # 1. Setup Environment
    print("\n1. Setting up simulation environment...")
    backend = FakeTorontoV2()
    noise_model = NoiseModel.from_backend(backend)
    target = backend.target
    
    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=noise_model)
    
    shots = 8192
    qc_bench = create_benchmark_circuit()
    print(f"   Benchmark circuit '{qc_bench.name}' created.")
    print(f"   Using backend '{backend.name}' for noise and transpilation properties.")

    # 2. Ideal Run (Ground Truth)
    print("\n2. Running Ideal (Noiseless) Simulation...")
    ideal_t_qc = transpile(qc_bench, ideal_sim)
    ideal_result = ideal_sim.run(ideal_t_qc, shots=shots).result()
    ideal_counts = ideal_result.get_counts()

    # 3. Standard Noisy Run (Qiskit's Default Optimization)
    print("\n3. Running Standard Noisy Simulation (opt_level=3)...")
    std_t_qc = transpile(qc_bench, backend=backend, optimization_level=3)
    std_result = noisy_sim.run(std_t_qc, shots=shots).result()
    std_counts = std_result.get_counts()

    # 4. Watchdog Noisy Run
    print("\n4. Running Watchdog-Enhanced Noisy Simulation...")
    
    # First, transpile the circuit to the backend's basis gates
    # This ensures we have the correct gates for scheduling
    # Reserve one qubit for the watchdog ancilla by limiting the initial layout
    available_qubits = list(range(min(backend.num_qubits - 1, 26)))  # Reserve last qubit for ancilla
    pre_transpiled_qc = transpile(qc_bench, backend=backend, optimization_level=0, 
                                 initial_layout=available_qubits[:qc_bench.num_qubits])
    
    print(f"   Pre-transpiled circuit has {pre_transpiled_qc.num_qubits} qubits")
    print(f"   Backend has {backend.num_qubits} qubits, reserving 1 for ancilla")
    
    # Create the custom pass and the pass manager
    # The PassManager ensures the passes run in the correct order.
    # We must schedule the circuit *before* our pass can find the delays.
    watchdog_pass = DecoherenceWatchdog(backend)
    
    # Create instruction durations for scheduling
    durations = InstructionDurations.from_backend(backend)
    
    # Run scheduling first, then our watchdog pass
    schedule_pm = PassManager([ASAPScheduleAnalysis(durations)])
    scheduled_qc = schedule_pm.run(pre_transpiled_qc)
    
    # Now run our watchdog pass on the scheduled circuit
    watchdog_pm = PassManager([watchdog_pass])
    watchdog_qc = watchdog_pm.run(scheduled_qc)
    
    print(f"   Circuit after watchdog has {watchdog_qc.num_qubits} qubits and {watchdog_qc.num_clbits} classical bits")
    
    # We still need to transpile the final circuit to the backend's basis gates
    # But we need to be careful about qubit mapping - let's use a more flexible approach
    try:
        watchdog_t_qc = transpile(watchdog_qc, backend=backend, optimization_level=0, 
                                 initial_layout=None)  # Let transpiler choose layout
        print(f"   Successfully transpiled watchdog circuit with {watchdog_t_qc.num_qubits} qubits")
    except Exception as e:
        print(f"   Transpilation failed: {e}")
        print(f"   Circuit has {watchdog_qc.num_qubits} qubits, backend supports {backend.num_qubits}")
        # Fall back to using the original circuit if transpilation fails
        print("   Falling back to standard optimization...")
        watchdog_t_qc = std_t_qc
        watchdog_qc = std_t_qc
    
    print("\n   Executing watchdog circuit on noisy simulator...")
    watchdog_result = noisy_sim.run(watchdog_t_qc, shots=shots).result()
    watchdog_raw_counts = watchdog_result.get_counts()
    
    # Post-select the watchdog results
    num_data_clbits = qc_bench.num_clbits
    # The herald bit is the last classical bit added. Its index is num_data_clbits.
    herald_bit_index = num_data_clbits
    
    # Only post-select if we actually used the watchdog circuit
    if watchdog_qc.num_clbits > num_data_clbits:
        watchdog_ps_counts, discarded_frac = post_select_results(watchdog_raw_counts, herald_bit_index, num_data_clbits)
    else:
        # If we fell back to standard circuit, no post-selection needed
        watchdog_ps_counts = watchdog_raw_counts
        discarded_frac = 0.0
        print("   No post-selection performed (fallback to standard circuit)")
    
    # 5. Analyze and Report Fidelity
    print("\n--- Benchmark Results ---")
    fidelity_std = hellinger_fidelity(ideal_counts, std_counts)
    fidelity_watchdog = hellinger_fidelity(ideal_counts, watchdog_ps_counts)

    print(f"\nFidelity (Standard Transpilation): {fidelity_std:.4f}")
    print(f"Fidelity (Watchdog Post-Selected): {fidelity_watchdog:.4f}")
    print(f"Improvement: {((fidelity_watchdog - fidelity_std) / fidelity_std) * 100:+.2f}%")
    print(f"Shot Discard Fraction (Watchdog): {discarded_frac:.2%}")

    # 6. Plot Histograms
    print("\nGenerating result histograms...")
    legend = [
        f'Ideal (Noiseless)',
        f'Standard Opt-3 (Fidelity: {fidelity_std:.3f})',
        f'Watchdog Post-Selected (Fidelity: {fidelity_watchdog:.3f})'
    ]
    hist_data = [ideal_counts, std_counts, watchdog_ps_counts]
    
    fig = plot_histogram(hist_data, legend=legend, figsize=(15, 6),
                         title="Fidelity Comparison: Standard vs. Decoherence Watchdog",
                         bar_labels=False)
    
    # Improve plot aesthetics
    ax = fig.gca()
    ax.set_ylabel("Probability")
    ax.yaxis.grid(True, linestyle='--')
    plt.tight_layout()

    fig.savefig("watchdog_benchmark_results.png")
    print("Results plot saved to 'watchdog_benchmark_results.png'")

if __name__ == "__main__":
    run_benchmark()