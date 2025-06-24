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
from qiskit.circuit.library import CXGate, Measure, Reset, HGate
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

# In the DecoherenceWatchdog class...

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the DecoherenceWatchdog pass on `dag`.
        This corrected version builds the gadget AROUND the idle period.
        """
        print("[WatchdogPass] Analyzing circuit for vulnerabilities...")
        
        # 1. Schedule the circuit to find idle times (represented as Delays)
        idle_periods = []
        qubit_indices = {qubit: i for i, qubit in enumerate(dag.qubits)}

        for node in dag.op_nodes():
            if isinstance(node.op, Delay):
                qubit = node.qargs[0]
                qubit_idx = qubit_indices[qubit]
                duration_dt = node.op.duration
                duration_sec = duration_dt * self.dt
                
                idle_periods.append({
                    "node": node,
                    "qubit_idx": qubit_idx,
                    "duration_dt": duration_dt, # Store duration in dt
                    "duration_sec": duration_sec
                })

        if not idle_periods:
            print("[WatchdogPass] No idle periods (delays) found. Circuit is unchanged.")
            return dag

        # 2. Find the most vulnerable idle period
        max_vulnerability = -1.0
        target_idle_info = None
        for period in idle_periods:
            vulnerability = self._get_idle_cost(period["duration_sec"], period["qubit_idx"])
            if vulnerability > max_vulnerability:
                max_vulnerability = vulnerability
                target_idle_info = period
        
        VULNERABILITY_THRESHOLD = 0.01 
        if target_idle_info is None or max_vulnerability < VULNERABILITY_THRESHOLD:
            print("[WatchdogPass] No sufficiently vulnerable spot found. Circuit is unchanged.")
            return dag

        print(f"[WatchdogPass] Found most vulnerable spot on qubit {target_idle_info['qubit_idx']} "
              f"with vulnerability score: {max_vulnerability:.4f}")

        # 3. Create the complete Watchdog Gadget Sequence (Entangle -> Idle -> Disentangle)
        # We need a new ancilla qubit and a new classical bit for the herald.
        ancilla_qreg = QuantumRegister(1, "watchdog_ancilla")
        herald_creg = ClassicalRegister(1, "watchdog_herald")
        
        # Create a new DAG for the entire sequence that will replace the single delay node.
        watchdog_sequence_dag = DAGCircuit()
        watchdog_sequence_dag.add_qreg(ancilla_qreg)
        watchdog_sequence_dag.add_creg(herald_creg)
        
        # The gadget needs a reference to the data qubit and the new ancilla
        data_qubit_ref = QuantumRegister(1, 'data_ref')[0]
        watchdog_sequence_dag.add_qreg(data_qubit_ref._register)
        ancilla_qubit_ref = watchdog_sequence_dag.qubits[0]
        herald_bit_ref = watchdog_sequence_dag.clbits[0]

        # --- Gadget Part 1: Entangle ---
        # This part goes BEFORE the idle period.
        watchdog_sequence_dag.apply_operation_back(Reset(), qargs=[ancilla_qubit_ref])
        watchdog_sequence_dag.apply_operation_back(HGate(), qargs=[ancilla_qubit_ref])
        # CORRECTED CNOT: data is control, ancilla is target.
        watchdog_sequence_dag.apply_operation_back(CXGate(), qargs=[data_qubit_ref, ancilla_qubit_ref])

        # --- Gadget Part 2: The Original Idle Period ---
        # We re-insert the delay that we are replacing. This is the "watch" period.
        idle_duration_dt = target_idle_info["duration_dt"]
        watchdog_sequence_dag.apply_operation_back(Delay(idle_duration_dt, "dt"), qargs=[data_qubit_ref])
        
        # --- Gadget Part 3: Disentangle and Measure ---
        # This part goes AFTER the idle period.
        watchdog_sequence_dag.apply_operation_back(CXGate(), qargs=[data_qubit_ref, ancilla_qubit_ref])
        watchdog_sequence_dag.apply_operation_back(HGate(), qargs=[ancilla_qubit_ref])
        watchdog_sequence_dag.apply_operation_back(Measure(), qargs=[ancilla_qubit_ref], cargs=[herald_bit_ref])
        
        # 4. Insert the full gadget sequence into the main DAG
        target_node = target_idle_info["node"]
        data_qubit_main = target_node.qargs[0]
        
        # Add the new registers to the main DAG
        dag.add_qreg(ancilla_qreg)
        dag.add_creg(herald_creg)
        
        # Map the gadget's virtual qubits/bits to the physical ones in the main DAG
        # The new ancilla qubit is the last one added to the dag.
        ancilla_qubit_main = dag.qubits[-1]
        qubit_map = {data_qubit_ref: data_qubit_main, ancilla_qubit_ref: ancilla_qubit_main}
        
        # Replace the single Delay node with our entire watchdog sequence
        dag.substitute_node_with_dag(target_node, watchdog_sequence_dag, wires=qubit_map)
        
        ancilla_qubit_idx = qubit_indices.get(ancilla_qubit_main, len(dag.qubits) - 1)
        
        print(f"[WatchdogPass] Gadget sequence inserted on data qubit {target_idle_info['qubit_idx']} "
              f"using ancilla qubit {ancilla_qubit_idx}.")
        
        return dag
# --- Section 2: Circuit Utilities ---

def deflate_circuit(circuit):
    """
    Simple circuit deflation - removes unused qubits from a circuit.
    Based on the concept from mapomatic but implemented for compatibility.
    """
    # Find which qubits are actually used (have operations on them)
    used_qubits = set()
    
    # Check all instructions to see which qubits are used
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            used_qubits.add(circuit.qubits.index(qubit))
    
    # If all qubits are used, return the original circuit
    if len(used_qubits) == circuit.num_qubits:
        return circuit
    
    # Create a new circuit with only the used qubits
    used_qubit_list = sorted(list(used_qubits))
    new_qc = QuantumCircuit(len(used_qubit_list), circuit.num_clbits)
    
    # Create a mapping from old qubit indices to new ones
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_qubit_list)}
    
    # Copy all instructions with remapped qubits
    for instruction in circuit.data:
        # Map qubits
        new_qubits = []
        for qubit in instruction.qubits:
            old_idx = circuit.qubits.index(qubit)
            new_idx = qubit_map[old_idx]
            new_qubits.append(new_qc.qubits[new_idx])
        
        # Map classical bits (keep same mapping)
        new_clbits = []
        for clbit in instruction.clbits:
            clbit_idx = circuit.clbits.index(clbit)
            new_clbits.append(new_qc.clbits[clbit_idx])
        
        # Add the instruction to the new circuit
        new_qc.append(instruction.operation, new_qubits, new_clbits)
    
    return new_qc

# --- Section 3: Benchmarking and Simulation Logic ---

def create_benchmark_circuit():
    """Creates a 4-qubit GHZ circuit designed to have a long idle time on one qubit."""
    qc = QuantumCircuit(4, 4)
    
    # Create GHZ state: |0000⟩ + |1111⟩ (unnormalized)
    qc.h(0)  # Put first qubit in superposition
    qc.cx(0, 1)  # Entangle with second qubit
    qc.cx(1, 2)  # Entangle with third qubit
    qc.cx(2, 3)  # Entangle with fourth qubit
    
    qc.barrier()
    # Insert a long delay on qubit 2 to make it vulnerable to decoherence
    # This should break the GHZ entanglement
    qc.delay(8000, 2, "dt")
    qc.barrier()
    
    # Measure all qubits
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    qc.name = "GHZ_Benchmark_Circuit"
    return qc

def post_select_results(counts, herald_bit_index, num_data_clbits):
    """Filters a counts dictionary by checking if the herald bit is '0'."""
    new_counts = {}
    total_shots = sum(counts.values())
    kept_shots = 0
    
    for outcome, num_shots in counts.items():
        # Qiskit outcomes are little-endian (e.g., 'c4 c3 c2 c1 c0')
        # Herald bit should be at index 0 (leftmost bit)
        herald_val = outcome[herald_bit_index]
        if herald_val == '0':
            # Extract the data bits (remove the herald bit)
            # If herald is at index 0, data bits are outcome[1:]
            if herald_bit_index == 0:
                post_selected_outcome = outcome[1:]
            else:
                # If herald is at a different position, need to reconstruct
                post_selected_outcome = outcome[:herald_bit_index] + outcome[herald_bit_index+1:]
            
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
    num_qubits_to_use = min(qc_bench.num_qubits + 3, backend.num_qubits - 1)  # Leave room for ancilla
    available_qubits = list(range(num_qubits_to_use))
    pre_transpiled_qc = transpile(qc_bench, backend=backend, optimization_level=0, 
                                 initial_layout=available_qubits[:qc_bench.num_qubits])
    
    # Deflate the circuit to remove unused qubits
    pre_transpiled_qc = deflate_circuit(pre_transpiled_qc)
    
    print(f"   Pre-transpiled circuit has {pre_transpiled_qc.num_qubits} qubits")
    print(f"   Backend has {backend.num_qubits} qubits, can accommodate up to {backend.num_qubits - 1} + 1 ancilla")
    
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
        # First transpile with optimization
        watchdog_t_qc = transpile(watchdog_qc, backend=backend, optimization_level=0, 
                                 initial_layout=None)  # Let transpiler choose layout
        
        # Use mapomatic to deflate the circuit and reduce its footprint
        watchdog_t_qc = deflate_circuit(watchdog_t_qc)
        
        print(f"   Successfully transpiled and deflated watchdog circuit with {watchdog_t_qc.num_qubits} qubits")
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
    
    # Debug: Print some sample outcomes to understand the bit ordering
    print(f"   Sample raw watchdog outcomes (first 5):")
    sample_outcomes = list(watchdog_raw_counts.keys())[:5]
    for outcome in sample_outcomes:
        print(f"     {outcome} -> {watchdog_raw_counts[outcome]} counts")
    
    # Post-select the watchdog results
    num_data_clbits = qc_bench.num_clbits
    # The herald bit is the last classical bit added. In Qiskit's little-endian format,
    # it should be at the leftmost position (highest index) in the bit string
    herald_bit_index = 0  # Leftmost bit in little-endian format
    
    # Only post-select if we actually used the watchdog circuit
    if watchdog_qc.num_clbits > num_data_clbits:
        watchdog_ps_counts, discarded_frac = post_select_results(watchdog_raw_counts, herald_bit_index, num_data_clbits)
    else:
        # If we fell back to standard circuit, no post-selection needed
        watchdog_ps_counts = watchdog_raw_counts
        discarded_frac = 0.0
        print("   No post-selection performed (fallback to standard circuit)")
    
    # 5. Analyze and Report GHZ Fidelity
    print("\n--- GHZ Benchmark Results ---")
    
    # Calculate GHZ state probabilities for each case
    def calculate_ghz_fidelity(counts, label, is_post_selected=False):
        total_shots = sum(counts.values())
        
        if is_post_selected:
            # For post-selected data: convert 4-bit outcomes back to 5-bit with herald='0'
            # and calculate probabilities only among herald='0' states
            ghz_0000 = counts.get('0000', 0) / total_shots
            ghz_1111 = counts.get('1111', 0) / total_shots
        elif len(list(counts.keys())[0]) == 4:
            # For 4-qubit ideal and standard cases, look for 0000 and 1111
            ghz_0000 = counts.get('0000', 0) / total_shots
            ghz_1111 = counts.get('1111', 0) / total_shots
        else:
            # For 5-qubit raw watchdog data, look for x0000 and x1111 patterns
            ghz_0000 = 0
            ghz_1111 = 0
            for outcome, count in counts.items():
                data_bits = outcome[1:]  # Remove herald bit
                if data_bits == '0000':
                    ghz_0000 += count / total_shots
                elif data_bits == '1111':
                    ghz_1111 += count / total_shots
        
        ghz_total = ghz_0000 + ghz_1111
        
        print(f"\n{label}:")
        print(f"  P(|0000⟩): {ghz_0000:.4f}")
        print(f"  P(|1111⟩): {ghz_1111:.4f}")
        print(f"  GHZ Fidelity: {ghz_total:.4f}")
        print(f"  Other states: {1-ghz_total:.4f}")
        
        return ghz_total, ghz_0000, ghz_1111
    
    ideal_ghz_fid, ideal_0000, ideal_1111 = calculate_ghz_fidelity(ideal_counts, "1) Noiseless Ideal")
    std_ghz_fid, std_0000, std_1111 = calculate_ghz_fidelity(std_counts, "2) Noisy with Optimization=3")
    raw_ghz_fid, raw_0000, raw_1111 = calculate_ghz_fidelity(watchdog_raw_counts, "3) Noisy with Custom Transpiler")
    ps_ghz_fid, ps_0000, ps_1111 = calculate_ghz_fidelity(watchdog_ps_counts, "4) Noisy with Discarded Runs", is_post_selected=True)
    
    print(f"\n--- GHZ Fidelity Summary ---")
    print(f"Ideal GHZ Fidelity:           {ideal_ghz_fid:.4f}")
    print(f"Standard GHZ Fidelity:        {std_ghz_fid:.4f}")
    print(f"Watchdog Raw GHZ Fidelity:    {raw_ghz_fid:.4f}")
    print(f"Watchdog PS GHZ Fidelity:     {ps_ghz_fid:.4f}")
    print(f"Improvement (PS vs Standard): {((ps_ghz_fid - std_ghz_fid) / std_ghz_fid) * 100:+.2f}%")
    print(f"Shot Discard Fraction:        {discarded_frac:.2%}")

    # 6. Plot Histograms for all four cases
    print("\nGenerating comprehensive comparison histogram...")
    
    # Helper functions
    def expand_to_5bit_with_herald_0(counts_4bit):
        """Convert 4-bit outcomes to 5-bit with herald='0' for comparison"""
        expanded = {}
        for outcome, count in counts_4bit.items():
            expanded['0' + outcome] = count
        return expanded
    
    def normalize_counts(counts):
        total = sum(counts.values())
        return {outcome: count/total for outcome, count in counts.items()}
    
    # For post-selected data, we need to handle it specially
    # It has 4-bit outcomes but we want to show it in 5-bit format with herald='0'
    def get_5bit_probs_for_post_selected(ps_counts_4bit, raw_counts_5bit):
        """Convert post-selected 4-bit data to 5-bit probabilities, zeroing herald='1' states"""
        # Start with all zeros for all possible 5-bit states
        all_5bit_states = {}
        for outcome in raw_counts_5bit.keys():
            all_5bit_states[outcome] = 0
        
        # Fill in the herald='0' states from post-selected data
        total_ps_shots = sum(ps_counts_4bit.values())
        for outcome_4bit, count in ps_counts_4bit.items():
            outcome_5bit = '0' + outcome_4bit  # Add herald='0'
            if outcome_5bit in all_5bit_states:
                all_5bit_states[outcome_5bit] = count / total_ps_shots
        
        return all_5bit_states
    
    # Create 5-qubit probability distributions for all cases
    ideal_5bit_probs = normalize_counts(expand_to_5bit_with_herald_0(ideal_counts))
    std_5bit_probs = normalize_counts(expand_to_5bit_with_herald_0(std_counts))
    raw_5bit_probs = normalize_counts(watchdog_raw_counts)
    ps_5bit_probs = get_5bit_probs_for_post_selected(watchdog_ps_counts, watchdog_raw_counts)
    
    # Print shot statistics
    print(f"   1) Ideal shots: {sum(ideal_counts.values())}")
    print(f"   2) Standard shots: {sum(std_counts.values())}")
    print(f"   3) Watchdog raw shots: {sum(watchdog_raw_counts.values())}")
    print(f"   4) Watchdog post-selected shots: {sum(watchdog_ps_counts.values())}")
    
    # Create histogram with all four 5-qubit distributions
    legend = [
        f'1) Ideal (GHZ: {ideal_ghz_fid:.3f})',
        f'2) Standard (GHZ: {std_ghz_fid:.3f})',
        f'3) Watchdog Raw (GHZ: {raw_ghz_fid:.3f})',
        f'4) Watchdog PS (GHZ: {ps_ghz_fid:.3f})'
    ]
    hist_data = [ideal_5bit_probs, std_5bit_probs, raw_5bit_probs, ps_5bit_probs]
    
    fig = plot_histogram(hist_data, legend=legend, figsize=(20, 10),
                         title="GHZ State Analysis: 5-Qubit Probability Distributions",
                         bar_labels=False)
    
    # Improve plot aesthetics
    ax = fig.gca()
    ax.set_ylabel("Probability")
    ax.yaxis.grid(True, linestyle='--')
    plt.tight_layout()

    fig.savefig("watchdog_benchmark_results.png")
    print("Results plot saved to 'watchdog_benchmark_results.png'")
    
    # 7. Create comprehensive 5-qubit state probability table
    print("\n--- Complete 5-Qubit State Probability Distribution ---")
    
    # Convert 4-qubit states to 5-qubit format for comparison
    def expand_to_5bit_with_herald_0(counts_4bit):
        """Convert 4-bit outcomes to 5-bit with herald='0' for comparison"""
        expanded = {}
        for outcome, count in counts_4bit.items():
            expanded['0' + outcome] = count
        return expanded
    
    # Normalize all distributions to probabilities
    def normalize_counts(counts):
        total = sum(counts.values())
        return {outcome: count/total for outcome, count in counts.items()}
    
    ideal_5bit = expand_to_5bit_with_herald_0(ideal_counts)
    std_5bit = expand_to_5bit_with_herald_0(std_counts)
    raw_5bit = watchdog_raw_counts
    
    # For post-selected, create 5-bit representation with herald='0' states only
    ps_5bit = {}
    for outcome in raw_5bit.keys():
        ps_5bit[outcome] = 0  # Initialize all to zero
    # Fill in herald='0' states from post-selected data
    for outcome_4bit, count in watchdog_ps_counts.items():
        outcome_5bit = '0' + outcome_4bit
        if outcome_5bit in ps_5bit:
            ps_5bit[outcome_5bit] = count
    
    ideal_probs = normalize_counts(ideal_5bit)
    std_probs = normalize_counts(std_5bit)
    raw_probs = normalize_counts(raw_5bit)
    # For PS, normalize only among the herald='0' states that have non-zero counts
    ps_total_shots = sum(count for count in ps_5bit.values() if count > 0)
    ps_probs = {outcome: count/ps_total_shots if ps_total_shots > 0 else 0 for outcome, count in ps_5bit.items()}
    
    # Get all possible 5-bit outcomes
    all_5bit_outcomes = set()
    all_5bit_outcomes.update(ideal_probs.keys())
    all_5bit_outcomes.update(std_probs.keys())
    all_5bit_outcomes.update(raw_probs.keys())
    all_5bit_outcomes.update(ps_probs.keys())
    all_5bit_outcomes = sorted(list(all_5bit_outcomes))
    
    # Create comprehensive table
    print(f"{'5-Bit':<8} {'Data':<6} {'1)Ideal':<10} {'2)Std':<10} {'3)Raw':<10} {'4)PS':<10}")
    print(f"{'State':<8} {'Bits':<6} {'P(state)':<10} {'P(state)':<10} {'P(state)':<10} {'P(state)':<10}")
    print("-" * 64)
    
    # Separate herald=0 and herald=1 states for clarity
    herald_0_states = [s for s in all_5bit_outcomes if s[0] == '0']
    herald_1_states = [s for s in all_5bit_outcomes if s[0] == '1']
    
    print("Herald = '0' states:")
    for outcome in herald_0_states:
        data_bits = outcome[1:]
        ideal_p = ideal_probs.get(outcome, 0.0)
        std_p = std_probs.get(outcome, 0.0)
        raw_p = raw_probs.get(outcome, 0.0)
        ps_p = ps_probs.get(outcome, 0.0)
        
        # Highlight GHZ states
        ghz_marker = " *" if data_bits in ['0000', '1111'] else ""
        print(f"{outcome:<8} {data_bits:<6} {ideal_p:<10.4f} {std_p:<10.4f} {raw_p:<10.4f} {ps_p:<10.4f}{ghz_marker}")
    
    if herald_1_states:
        print("\nHerald = '1' states:")
        for outcome in herald_1_states:
            data_bits = outcome[1:]
            ideal_p = ideal_probs.get(outcome, 0.0)
            std_p = std_probs.get(outcome, 0.0)
            raw_p = raw_probs.get(outcome, 0.0)
            ps_p = ps_probs.get(outcome, 0.0)
            print(f"{outcome:<8} {data_bits:<6} {ideal_p:<10.4f} {std_p:<10.4f} {raw_p:<10.4f} {ps_p:<10.4f}")
    
    print("-" * 64)
    print(f"{'TOTAL':<8} {'':6} {1.0:<10.4f} {1.0:<10.4f} {1.0:<10.4f} {1.0:<10.4f}")
    
    print(f"\nLegend:")
    print(f"* = GHZ states (|0000⟩ or |1111⟩)")
    print(f"1) Ideal: Noiseless simulation")
    print(f"2) Std: Noisy with Qiskit optimization_level=3")
    print(f"3) Raw: Noisy with watchdog transpiler (no post-selection)")
    print(f"4) PS: Noisy with watchdog transpiler (post-selected on herald='0')")
    print(f"Herald bit is leftmost, data bits are rightmost 4 bits")

if __name__ == "__main__":
    run_benchmark()