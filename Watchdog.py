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
from qiskit.quantum_info import hellinger_fidelity, Statevector
from qiskit.visualization import plot_histogram
from qiskit.transpiler.passes import (UnitarySynthesis, BasisTranslator, Optimize1qGatesDecomposition)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider.backends import FakeTorontoV2
from qiskit.transpiler.passes import SabreLayout

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
        This version uses scheduling information from ASAPScheduleAnalysis to identify
        the most vulnerable idle period and inserts a watchdog gadget there.
        """
        print("[WatchdogPass] Analyzing circuit for vulnerabilities using scheduling information...")
        
        # 1. Access scheduling information from the property set
        # ASAPScheduleAnalysis stores timing information in the property set
        schedule_info = getattr(self.property_set, 'node_start_time', {})
        if not schedule_info:
            print("[WatchdogPass] Warning: No scheduling information found. Falling back to delay detection.")
            return self._fallback_delay_detection(dag)
        
        # 2. Analyze idle periods between operations to find vulnerabilities
        idle_periods = []
        qubit_indices = {qubit: i for i, qubit in enumerate(dag.qubits)}
        
        # Group operations by qubit to analyze idle times
        qubit_operations = {i: [] for i in range(len(dag.qubits))}
        
        for node in dag.op_nodes():
            if not isinstance(node.op, (Delay, Measure)):  # Skip delays and measurements for timing analysis
                for qubit in node.qargs:
                    qubit_idx = qubit_indices[qubit]
                    start_time = schedule_info.get(node, 0)
                    qubit_operations[qubit_idx].append((start_time, node))
        
        # Sort operations by start time for each qubit
        for qubit_idx in qubit_operations:
            qubit_operations[qubit_idx].sort(key=lambda x: x[0])
        
        # Find idle periods between operations
        for qubit_idx, ops in qubit_operations.items():
            if len(ops) < 2:
                continue
                
            for i in range(len(ops) - 1):
                current_end_time = ops[i][0] + self._get_operation_duration(ops[i][1])
                next_start_time = ops[i + 1][0]
                idle_duration_dt = next_start_time - current_end_time
                
                if idle_duration_dt > 100:  # Only consider significant idle periods (>100 dt)
                    idle_duration_sec = idle_duration_dt * self.dt
                    vulnerability = self._get_idle_cost(idle_duration_sec, qubit_idx)
                    
                    idle_periods.append({
                        "qubit_idx": qubit_idx,
                        "qubit": dag.qubits[qubit_idx],
                        "start_time": current_end_time,
                        "duration_dt": idle_duration_dt,
                        "duration_sec": idle_duration_sec,
                        "vulnerability": vulnerability,
                        "after_node": ops[i][1]
                    })
        
        # 3. If no significant idle periods found, fall back to delay detection
        if not idle_periods:
            print("[WatchdogPass] No significant idle periods found in scheduling. Checking for explicit delays.")
            return self._fallback_delay_detection(dag)
        
        # 4. Find the most vulnerable idle period
        target_idle_info = max(idle_periods, key=lambda x: x["vulnerability"])
        
        VULNERABILITY_THRESHOLD = 0.001
        if target_idle_info["vulnerability"] < VULNERABILITY_THRESHOLD:
            print(f"[WatchdogPass] Highest vulnerability {target_idle_info['vulnerability']:.4f} below threshold. Circuit unchanged.")
            return dag

        print(f"[WatchdogPass] Found most vulnerable idle period on qubit {target_idle_info['qubit_idx']} "
              f"with vulnerability score: {target_idle_info['vulnerability']:.4f} "
              f"(duration: {target_idle_info['duration_dt']} dt)")

        # 5. Insert watchdog gadget at the vulnerable location
        return self._insert_watchdog_gadget(dag, target_idle_info)
    
    def _get_operation_duration(self, node):
        """Estimate operation duration in dt units."""
        # Simple duration estimates - in practice these should come from backend
        duration_map = {
            'cx': 160,
            'x': 35,
            'h': 35,
            'measure': 5440,
            'reset': 840,
        }
        op_name = node.op.name.lower()
        return duration_map.get(op_name, 35)  # Default to single-qubit gate duration
    
    def _fallback_delay_detection(self, dag: DAGCircuit) -> DAGCircuit:
        """Fallback method that looks for explicit Delay instructions."""
        print("[WatchdogPass] Using fallback delay detection method...")
        
        # Find explicit delay instructions
        idle_periods = []
        qubit_indices = {qubit: i for i, qubit in enumerate(dag.qubits)}

        for node in dag.op_nodes():
            if isinstance(node.op, Delay):
                qubit = node.qargs[0]
                qubit_idx = qubit_indices[qubit]
                duration_dt = node.op.duration
                duration_sec = duration_dt * self.dt
                vulnerability = self._get_idle_cost(duration_sec, qubit_idx)
                
                idle_periods.append({
                    "node": node,
                    "qubit_idx": qubit_idx,
                    "qubit": qubit,
                    "duration_dt": duration_dt,
                    "duration_sec": duration_sec,
                    "vulnerability": vulnerability
                })

        if not idle_periods:
            print("[WatchdogPass] No idle periods (delays) found. Circuit is unchanged.")
            return dag

        # Find the most vulnerable idle period
        target_idle_info = max(idle_periods, key=lambda x: x["vulnerability"])
        
        VULNERABILITY_THRESHOLD = 0.001
        if target_idle_info["vulnerability"] < VULNERABILITY_THRESHOLD:
            print("[WatchdogPass] No sufficiently vulnerable spot found. Circuit is unchanged.")
            return dag

        print(f"[WatchdogPass] Found vulnerable delay on qubit {target_idle_info['qubit_idx']} "
              f"with vulnerability score: {target_idle_info['vulnerability']:.4f}")

        return self._insert_watchdog_for_delay(dag, target_idle_info)
    
    def _insert_watchdog_gadget(self, dag: DAGCircuit, idle_info):
        """Insert watchdog gadget at a vulnerable idle period identified by scheduling."""
        # For scheduled idle periods, we need to insert the gadget after a specific node
        # This is more complex than replacing a delay, so for now we'll insert a delay and replace it
        
        # Insert a delay instruction at the vulnerable location
        delay_duration = min(idle_info["duration_dt"], 5000)  # Cap at 5000 dt for reasonableness
        delay_op = Delay(delay_duration, "dt")
        
        # Find a good insertion point after the specified node
        after_node = idle_info["after_node"]
        target_qubit = idle_info["qubit"]
        
        # Insert delay after the specified node
        dag.apply_operation_after(after_node, delay_op, qargs=[target_qubit])
        
        # Now find this delay and replace it with watchdog
        for node in dag.op_nodes():
            if isinstance(node.op, Delay) and node.qargs[0] == target_qubit:
                delay_info = {
                    "node": node,
                    "qubit_idx": idle_info["qubit_idx"],
                    "qubit": target_qubit,
                    "duration_dt": delay_duration,
                    "duration_sec": delay_duration * self.dt,
                    "vulnerability": idle_info["vulnerability"]
                }
                return self._insert_watchdog_for_delay(dag, delay_info)
        
        print("[WatchdogPass] Warning: Could not insert delay for watchdog replacement.")
        return dag
    
    def _insert_watchdog_for_delay(self, dag: DAGCircuit, target_idle_info):
        """Insert watchdog gadget by replacing a delay instruction."""
        # Add ancilla and herald registers to the main DAG
        ancilla_qreg = QuantumRegister(1, "watchdog_ancilla")
        herald_creg = ClassicalRegister(1, "watchdog_herald")
        dag.add_qreg(ancilla_qreg)
        dag.add_creg(herald_creg)
        
        # Get references to the newly added qubits/bits
        ancilla_qubit = dag.qubits[-1]  # Last qubit added
        herald_bit = dag.clbits[-1]     # Last classical bit added
        
        # Get the target delay node and its parameters
        target_node = target_idle_info["node"]
        data_qubit = target_idle_info["qubit"]
        idle_duration_dt = target_idle_info["duration_dt"]
        
        # Create a new DAG for the watchdog sequence
        watchdog_dag = DAGCircuit()
        
        # The watchdog DAG needs the same wire structure as what we're replacing
        temp_data_qreg = QuantumRegister(1, "temp_data")
        temp_ancilla_qreg = QuantumRegister(1, "temp_ancilla") 
        temp_herald_creg = ClassicalRegister(1, "temp_herald")
        
        watchdog_dag.add_qreg(temp_data_qreg)
        watchdog_dag.add_qreg(temp_ancilla_qreg)
        watchdog_dag.add_creg(temp_herald_creg)
        
        # Get wire references within the watchdog DAG
        temp_data = watchdog_dag.qubits[0]
        temp_ancilla = watchdog_dag.qubits[1]
        temp_herald = watchdog_dag.clbits[0]
        
        # Build the watchdog sequence:
        # 1. H on ancilla (prepare superposition)
        watchdog_dag.apply_operation_back(HGate(), qargs=[temp_ancilla])
        
        # 2. CNOT data->ancilla (entangle)
        watchdog_dag.apply_operation_back(CXGate(), qargs=[temp_data, temp_ancilla])
        
        # 3. Delay on data qubit (the vulnerable period we're protecting)
        watchdog_dag.apply_operation_back(Delay(idle_duration_dt, "dt"), qargs=[temp_data])
        
        # 4. CNOT data->ancilla (disentangle)
        watchdog_dag.apply_operation_back(CXGate(), qargs=[temp_data, temp_ancilla])
        
        # 5. H on ancilla (measure basis)
        watchdog_dag.apply_operation_back(HGate(), qargs=[temp_ancilla])
        
        # 6. Measure ancilla (herald bit)
        watchdog_dag.apply_operation_back(Measure(), qargs=[temp_ancilla], cargs=[temp_herald])
        
        # Create wire mapping from watchdog DAG to main DAG
        wire_map = {
            temp_data: data_qubit,         # Map temp data qubit to the original data qubit
            temp_ancilla: ancilla_qubit,   # Map temp ancilla to the real ancilla we added
            temp_herald: herald_bit        # Map temp herald to the real herald bit we added
        }
        
        # Use DAG substitution to replace the delay node with the watchdog sequence
        dag.substitute_node_with_dag(target_node, watchdog_dag, wires=wire_map)
        
        ancilla_qubit_idx = len(dag.qubits) - 1
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
    """Creates a 4-qubit GHZ circuit without artificial delays."""
    qc = QuantumCircuit(4, 4)
    
    # Create GHZ state: |0000⟩ + |1111⟩ (unnormalized)
    qc.h(0)  # Put first qubit in superposition
    qc.cx(0, 1)  # Entangle with second qubit
    qc.cx(1, 2)  # Entangle with third qubit
    qc.cx(2, 3)  # Entangle with fourth qubit
    
    # Add a delay to create a vulnerable period for the watchdog
    qc.delay(5000, 2, "dt")  # Longer delay on qubit 2 to create vulnerability
    
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
    """Executes the full benchmark protocol using the improved logic structure."""
    print("--- Starting Ancilla-Assisted Decoherence Watchdog Benchmark ---")
    sns.set_style("whitegrid")

    # Setup Environment
    print("\n0. Setting up simulation environment...")
    backend = FakeTorontoV2()
    noise_model = NoiseModel.from_backend(backend)
    
    shots = 8192
    benchmark_circuit = create_benchmark_circuit()
    print(f"   Benchmark circuit '{benchmark_circuit.name}' created.")
    print(f"   Using backend '{backend.name}' for noise and transpilation properties.")

    # A. Calculate the ideal, noise-free result for fidelity comparison
    print("\nA. Calculating ideal noise-free reference...")
    ideal_circuit = benchmark_circuit.copy()
    ideal_circuit.remove_final_measurements(inplace=True)
    ideal_state = Statevector(ideal_circuit)
    ideal_distribution = ideal_state.probabilities_dict()
    print(f"   Ideal distribution calculated with {len(ideal_distribution)} basis states")

    # B. Run the Baseline: Qiskit's best generic optimization
    print("\nB. Running Baseline (optimization_level=3)...")
    baseline_circuit = transpile(benchmark_circuit, backend, optimization_level=0)
    
    # Deflate the baseline circuit to remove unused qubits
    baseline_circuit = deflate_circuit(baseline_circuit)
    print(f"   [Pipeline] Baseline circuit deflated to {baseline_circuit.num_qubits} qubits")
    
    sim_noise = AerSimulator.from_backend(backend)
    baseline_result = sim_noise.run(baseline_circuit, shots=shots).result()
    baseline_counts = baseline_result.get_counts()
    baseline_fidelity = hellinger_fidelity(ideal_distribution, baseline_counts)
    print(f"   Baseline Fidelity: {baseline_fidelity:.4f}")

    # 4. Run Our Custom Solution
    print("\n4. Running Our Custom 'Decoherence Watchdog' Pass...")
    print("   [Pipeline] Building a professional-grade transpilation pipeline...")

    # Build a structured PassManager following Qiskit documentation
    
    # Stage 1: Layout and basis translation
    print("   [Pipeline] Stage 1: Layout and Basis Translation...")
    layout_pm = PassManager([
        SabreLayout(coupling_map=backend.coupling_map, max_iterations=4, seed=42),
        UnitarySynthesis(basis_gates=backend.target.operation_names),
    ])
    laid_out_circuit = layout_pm.run(benchmark_circuit)
    
    # Stage 2: Schedule analysis and watchdog insertion
    print("   [Pipeline] Stage 2: Schedule Analysis and Watchdog Insertion...")
    
    # Create comprehensive InstructionDurations that should work with most backends
    try:
        # Try to get durations from backend first
        base_durations = InstructionDurations.from_backend(backend)
        
        # Add/update all missing durations to ensure comprehensive coverage
        additional_durations = [
            ('h', None, 35),        # Hadamard gate duration
            ('x', None, 35),        # X gate duration
            ('reset', None, 840),   # Reset duration
            ('delay', None, 1),     # Delay duration (per dt)
            ('barrier', None, 0),   # Barrier has no duration
            ('id', None, 0),        # Identity gate has no duration
            ('rz', None, 0),        # Virtual Z rotation
            ('sx', None, 35),       # SX gate duration
        ]
        
        # Add durations for all qubits in the backend to avoid missing duration errors
        comprehensive_durations = []
        for name, qubits, duration in additional_durations:
            if qubits is None:
                for qubit_idx in range(backend.num_qubits):
                    comprehensive_durations.append((name, [qubit_idx], duration))
                # Also add the general version
                comprehensive_durations.append((name, None, duration))
            else:
                comprehensive_durations.append((name, qubits, duration))
        
        base_durations.update(comprehensive_durations)
        durations = base_durations
        print("   [Pipeline] Enhanced backend durations with comprehensive fallbacks")
        
    except Exception as e:
        print(f"   [Pipeline] Backend duration extraction failed: {e}")
        
        # Create complete fallback durations for all qubits
        fallback_durations = []
        basic_gates = [
            ('cx', 160),    # CNOT duration
            ('x', 35),      # X gate duration
            ('h', 35),      # Hadamard gate duration  
            ('measure', 5440), # Measurement duration
            ('reset', 840), # Reset duration
            ('delay', 1),   # Delay duration (per dt)
            ('barrier', 0), # Barrier has no duration
            ('id', 0),      # Identity gate has no duration
            ('rz', 0),      # Virtual Z rotation
            ('sx', 35),     # SX gate duration
        ]
        
        # Add durations for every qubit to avoid missing duration errors
        for gate_name, gate_duration in basic_gates:
            # Add general version
            fallback_durations.append((gate_name, None, gate_duration))
            # Add per-qubit versions
            for qubit_idx in range(backend.num_qubits):
                fallback_durations.append((gate_name, [qubit_idx], gate_duration))
        
        # For two-qubit gates, add all qubit pairs from coupling map
        if hasattr(backend, 'coupling_map') and backend.coupling_map:
            for edge in backend.coupling_map.get_edges():
                fallback_durations.append(('cx', list(edge), 160))
        
        durations = InstructionDurations(
            fallback_durations, 
            dt=backend.dt if hasattr(backend, 'dt') else 0.1/1e9
        )
        print("   [Pipeline] Created comprehensive fallback instruction durations")
    
    # Try to run with scheduling analysis, but fall back if it fails
    try:
        # First run scheduling analysis
        schedule_analysis_pm = PassManager([ASAPScheduleAnalysis(durations)])
        scheduled_circuit = schedule_analysis_pm.run(laid_out_circuit)
        
        # Then run watchdog pass with access to scheduling information
        watchdog_pm = PassManager([DecoherenceWatchdog(backend)])
        # Transfer the property set from scheduling analysis to watchdog pass
        watchdog_pm.property_set = schedule_analysis_pm.property_set
        
        watchdog_inserted_circuit = watchdog_pm.run(scheduled_circuit)
        print("   [Pipeline] Successfully used ASAPScheduleAnalysis for idle time detection")
    except Exception as e:
        print(f"   [Pipeline] ASAPScheduleAnalysis failed: {e}")
        print("   [Pipeline] Falling back to basic watchdog insertion without scheduling")
        
        # Fallback: run without scheduling analysis
        watchdog_pm = PassManager([
            DecoherenceWatchdog(backend),
        ])
        watchdog_inserted_circuit = watchdog_pm.run(laid_out_circuit)

    # Stage 3: Final optimization 
    print("   [Pipeline] Stage 3: Final Optimization...")
    final_opt_pm = PassManager([
        Optimize1qGatesDecomposition(basis=list(backend.target.operation_names)),
    ])
    watchdog_circuit = final_opt_pm.run(watchdog_inserted_circuit)
    
    # Final step: Deflate the circuit to remove unused qubits
    print("   [Pipeline] Final deflation to remove unused qubits...")
    final_watchdog_circuit = deflate_circuit(watchdog_circuit)
    print(f"   [Pipeline] Final circuit deflated to {final_watchdog_circuit.num_qubits} qubits and {final_watchdog_circuit.num_clbits} classical bits")
    
    # Run the final circuit
    watchdog_result = sim_noise.run(final_watchdog_circuit, shots=shots, memory=True).result()
    watchdog_counts = watchdog_result.get_counts()
    memory = watchdog_result.get_memory()
    
    print(f"   Sample watchdog outcomes (first 5):")
    sample_outcomes = list(watchdog_counts.keys())[:5]
    for outcome in sample_outcomes:
        print(f"     {outcome} -> {watchdog_counts[outcome]} counts")



    # D. Post-process watchdog results
    print("\nD. Post-processing watchdog results...")
    
    num_data_clbits = benchmark_circuit.num_clbits
    herald_bit_index = 0  # Leftmost bit in Qiskit's little-endian format
    
    # Check if we have a herald bit (more classical bits than original)
    # Create expanded ideal distribution for 5-bit comparison (herald='0' for all ideal states)
    def expand_ideal_to_5bit(ideal_4bit_dist):
        """Expand 4-bit ideal distribution to 5-bit with herald='0'"""
        expanded = {}
        for state, prob in ideal_4bit_dist.items():
            expanded['0' + state] = prob  # Herald='0' + data bits
        return expanded
    
    if final_watchdog_circuit.num_clbits > num_data_clbits:
        print("   Performing post-selection on herald bit...")
        watchdog_ps_counts, discard_fraction = post_select_results(watchdog_counts, herald_bit_index, num_data_clbits)
        
        # For proper fidelity comparison, we need to expand ideal distribution to 5-bit
        ideal_5bit_distribution = expand_ideal_to_5bit(ideal_distribution)
        
        # Raw watchdog fidelity: compare expanded ideal with raw watchdog (both 5-bit)
        watchdog_fidelity = hellinger_fidelity(ideal_5bit_distribution, watchdog_counts)
        
        # Post-selected fidelity: compare 4-bit ideal with post-selected results (both 4-bit)
        watchdog_ps_fidelity = hellinger_fidelity(ideal_distribution, watchdog_ps_counts)
        
        print(f"   Raw Watchdog Fidelity: {watchdog_fidelity:.4f}")
        print(f"   Post-Selected Fidelity: {watchdog_ps_fidelity:.4f}")
        print(f"   Shot Discard Fraction: {discard_fraction:.2%}")
    else:
        print("   No herald bit found - watchdog pass did not modify circuit")
        watchdog_ps_counts = watchdog_counts
        discard_fraction = 0.0
        watchdog_fidelity = hellinger_fidelity(ideal_distribution, watchdog_counts)
        watchdog_ps_fidelity = watchdog_fidelity
        discard_fraction = 0.0
        watchdog_fidelity = hellinger_fidelity(ideal_distribution, watchdog_counts)
        watchdog_ps_fidelity = watchdog_fidelity

    # E. Compare Results
    print("\n--- Final Benchmark Results ---")
    print(f"Ideal Reference:              Perfect (1.0000)")
    print(f"Baseline Fidelity:            {baseline_fidelity:.4f}")
    print(f"Watchdog Raw Fidelity:        {watchdog_fidelity:.4f}")
    print(f"Watchdog Post-Selected:       {watchdog_ps_fidelity:.4f}")
    print(f"Improvement (PS vs Baseline): {((watchdog_ps_fidelity - baseline_fidelity) / baseline_fidelity) * 100:+.2f}%")
    print(f"Shot Discard Rate:            {discard_fraction:.2%}")

    # F. Calculate GHZ state probabilities for detailed analysis
    print("\n--- GHZ State Analysis ---")
    
    def calculate_ghz_probabilities(counts, label, herald_filter=None):
        """Calculate GHZ state probabilities with proper herald handling"""
        total_shots = sum(counts.values())
        if not counts or total_shots == 0:
            print(f"{label}: No valid counts")
            return 0.0
        
        if len(list(counts.keys())[0]) == 4:
            # 4-qubit data
            p_0000 = counts.get('0000', 0) / total_shots
            p_1111 = counts.get('1111', 0) / total_shots
        else:
            # 5-qubit data - need to handle herald filtering correctly
            if herald_filter is not None:
                # For post-selected case, only consider herald='0' states
                filtered_counts = {}
                for outcome, count in counts.items():
                    herald_bit = outcome[0]  # Herald is leftmost bit
                    data_bits = outcome[1:]  # Data bits are rightmost 4
                    if herald_bit == herald_filter:
                        if data_bits in filtered_counts:
                            filtered_counts[data_bits] += count
                        else:
                            filtered_counts[data_bits] = count
                
                filtered_total = sum(filtered_counts.values())
                if filtered_total == 0:
                    p_0000 = p_1111 = 0.0
                else:
                    p_0000 = filtered_counts.get('0000', 0) / filtered_total
                    p_1111 = filtered_counts.get('1111', 0) / filtered_total
            else:
                # For raw case, sum across all herald values for each data state
                data_counts = {}
                for outcome, count in counts.items():
                    data_bits = outcome[1:]  # Extract data bits
                    if data_bits in data_counts:
                        data_counts[data_bits] += count
                    else:
                        data_counts[data_bits] = count
                
                p_0000 = data_counts.get('0000', 0) / total_shots
                p_1111 = data_counts.get('1111', 0) / total_shots
        
        ghz_fidelity = p_0000 + p_1111
        print(f"{label}:")
        print(f"  P(|0000⟩): {p_0000:.4f}")
        print(f"  P(|1111⟩): {p_1111:.4f}")
        print(f"  GHZ Fidelity: {ghz_fidelity:.4f}")
        return ghz_fidelity
    
    # Add ideal distribution analysis
    ideal_ghz_fid = calculate_ghz_probabilities(ideal_distribution, "Ideal (Noise-Free)")
    baseline_ghz_fid = calculate_ghz_probabilities(baseline_counts, "Baseline")
    watchdog_raw_ghz_fid = calculate_ghz_probabilities(watchdog_counts, "Watchdog Raw")
    watchdog_ps_ghz_fid = calculate_ghz_probabilities(watchdog_ps_counts, "Watchdog Post-Selected", herald_filter='0')

    # G. Generate comprehensive visualization and analysis
    print("\nG. Generating comprehensive comparison plots and analysis...")
    
    # Helper functions for data preparation
    def expand_to_5bit_with_herald_0(counts_4bit):
        """Convert 4-bit outcomes to 5-bit with herald='0' for comparison"""
        expanded = {}
        for outcome, count in counts_4bit.items():
            expanded['0' + outcome] = count
        return expanded
    
    def normalize_counts(counts):
        total = sum(counts.values())
        return {outcome: count/total for outcome, count in counts.items()}
    
    def get_5bit_probs_for_post_selected(ps_counts_4bit, raw_counts_5bit):
        """Convert post-selected 4-bit data to 5-bit probabilities, zeroing herald='1' states"""
        all_5bit_states = {}
        for outcome in raw_counts_5bit.keys():
            all_5bit_states[outcome] = 0
        
        total_ps_shots = sum(ps_counts_4bit.values())
        if total_ps_shots > 0:
            for outcome_4bit, count in ps_counts_4bit.items():
                outcome_5bit = '0' + outcome_4bit
                if outcome_5bit in all_5bit_states:
                    all_5bit_states[outcome_5bit] = count / total_ps_shots
        
        return all_5bit_states
    
    # Create normalized 5-qubit probability distributions for all cases including ideal
    ideal_5bit_probs = normalize_counts(expand_to_5bit_with_herald_0(ideal_distribution))
    baseline_5bit_probs = normalize_counts(expand_to_5bit_with_herald_0(baseline_counts))
    
    # Handle watchdog data based on whether it has herald bit
    if final_watchdog_circuit.num_clbits > num_data_clbits:
        # We have a real watchdog circuit with herald bit
        watchdog_raw_5bit_probs = normalize_counts(watchdog_counts)
        watchdog_ps_5bit_probs = get_5bit_probs_for_post_selected(watchdog_ps_counts, watchdog_counts)
    else:
        # Watchdog circuit fell back to standard circuit
        watchdog_raw_5bit_probs = normalize_counts(expand_to_5bit_with_herald_0(watchdog_counts))
        watchdog_ps_5bit_probs = normalize_counts(expand_to_5bit_with_herald_0(watchdog_ps_counts))
    
    # Print shot statistics
    print(f"   Shot statistics:")
    print(f"   - Baseline shots: {sum(baseline_counts.values())}")
    print(f"   - Watchdog raw shots: {sum(watchdog_counts.values())}")
    print(f"   - Watchdog post-selected shots: {sum(watchdog_ps_counts.values())}")
    
    # Create comprehensive histogram with all normalized distributions including ideal
    legend = [
        f'Ideal (GHZ: {ideal_ghz_fid:.3f})',
        f'Baseline (GHZ: {baseline_ghz_fid:.3f})',
        f'Watchdog Raw (GHZ: {watchdog_raw_ghz_fid:.3f})',
        f'Watchdog PS (GHZ: {watchdog_ps_ghz_fid:.3f})'
    ]
    hist_data = [ideal_5bit_probs, baseline_5bit_probs, watchdog_raw_5bit_probs, watchdog_ps_5bit_probs]
    
    fig = plot_histogram(hist_data, legend=legend, figsize=(20, 10),
                         title="Decoherence Watchdog: 5-Qubit Probability Distributions",
                         bar_labels=False)
    
    ax = fig.gca()
    ax.set_ylabel("Probability")
    ax.yaxis.grid(True, linestyle='--')
    plt.tight_layout()
    
    fig.savefig("watchdog_benchmark_results.png", dpi=300, bbox_inches='tight')
    print("   Results plot saved to 'watchdog_benchmark_results.png'")
    
    # H. Create comprehensive 5-qubit state probability table
    print("\n--- Complete 5-Qubit State Probability Distribution ---")
    
    # Prepare all data for the table including ideal
    ideal_5bit = expand_to_5bit_with_herald_0(ideal_distribution)
    ideal_probs = normalize_counts(ideal_5bit)
    baseline_5bit = expand_to_5bit_with_herald_0(baseline_counts)
    baseline_probs = normalize_counts(baseline_5bit)
    
    if final_watchdog_circuit.num_clbits > num_data_clbits:
        # Real watchdog circuit with herald bit
        raw_5bit = watchdog_counts
        raw_probs = normalize_counts(raw_5bit)
        
        # For post-selected, create 5-bit representation with herald='0' states only
        ps_5bit = {}
        for outcome in raw_5bit.keys():
            ps_5bit[outcome] = 0
        for outcome_4bit, count in watchdog_ps_counts.items():
            outcome_5bit = '0' + outcome_4bit
            if outcome_5bit in ps_5bit:
                ps_5bit[outcome_5bit] = count
        
        ps_total_shots = sum(count for count in ps_5bit.values() if count > 0)
        ps_probs = {outcome: count/ps_total_shots if ps_total_shots > 0 else 0 for outcome, count in ps_5bit.items()}
    else:
        # Fallback case - no herald bit
        raw_5bit = expand_to_5bit_with_herald_0(watchdog_counts)
        raw_probs = normalize_counts(raw_5bit)
        ps_5bit = expand_to_5bit_with_herald_0(watchdog_ps_counts)
        ps_probs = normalize_counts(ps_5bit)
    
    # Get all possible 5-bit outcomes
    all_5bit_outcomes = set()
    all_5bit_outcomes.update(ideal_probs.keys())
    all_5bit_outcomes.update(baseline_probs.keys())
    all_5bit_outcomes.update(raw_probs.keys())
    all_5bit_outcomes.update(ps_probs.keys())
    all_5bit_outcomes = sorted(list(all_5bit_outcomes))
    
    # Create comprehensive table
    print(f"{'5-Bit':<8} {'Data':<6} {'Ideal':<10} {'Baseline':<10} {'Raw':<10} {'PS':<10}")
    print(f"{'State':<8} {'Bits':<6} {'P(state)':<10} {'P(state)':<10} {'P(state)':<10} {'P(state)':<10}")
    print("-" * 64)
    
    # Separate herald=0 and herald=1 states for clarity
    herald_0_states = [s for s in all_5bit_outcomes if s[0] == '0']
    herald_1_states = [s for s in all_5bit_outcomes if s[0] == '1']
    
    print("Herald = '0' states:")
    for outcome in herald_0_states:
        data_bits = outcome[1:]
        ideal_p = ideal_probs.get(outcome, 0.0)
        baseline_p = baseline_probs.get(outcome, 0.0)
        raw_p = raw_probs.get(outcome, 0.0)
        ps_p = ps_probs.get(outcome, 0.0)
        
        # Highlight GHZ states
        ghz_marker = " *" if data_bits in ['0000', '1111'] else ""
        print(f"{outcome:<8} {data_bits:<6} {ideal_p:<10.4f} {baseline_p:<10.4f} {raw_p:<10.4f} {ps_p:<10.4f}{ghz_marker}")
    
    if herald_1_states:
        print("\nHerald = '1' states:")
        for outcome in herald_1_states:
            data_bits = outcome[1:]
            ideal_p = ideal_probs.get(outcome, 0.0)
            baseline_p = baseline_probs.get(outcome, 0.0)
            raw_p = raw_probs.get(outcome, 0.0)
            ps_p = ps_probs.get(outcome, 0.0)
            print(f"{outcome:<8} {data_bits:<6} {ideal_p:<10.4f} {baseline_p:<10.4f} {raw_p:<10.4f} {ps_p:<10.4f}")
    
    print("-" * 64)
    print(f"{'TOTAL':<8} {'':6} {1.0:<10.4f} {1.0:<10.4f} {1.0:<10.4f} {1.0:<10.4f}")
    
    print(f"\nLegend:")
    print(f"* = GHZ states (|0000⟩ or |1111⟩)")
    print(f"Ideal: Noise-free perfect GHZ state")
    print(f"Baseline: Noisy with Qiskit optimization_level=3")
    print(f"Raw: Noisy with watchdog transpiler (no post-selection)")
    print(f"PS: Noisy with watchdog transpiler (post-selected on herald='0')")
    print(f"Herald bit is leftmost, data bits are rightmost 4 bits")
    
    print("\n--- Benchmark Complete ---")

if __name__ == "__main__":
    run_benchmark()