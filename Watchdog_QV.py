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
from qiskit.circuit.library import QuantumVolume

# --- Section 1: Custom Transpiler Pass Definition ---

class DecoherenceWatchdog(TransformationPass):
    """
    A transpiler pass to mitigate decoherence in multiple vulnerable idle
    qubits by inserting ancilla-based heralding gadgets at multiple points.
    This version uses modern Qiskit Target and scheduling APIs.
    """

    def __init__(self, backend, durations=None, max_watchdogs=3, min_vulnerability=0.001):
        """
        DecoherenceWatchdog initializer.

        Args:
            backend: A Qiskit backend object with a valid Target.
            durations: InstructionDurations object for getting operation durations.
                      If None, will try to extract from backend target.
            max_watchdogs: Maximum number of watchdog gadgets to insert (default: 3)
            min_vulnerability: Minimum vulnerability threshold for insertion (default: 0.001)
        """
        super().__init__()
        self.backend = backend
        self.target = backend.target
        self.t2_times = {}
        self.dt = self.target.dt
        self._start_times = None  # External scheduling information can be provided here
        self.max_watchdogs = max_watchdogs
        self.min_vulnerability = min_vulnerability
        
        # Store durations for operation duration calculations
        if durations is not None:
            self.durations = durations
        elif hasattr(backend.target, 'durations'):
            self.durations = backend.target.durations()
        else:
            # Fallback: create basic durations
            try:
                self.durations = InstructionDurations.from_backend(backend)
            except:
                self.durations = None

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
        This version supports multi-point insertion by identifying multiple vulnerable 
        idle periods and inserting watchdog gadgets at the most critical locations.
        """
        print(f"[WatchdogPass] Analyzing circuit for vulnerabilities (max_watchdogs={self.max_watchdogs}, min_vulnerability={self.min_vulnerability})...")
        
        # 1. Access scheduling information from property set or external source
        schedule_info = self._start_times or getattr(self.property_set, 'node_start_time', {})
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
        
        # 4. Select multiple vulnerable idle periods for multi-point protection
        # Filter by minimum vulnerability threshold
        vulnerable_periods = [p for p in idle_periods if p["vulnerability"] >= self.min_vulnerability]
        
        if not vulnerable_periods:
            print(f"[WatchdogPass] No idle periods above vulnerability threshold {self.min_vulnerability}. Circuit unchanged.")
            return dag
        
        # Sort by vulnerability (highest first) and select up to max_watchdogs
        vulnerable_periods.sort(key=lambda x: x["vulnerability"], reverse=True)
        selected_periods = vulnerable_periods[:self.max_watchdogs]
        
        print(f"[WatchdogPass] Found {len(vulnerable_periods)} vulnerable periods, inserting {len(selected_periods)} watchdog gadgets:")
        for i, period in enumerate(selected_periods):
            print(f"   {i+1}. Qubit {period['qubit_idx']}: vulnerability={period['vulnerability']:.4f}, duration={period['duration_dt']} dt")

        # 5. Insert watchdog gadgets at all selected vulnerable locations
        return self._insert_multiple_watchdog_gadgets(dag, selected_periods)
    
    def _get_operation_duration(self, node):
        """Get operation duration in dt units using the instruction durations."""
        if self.durations is None:
            # Fallback to hardcoded estimates if no durations available
            duration_map = {
                'cx': 160, 'x': 35, 'h': 35, 'measure': 5440, 'reset': 840,
                'rz': 0, 'sx': 35, 'id': 0, 'barrier': 0, 'swap': 240,
                'u': 35, 'u1': 0, 'u2': 35, 'u3': 35, 'delay': 1
            }
            op_name = node.op.name.lower()
            return duration_map.get(op_name, 35)  # Default to single-qubit gate duration
        
        if node.op.name == "delay":
            # Delay duration is directly specified
            return node.op.duration
        
        # Try to get duration from InstructionDurations object
        unit = "s" if self.durations.dt is None else "dt"
        
        try:
            # Extract qubit indices - simplified approach
            indices = [i for i in range(len(node.qargs))] if node.qargs else []
            duration = self.durations.get(node.op.name, indices, unit=unit)
            
            if duration is not None:
                return int(duration)
        except Exception:
            pass  # Fall through to fallback
        
        # Fallback duration map
        duration_map = {
            'cx': 160, 'x': 35, 'h': 35, 'measure': 5440, 'reset': 840,
            'rz': 0, 'sx': 35, 'id': 0, 'barrier': 0, 'u': 35, 'u3': 35,
            'swap': 240, 'u1': 0, 'u2': 35, 'delay': 1
        }
        return duration_map.get(node.op.name.lower(), 35)
    
    def _fallback_delay_detection(self, dag: DAGCircuit) -> DAGCircuit:
        """Fallback method that looks for explicit Delay instructions and supports multi-point insertion."""
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

        # Filter by minimum vulnerability threshold
        vulnerable_periods = [p for p in idle_periods if p["vulnerability"] >= self.min_vulnerability]
        
        if not vulnerable_periods:
            print(f"[WatchdogPass] No delays above vulnerability threshold {self.min_vulnerability}. Circuit unchanged.")
            return dag
        
        # Sort by vulnerability (highest first) and select up to max_watchdogs
        vulnerable_periods.sort(key=lambda x: x["vulnerability"], reverse=True)
        selected_periods = vulnerable_periods[:self.max_watchdogs]
        
        print(f"[WatchdogPass] Found {len(vulnerable_periods)} vulnerable delays, inserting {len(selected_periods)} watchdog gadgets:")
        for i, period in enumerate(selected_periods):
            print(f"   {i+1}. Qubit {period['qubit_idx']}: vulnerability={period['vulnerability']:.4f}, duration={period['duration_dt']} dt")

        return self._insert_multiple_watchdog_for_delays(dag, selected_periods)
    
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
        
        # Insert delay after the specified node - use the correct method name
        try:
            # Try the new method name first
            dag.apply_operation_back(delay_op, qargs=[target_qubit])
        except AttributeError:
            try:
                # Try older method name
                dag.apply_operation_after(after_node, delay_op, qargs=[target_qubit])
            except AttributeError:
                # Fallback: just add to the end
                dag.apply_operation_back(delay_op, qargs=[target_qubit])
        
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
        # Create unique names for ancilla and herald registers to avoid conflicts
        # Count existing watchdog herald registers
        existing_herald_regs = [creg for creg in dag.cregs if "watchdog_herald" in str(creg)]
        watchdog_id = len(existing_herald_regs)
        
        # Add ancilla and herald registers to the main DAG with unique names
        ancilla_qreg = QuantumRegister(1, f"watchdog_ancilla_{watchdog_id}")
        herald_creg = ClassicalRegister(1, f"watchdog_herald_{watchdog_id}")
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
        print(f"[WatchdogPass] Watchdog {watchdog_id} inserted on data qubit {target_idle_info['qubit_idx']} "
              f"using ancilla qubit {ancilla_qubit_idx}.")
        
        return dag
    
    def _insert_multiple_watchdog_gadgets(self, dag: DAGCircuit, idle_periods):
        """Insert multiple watchdog gadgets at vulnerable idle periods identified by scheduling."""
        # Process each idle period directly by inserting watchdog gadgets
        for i, idle_info in enumerate(idle_periods):
            print(f"[WatchdogPass] Inserting watchdog {i+1}/{len(idle_periods)} on qubit {idle_info['qubit_idx']}")
            
            # Create a delay node to insert in the vulnerable location
            delay_duration = min(idle_info["duration_dt"], 5000)  # Cap at 5000 dt for reasonableness
            delay_op = Delay(delay_duration, "dt")
            target_qubit = idle_info["qubit"]
            
            # Add the delay directly to the DAG
            dag.apply_operation_back(delay_op, qargs=[target_qubit])
            
            # Find the delay node we just added and replace it with watchdog
            delay_nodes = [node for node in dag.op_nodes() if isinstance(node.op, Delay) and node.qargs[0] == target_qubit]
            if delay_nodes:
                # Use the most recently added delay (last in the list)
                latest_delay = delay_nodes[-1]
                delay_info = {
                    "node": latest_delay,
                    "qubit_idx": idle_info["qubit_idx"],
                    "qubit": target_qubit,
                    "duration_dt": delay_duration,
                    "duration_sec": delay_duration * self.dt,
                    "vulnerability": idle_info["vulnerability"]
                }
                dag = self._insert_watchdog_for_delay(dag, delay_info)
        
        return dag
    
    def _insert_multiple_watchdog_for_delays(self, dag: DAGCircuit, idle_periods):
        """Insert multiple watchdog gadgets by replacing multiple delay instructions."""
        # Process delays in reverse order to avoid DAG modification issues during iteration
        sorted_periods = sorted(idle_periods, key=lambda x: x["vulnerability"], reverse=True)
        
        for i, idle_info in enumerate(sorted_periods):
            print(f"[WatchdogPass] Processing watchdog {i+1}/{len(sorted_periods)} on qubit {idle_info['qubit_idx']}")
            dag = self._insert_watchdog_for_delay(dag, idle_info)
        
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
    """Creates a Quantum Volume circuit to test generality."""
    from qiskit.circuit.library import QuantumVolume
    
    num_qubits = 4
    depth = 4
    # Using a fixed seed is critical for reproducibility
    qc = QuantumVolume(num_qubits, depth, seed=42).decompose()
    
    # Add explicit delays to create vulnerable periods for the watchdog
    # Different delays on different qubits to create diverse vulnerable periods
    delays = [4000, 3500, 3000, 2500]  # Different delays for each qubit
    for qubit_idx in range(num_qubits):
        qc.delay(delays[qubit_idx], qubit_idx)
        # Add a second set of delays to create more vulnerable periods
        qc.delay(delays[qubit_idx] // 2, qubit_idx)
    
    qc.measure_all() # Ensure measurements are added
    qc.name = "QuantumVolume_4x4"
    return qc

def post_select_results(counts, herald_bit_index, num_data_clbits):
    """Filters a counts dictionary by checking if the herald bit is '0' (single herald version)."""
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

def post_select_results_multi_herald(counts, herald_bit_indices, num_data_clbits, strategy='strict'):
    """
    Filters a counts dictionary by checking multiple herald bits using different strategies.
    
    Args:
        counts: Dictionary of measurement outcomes and their counts
        herald_bit_indices: List of indices for herald bits (from left to right in outcome string)
        num_data_clbits: Number of data classical bits
        strategy: Post-selection strategy - 'strict', 'majority', 'permissive', or 'weighted'
    
    Returns:
        new_counts: Filtered counts dictionary
        discard_fraction: Fraction of shots discarded
        strategy_info: Dictionary with additional strategy-specific information
    """
    new_counts = {}
    total_shots = sum(counts.values())
    kept_shots = 0
    strategy_info = {}
    
    # Track herald statistics for analysis
    herald_stats = {
        'all_zero': 0,
        'all_one': 0,
        'mixed': 0,
        'zero_count_distribution': {}
    }
    
    for outcome, num_shots in counts.items():
        # Extract herald values for all herald bits
        herald_values = [outcome[idx] for idx in herald_bit_indices]
        zero_count = herald_values.count('0')
        one_count = herald_values.count('1')
        
        # Update statistics
        if zero_count == len(herald_values):
            herald_stats['all_zero'] += num_shots
        elif one_count == len(herald_values):
            herald_stats['all_one'] += num_shots
        else:
            herald_stats['mixed'] += num_shots
        
        herald_stats['zero_count_distribution'][zero_count] = herald_stats['zero_count_distribution'].get(zero_count, 0) + num_shots
        
        # Apply post-selection strategy
        keep_shot = False
        weight = 1.0
        
        if strategy == 'strict':
            # Discard if ANY herald bit is 1 (require all herald bits to be 0)
            keep_shot = (one_count == 0)
            
        elif strategy == 'majority':
            # Keep if MAJORITY of herald bits are 0
            keep_shot = (zero_count > len(herald_values) / 2)
            
        elif strategy == 'permissive':
            # Keep if AT LEAST ONE herald bit is 0
            keep_shot = (zero_count > 0)
            
        elif strategy == 'weighted':
            # Always keep, but weight by fraction of herald bits that are 0
            keep_shot = True
            weight = zero_count / len(herald_values) if len(herald_values) > 0 else 1.0
            
        else:
            raise ValueError(f"Unknown post-selection strategy: {strategy}")
        
        if keep_shot:
            # Extract the data bits (remove all herald bits)
            data_outcome = outcome
            # Remove herald bits in reverse order to maintain indices
            for idx in sorted(herald_bit_indices, reverse=True):
                data_outcome = data_outcome[:idx] + data_outcome[idx+1:]
            
            effective_shots = int(num_shots * weight)
            new_counts[data_outcome] = new_counts.get(data_outcome, 0) + effective_shots
            kept_shots += effective_shots
    
    discard_fraction = (total_shots - kept_shots) / total_shots if total_shots > 0 else 0
    
    strategy_info = {
        'strategy': strategy,
        'herald_count': len(herald_bit_indices),
        'herald_stats': herald_stats,
        'effective_shots': kept_shots,
        'discard_fraction': discard_fraction
    }
    
    return new_counts, discard_fraction, strategy_info

def run_benchmark():
    """Executes the full benchmark protocol using the improved logic structure."""
    print("--- Starting Ancilla-Assisted Decoherence Watchdog Benchmark ---")
    sns.set_style("whitegrid")

    # Setup Environment
    print("\n0. Setting up simulation environment...")
    backend = FakeTorontoV2()
    noise_model = NoiseModel.from_backend(backend)
    
    shots = 300
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
    baseline_circuit = transpile(benchmark_circuit, backend, optimization_level=3)
    
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
            ('swap', None, 240),    # SWAP gate duration (3 CNOT gates)
            ('u', None, 35),        # General U gate duration
            ('u1', None, 0),        # U1 gate duration (virtual)
            ('u2', None, 35),       # U2 gate duration
            ('u3', None, 35),       # U3 gate duration
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
            ('swap', 240),  # SWAP gate duration (3 CNOT gates)
            ('u', 35),      # General U gate duration
            ('u1', 0),      # U1 gate duration (virtual)
            ('u2', 35),     # U2 gate duration
            ('u3', 35),     # U3 gate duration
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
        # First run scheduling analysis to get timing information
        schedule_analysis_pm = PassManager([ASAPScheduleAnalysis(durations)])
        temp_circuit = schedule_analysis_pm.run(laid_out_circuit)
        
        # Extract the scheduling information from the property set
        start_times = schedule_analysis_pm.property_set.get('node_start_time', {})
        
        # Now run the watchdog pass on the original circuit with the scheduling info
        watchdog_pass = DecoherenceWatchdog(backend, durations, max_watchdogs=5, min_vulnerability=0.001)
        # Pass the scheduling information to the watchdog pass
        if hasattr(watchdog_pass, '_start_times'):
            watchdog_pass._start_times = start_times
        
        watchdog_pm = PassManager([watchdog_pass])
        watchdog_inserted_circuit = watchdog_pm.run(laid_out_circuit)
        print("   [Pipeline] Successfully used ASAPScheduleAnalysis for idle time detection")
    except Exception as e:
        print(f"   [Pipeline] ASAPScheduleAnalysis failed: {e}")
        print("   [Pipeline] Falling back to basic watchdog insertion without scheduling")
        
        # Fallback: run without scheduling analysis
        watchdog_pm = PassManager([
            DecoherenceWatchdog(backend, durations, max_watchdogs=5, min_vulnerability=0.001),
        ])
        watchdog_inserted_circuit = watchdog_pm.run(laid_out_circuit)

    # Stage 3: Final optimization and basis gate decomposition
    print("   [Pipeline] Stage 3: Final Optimization...")
    final_opt_pm = PassManager([
        Optimize1qGatesDecomposition(basis=list(backend.target.operation_names)),
        # Use UnitarySynthesis for proper gate decomposition
        UnitarySynthesis(basis_gates=['cx', 'id', 'rz', 'sx', 'x'])
    ])
    watchdog_circuit = final_opt_pm.run(watchdog_inserted_circuit)
    
    # Final step: Deflate the circuit to remove unused qubits
    print("   [Pipeline] Final deflation to remove unused qubits...")
    final_watchdog_circuit = deflate_circuit(watchdog_circuit)
    print(f"   [Pipeline] Final circuit deflated to {final_watchdog_circuit.num_qubits} qubits and {final_watchdog_circuit.num_clbits} classical bits")
    
    # Run the final circuit with additional transpilation for AER compatibility
    print("   [Pipeline] Additional transpilation for AER compatibility...")
    aer_compatible_circuit = transpile(final_watchdog_circuit, 
                                      backend=sim_noise,
                                      optimization_level=1)
    
    watchdog_result = sim_noise.run(aer_compatible_circuit, shots=shots, memory=True).result()
    watchdog_counts = watchdog_result.get_counts()
    memory = watchdog_result.get_memory()
    
    print(f"   Sample watchdog outcomes (first 5):")
    sample_outcomes = list(watchdog_counts.keys())[:5]
    for outcome in sample_outcomes:
        print(f"     {outcome} -> {watchdog_counts[outcome]} counts")


    # D. Post-process watchdog results
    print("\nD. Post-processing watchdog results...")
    
    num_data_clbits = benchmark_circuit.num_clbits
    
    # Detect number of herald bits based on circuit expansion
    num_herald_bits = final_watchdog_circuit.num_clbits - num_data_clbits
    
    if num_herald_bits > 0:
        print(f"   Detected {num_herald_bits} herald bit(s) - performing multi-herald post-selection...")
        
        # Herald bits are at the leftmost positions in Qiskit's little-endian format
        herald_bit_indices = list(range(num_herald_bits))
        
        # Test different post-selection strategies for multi-herald circuits
        strategies = ['strict', 'majority', 'permissive', 'weighted'] if num_herald_bits > 1 else ['strict']
        
        post_selection_results = {}
        
        for strategy in strategies:
            if num_herald_bits == 1:
                # Use original single-herald function for backward compatibility
                ps_counts, discard_frac = post_select_results(watchdog_counts, herald_bit_indices[0], num_data_clbits)
                strategy_info = {
                    'strategy': strategy,
                    'herald_count': 1,
                    'effective_shots': sum(ps_counts.values()),
                    'discard_fraction': discard_frac
                }
            else:
                # Use new multi-herald function
                ps_counts, discard_frac, strategy_info = post_select_results_multi_herald(
                    watchdog_counts, herald_bit_indices, num_data_clbits, strategy)
            
            # Calculate fidelities for this strategy
            # For proper fidelity comparison, we need to expand ideal distribution
            def expand_ideal_to_full_bits(ideal_dist, total_bits, herald_bits):
                """Expand ideal distribution to include herald bits (all set to '0')"""
                expanded = {}
                herald_pattern = '0' * herald_bits  # All heralds should be '0' in ideal case
                for state, prob in ideal_dist.items():
                    expanded[herald_pattern + state] = prob
                return expanded
            
            ideal_full_distribution = expand_ideal_to_full_bits(ideal_distribution, 
                                                              final_watchdog_circuit.num_clbits, 
                                                              num_herald_bits)
            
            # Raw watchdog fidelity: compare expanded ideal with raw watchdog
            watchdog_fidelity = hellinger_fidelity(ideal_full_distribution, watchdog_counts)
            
            # Post-selected fidelity: compare original ideal with post-selected results
            watchdog_ps_fidelity = hellinger_fidelity(ideal_distribution, ps_counts)
            
            post_selection_results[strategy] = {
                'ps_counts': ps_counts,
                'discard_fraction': discard_frac,
                'raw_fidelity': watchdog_fidelity,
                'ps_fidelity': watchdog_ps_fidelity,
                'strategy_info': strategy_info
            }
            
            print(f"   {strategy.upper()} Strategy:")
            print(f"     Post-Selected Fidelity: {watchdog_ps_fidelity:.4f}")
            print(f"     Shot Discard Fraction: {discard_frac:.2%}")
            print(f"     Effective Shots: {strategy_info['effective_shots']}")
        
        # Select the best strategy based on post-selected fidelity
        best_strategy = max(post_selection_results.keys(), 
                           key=lambda s: post_selection_results[s]['ps_fidelity'])
        
        print(f"\n   Best performing strategy: {best_strategy.upper()}")
        
        # Use the best strategy results for final comparison
        best_results = post_selection_results[best_strategy]
        watchdog_ps_counts = best_results['ps_counts']
        discard_fraction = best_results['discard_fraction']
        watchdog_fidelity = best_results['raw_fidelity']
        watchdog_ps_fidelity = best_results['ps_fidelity']
        
    else:
        print("   No herald bits found - watchdog pass did not modify circuit")
        watchdog_ps_counts = watchdog_counts
        discard_fraction = 0.0
        watchdog_fidelity = hellinger_fidelity(ideal_distribution, watchdog_counts)
        watchdog_ps_fidelity = watchdog_fidelity
        post_selection_results = {}
        best_strategy = 'none'

    # E. Compare Results
    print("\n--- Final Benchmark Results ---")
    print(f"Ideal Reference:              Perfect (1.0000)")
    print(f"Baseline Fidelity:            {baseline_fidelity:.4f}")
    print(f"Watchdog Raw Fidelity:        {watchdog_fidelity:.4f}")
    print(f"Watchdog Post-Selected:       {watchdog_ps_fidelity:.4f}")
    print(f"Improvement (PS vs Baseline): {((watchdog_ps_fidelity - baseline_fidelity) / baseline_fidelity) * 100:+.2f}%")
    print(f"Shot Discard Rate:            {discard_fraction:.2%}")
    if num_herald_bits > 0:
        print(f"Number of Herald Bits:        {num_herald_bits}")
        print(f"Best Post-Selection Strategy: {best_strategy.upper()}")
        
        # Show comparison of all strategies if multiple herald bits
        if len(post_selection_results) > 1:
            print("\n--- Multi-Herald Strategy Comparison ---")
            for strategy, results in post_selection_results.items():
                improvement = ((results['ps_fidelity'] - baseline_fidelity) / baseline_fidelity) * 100
                print(f"{strategy.upper():10s}: Fidelity={results['ps_fidelity']:.4f}, "
                      f"Discard={results['discard_fraction']:.2%}, "
                      f"Improvement={improvement:+.2f}%")

    # F. Calculate Quantum Volume state analysis for detailed comparison
    print("\n--- Quantum Volume Analysis ---")
    print("(Detailed entropy and uniformity analysis will be calculated from table values after table generation)")

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
    
    def get_full_bit_probs_for_post_selected(ps_counts_data, raw_counts_full, num_herald_bits):
        """Convert post-selected data-only counts to full-bit probabilities, zeroing herald='1' states"""
        all_full_states = {}
        for outcome in raw_counts_full.keys():
            all_full_states[outcome] = 0
        
        total_ps_shots = sum(ps_counts_data.values())
        if total_ps_shots > 0:
            herald_pattern = '0' * num_herald_bits  # All heralds should be '0'
            for outcome_data, count in ps_counts_data.items():
                outcome_full = herald_pattern + outcome_data
                if outcome_full in all_full_states:
                    all_full_states[outcome_full] = count / total_ps_shots
        
        return all_full_states
    
    def expand_to_full_bits_with_herald_0(counts_data, num_herald_bits):
        """Convert data-only outcomes to full-bit outcomes with herald='0'"""
        expanded = {}
        herald_pattern = '0' * num_herald_bits
        for outcome, count in counts_data.items():
            expanded[herald_pattern + outcome] = count
        return expanded
    
    # Create normalized full-bit probability distributions for all cases including ideal
    ideal_full_probs = normalize_counts(expand_to_full_bits_with_herald_0(ideal_distribution, num_herald_bits))
    baseline_full_probs = normalize_counts(expand_to_full_bits_with_herald_0(baseline_counts, num_herald_bits))
    
    # Handle watchdog data based on whether it has herald bits
    if num_herald_bits > 0:
        # We have a real watchdog circuit with herald bits
        watchdog_raw_full_probs = normalize_counts(watchdog_counts)
        watchdog_ps_full_probs = get_full_bit_probs_for_post_selected(watchdog_ps_counts, watchdog_counts, num_herald_bits)
    else:
        # Watchdog circuit fell back to standard circuit
        watchdog_raw_full_probs = normalize_counts(expand_to_full_bits_with_herald_0(watchdog_counts, num_herald_bits))
        watchdog_ps_full_probs = normalize_counts(expand_to_full_bits_with_herald_0(watchdog_ps_counts, num_herald_bits))
    
    # Print shot statistics
    print(f"   Shot statistics:")
    print(f"   - Baseline shots: {sum(baseline_counts.values())}")
    print(f"   - Watchdog raw shots: {sum(watchdog_counts.values())}")
    print(f"   - Watchdog post-selected shots: {sum(watchdog_ps_counts.values())}")
    
    # Create comprehensive histogram with all normalized distributions including ideal
    legend = [
        f'Ideal',
        f'Baseline', 
        f'Watchdog Raw',
        f'Watchdog PS'
    ]
    hist_data = [ideal_full_probs, baseline_full_probs, watchdog_raw_full_probs, watchdog_ps_full_probs]
    
    # Calculate entropies and uniformity metrics before plotting
    def calculate_entropy(probs):
        """Calculate Shannon entropy of a probability distribution"""
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def calculate_uniformity(probs, num_states):
        """Calculate how close distribution is to uniform (1.0 = perfectly uniform)"""
        if not probs:
            return 0.0
        ideal_prob = 1.0 / num_states
        chi_squared = sum((p - ideal_prob)**2 for p in probs.values()) / ideal_prob
        return max(0.0, 1.0 - chi_squared / num_states)
    
    # Extract 4-bit data distributions for entropy analysis
    def extract_4bit_data_probs(probs_5bit, herald_filter=None):
        """Extract 4-bit data probabilities, optionally filtering by herald bit"""
        data_probs = {}
        for outcome_5bit, prob in probs_5bit.items():
            herald_bit = outcome_5bit[0]
            data_4bit = outcome_5bit[1:]  # Skip herald bit to get 4-bit
            
            if herald_filter is None or herald_bit == herald_filter:
                data_probs[data_4bit] = data_probs.get(data_4bit, 0.0) + prob
        return data_probs
    
    # Calculate metrics for all methods (after all probability distributions are defined)
    ideal_4bit_probs = extract_4bit_data_probs(ideal_full_probs)
    baseline_4bit_probs = extract_4bit_data_probs(baseline_full_probs)
    raw_4bit_probs = extract_4bit_data_probs(watchdog_raw_full_probs)
    ps_4bit_probs = extract_4bit_data_probs(watchdog_ps_full_probs, herald_filter='0')
    
    ideal_entropy = calculate_entropy(ideal_4bit_probs)
    baseline_entropy = calculate_entropy(baseline_4bit_probs)
    raw_entropy = calculate_entropy(raw_4bit_probs)
    ps_entropy = calculate_entropy(ps_4bit_probs)
    
    ideal_uniformity = calculate_uniformity(ideal_4bit_probs, 16)
    baseline_uniformity = calculate_uniformity(baseline_4bit_probs, 16)
    raw_uniformity = calculate_uniformity(raw_4bit_probs, 16)
    ps_uniformity = calculate_uniformity(ps_4bit_probs, 16)
    
    # Create comprehensive multi-panel visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Fidelity Comparison (Main Result)
    methods = ['Ideal', 'Baseline', 'Watchdog Raw', 'Watchdog PS']
    fidelities = [1.0, baseline_fidelity, watchdog_fidelity, watchdog_ps_fidelity]
    colors = ['#2E8B57', '#CD853F', '#4682B4', '#DC143C']
    
    bars1 = ax1.bar(methods, fidelities, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax1.set_title('A. Circuit Fidelity Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, fid in zip(bars1, fidelities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{fid:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight improvement
    improvement = ((watchdog_ps_fidelity - baseline_fidelity) / baseline_fidelity) * 100
    ax1.text(0.5, 0.9, f'Improvement: +{improvement:.2f}%', 
             transform=ax1.transAxes, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Panel B: Quality Metrics (Entropy & Uniformity)
    metrics = ['Entropy', 'Uniformity']
    ideal_metrics = [ideal_entropy/4.0, ideal_uniformity]  # Normalize entropy to [0,1]
    baseline_metrics = [baseline_entropy/4.0, baseline_uniformity]
    ps_metrics = [ps_entropy/4.0, ps_uniformity]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    ax2.bar(x - width, ideal_metrics, width, label='Ideal', color='#2E8B57', alpha=0.8)
    ax2.bar(x, baseline_metrics, width, label='Baseline', color='#CD853F', alpha=0.8)
    ax2.bar(x + width, ps_metrics, width, label='Watchdog PS', color='#DC143C', alpha=0.8)
    
    ax2.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
    ax2.set_title('B. Distribution Quality Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    # Panel C: Top State Probabilities
    # Get all possible 5-bit outcomes for sorting
    all_5bit_outcomes = set()
    all_5bit_outcomes.update(ideal_full_probs.keys())
    all_5bit_outcomes.update(baseline_full_probs.keys())
    all_5bit_outcomes.update(watchdog_raw_full_probs.keys())
    all_5bit_outcomes.update(watchdog_ps_full_probs.keys())
    all_5bit_outcomes = sorted(list(all_5bit_outcomes))
    
    # Calculate combined probabilities for sorting
    combined_probs = {}
    for outcome in all_5bit_outcomes:
        ideal_p = ideal_full_probs.get(outcome, 0.0)
        baseline_p = baseline_full_probs.get(outcome, 0.0)
        raw_p = watchdog_raw_full_probs.get(outcome, 0.0)
        ps_p = watchdog_ps_full_probs.get(outcome, 0.0)
        combined_probs[outcome] = max(ideal_p, baseline_p, raw_p, ps_p)
    top_6_states = sorted(all_5bit_outcomes, key=lambda x: combined_probs[x], reverse=True)[:6]
    
    x_pos = np.arange(len(top_6_states))
    width = 0.2
    
    ideal_top = [ideal_full_probs.get(state, 0) for state in top_6_states]
    baseline_top = [baseline_full_probs.get(state, 0) for state in top_6_states]
    raw_top = [watchdog_raw_full_probs.get(state, 0) for state in top_6_states]
    ps_top = [watchdog_ps_full_probs.get(state, 0) for state in top_6_states]
    
    ax3.bar(x_pos - 1.5*width, ideal_top, width, label='Ideal', color='#2E8B57', alpha=0.8)
    ax3.bar(x_pos - 0.5*width, baseline_top, width, label='Baseline', color='#CD853F', alpha=0.8)
    ax3.bar(x_pos + 0.5*width, raw_top, width, label='Raw', color='#4682B4', alpha=0.8)
    ax3.bar(x_pos + 1.5*width, ps_top, width, label='PS', color='#DC143C', alpha=0.8)
    
    ax3.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax3.set_title('C. Top Quantum States Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'|{state[1:]}>' for state in top_6_states], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Method Efficiency Analysis
    efficiency_metrics = ['Shot Efficiency', 'Error Detection', 'Fidelity Gain']
    efficiency_values = [
        (1 - discard_fraction) * 100,  # Shot efficiency %
        discard_fraction * 100,        # Error detection rate %
        improvement                    # Fidelity improvement %
    ]
    efficiency_colors = ['#32CD32', '#FF6347', '#4169E1']
    
    bars4 = ax4.bar(efficiency_metrics, efficiency_values, color=efficiency_colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Watchdog Method Efficiency', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars4, efficiency_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add method explanation
    fig.suptitle('Ancilla-Assisted Decoherence Watchdog: Quantum Volume Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save with high quality
    filename_comprehensive = f"watchdog_qv_comprehensive_analysis_{benchmark_circuit.num_qubits}q_{shots}shots.png"
    fig.savefig(filename_comprehensive, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"   Comprehensive analysis plot saved to '{filename_comprehensive}'")
    
    # Also keep the original simple histogram
    fig_simple = plot_histogram(hist_data, legend=legend, figsize=(20, 10),
                         title="Decoherence Watchdog: Quantum Volume Circuit Analysis",
                         bar_labels=False)
    
    ax_simple = fig_simple.gca()
    ax_simple.set_ylabel("Probability")
    ax_simple.yaxis.grid(True, linestyle='--')
    plt.tight_layout()
    
    filename_simple = f"watchdog_qv_benchmark_results_{benchmark_circuit.num_qubits}q_{shots}shots.png"
    fig_simple.savefig(filename_simple, dpi=300, bbox_inches='tight')
    print(f"   Simple histogram saved to '{filename_simple}'")
    
    # H. Create comprehensive 4-qubit state probability table
    print("\n--- Complete 4-Qubit State Probability Distribution ---")
    
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
    
    # Create comprehensive table with top most probable states
    print(f"{'5-Bit':<8} {'Data':<6} {'Ideal':<10} {'Baseline':<10} {'Raw':<10} {'PS':<10}")
    print(f"{'State':<8} {'Bits':<6} {'P(state)':<10} {'P(state)':<10} {'P(state)':<10} {'P(state)':<10}")
    print("-" * 64)
    
    # For Quantum Volume, show the most probable states first
    # Calculate combined probabilities for sorting
    combined_probs = {}
    for outcome in all_5bit_outcomes:
        ideal_p = ideal_probs.get(outcome, 0.0)
        baseline_p = baseline_probs.get(outcome, 0.0)
        raw_p = raw_probs.get(outcome, 0.0)
        ps_p = ps_probs.get(outcome, 0.0)
        # Sort by maximum probability across all methods
        combined_probs[outcome] = max(ideal_p, baseline_p, raw_p, ps_p)
    top_states = sorted(all_5bit_outcomes, key=lambda x: combined_probs[x], reverse=True)[:12]
    
    print("Top most probable states:")
    for outcome in top_states:
        data_bits = outcome[1:]
        ideal_p = ideal_probs.get(outcome, 0.0)
        baseline_p = baseline_probs.get(outcome, 0.0)
        raw_p = raw_probs.get(outcome, 0.0)
        ps_p = ps_probs.get(outcome, 0.0)
        
        print(f"{outcome:<8} {data_bits:<6} {ideal_p:<10.4f} {baseline_p:<10.4f} {raw_p:<10.4f} {ps_p:<10.4f}")
    
    # Separate herald=0 and herald=1 states for herald analysis
    herald_0_states = [s for s in all_5bit_outcomes if s[0] == '0']
    herald_1_states = [s for s in all_5bit_outcomes if s[0] == '1']
    
    # Calculate herald bit statistics
    herald_0_total = sum(raw_probs.get(s, 0.0) for s in herald_0_states)
    herald_1_total = sum(raw_probs.get(s, 0.0) for s in herald_1_states)
    
    print(f"\nHerald bit statistics (Raw Watchdog):")
    print(f"  Herald = '0' (keep): {herald_0_total:.4f} ({herald_0_total*100:.1f}%)")
    print(f"  Herald = '1' (discard): {herald_1_total:.4f} ({herald_1_total*100:.1f}%)")
    
    print("-" * 64)
    print(f"{'TOTAL':<8} {'':6} {1.0:<10.4f} {1.0:<10.4f} {1.0:<10.4f} {1.0:<10.4f}")
    
    # I. Calculate Quantum Volume specific analysis
    print("\n--- Quantum Volume Circuit Analysis ---")
    
    print(f"Circuit Details:")
    print(f"  Type: Quantum Volume (4 qubits, depth 4, seed=42)")
    print(f"  Expected: Pseudo-random distribution over computational basis")
    print(f"  Measurement: All computational basis states possible")
    
    print(f"\nDistribution Analysis (Shannon Entropy):")
    print(f"  Ideal entropy:     {ideal_entropy:.4f} bits")
    print(f"  Baseline entropy:  {baseline_entropy:.4f} bits")
    print(f"  Raw entropy:       {raw_entropy:.4f} bits")
    print(f"  Post-selected:     {ps_entropy:.4f} bits")
    print(f"  Max possible:      {math.log2(16):.4f} bits (uniform over 16 states)")
    
    print(f"\nUniformity Analysis (1.0 = perfectly uniform):")
    print(f"  Ideal uniformity:     {ideal_uniformity:.4f}")
    print(f"  Baseline uniformity:  {baseline_uniformity:.4f}")
    print(f"  Raw uniformity:       {raw_uniformity:.4f}")
    print(f"  Post-selected:        {ps_uniformity:.4f}")
    
    # Find the most and least probable 4-bit states
    print(f"\n4-Bit Data State Analysis:")
    sorted_ideal = sorted(ideal_4bit_probs.items(), key=lambda x: x[1], reverse=True)
    sorted_baseline = sorted(baseline_4bit_probs.items(), key=lambda x: x[1], reverse=True)
    sorted_ps = sorted(ps_4bit_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Most probable state (Ideal): |{sorted_ideal[0][0]}⟩ = {sorted_ideal[0][1]:.4f}")
    print(f"  Most probable state (Baseline): |{sorted_baseline[0][0]}⟩ = {sorted_baseline[0][1]:.4f}")
    print(f"  Most probable state (Post-Selected): |{sorted_ps[0][0]}⟩ = {sorted_ps[0][1]:.4f}")
    
    print(f"\nLegend:")
    print(f"Ideal: Noise-free Quantum Volume circuit")
    print(f"Baseline: Noisy with Qiskit optimization_level=3")
    print(f"Raw: Noisy with watchdog transpiler (no post-selection)")
    print(f"PS: Noisy with watchdog transpiler (post-selected on herald='0')")
    print(f"Herald bit is leftmost, data bits are rightmost 4 bits")
    print(f"Higher entropy and uniformity indicate better preservation of quantum volume properties")
    
    print("\n--- Benchmark Complete ---")

if __name__ == "__main__":
    run_benchmark()