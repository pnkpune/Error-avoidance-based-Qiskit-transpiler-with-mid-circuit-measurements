# Ancilla-Assisted Decoherence Watchdog Transpiler

**Repository for the Transpilation Challenge - 2025 Qiskit Hackathon at Munich World of Quantum**

This project implements a novel quantum error mitigation technique using an **Ancilla-Assisted Decoherence Watchdog** transpiler pass. The approach improves circuit fidelity on noisy quantum hardware by inserting error-heralding gadgets at the most vulnerable idle periods in quantum circuits.

## 🎯 Project Overview

### The Challenge
Deep quantum circuits suffer from decoherence during idle periods, where qubits wait without active operations. Traditional approaches either ignore these vulnerable periods or apply blanket error correction, missing opportunities for targeted mitigation.

### Our Solution: Decoherence Watchdog
We implement a **custom Qiskit transpiler pass** that:

1. **Analyzes circuit scheduling** using `ASAPScheduleAnalysis` to identify idle periods
2. **Calculates vulnerability scores** based on T₂ decoherence times and idle duration
3. **Inserts watchdog gadgets** at the most vulnerable locations
4. **Uses error heralding** with an ancilla qubit to detect and flag potential errors
5. **Enables post-selection** to improve circuit fidelity

### Key Innovation
The watchdog gadget uses quantum entanglement between data and ancilla qubits during vulnerable idle periods. If decoherence occurs, it's detected through measurement of the herald bit, allowing post-selection of successful runs.

## Technical Implementation

### Core Components

#### 1. DecoherenceWatchdog Transpiler Pass
- **Class**: `DecoherenceWatchdog(TransformationPass)`
- **Purpose**: Custom transpiler pass that identifies vulnerable idle periods and inserts error-heralding gadgets
- **Key Features**:
  - Uses modern Qiskit Target and scheduling APIs
  - Integrates with `ASAPScheduleAnalysis` for precise timing analysis
  - Calculates decoherence probability: `P = 1 - exp(-t/T₂)`
  - Supports both scheduled and explicit delay detection
  - Robust error handling with comprehensive fallbacks

#### 2. Watchdog Gadget Sequence
The inserted gadget follows this protocol:
```
1. H(ancilla)           # Prepare superposition
2. CNOT(data, ancilla)  # Entangle with data qubit
3. Delay(data, t)       # Vulnerable idle period
4. CNOT(data, ancilla)  # Disentangle
5. H(ancilla)           # Measurement basis
6. Measure(ancilla)     # Herald bit
```

**Circuit Diagram:**
```
Data    ─────●─────[Delay(t)]───●─────
             │                  │
Ancilla ─[H]─X──────────────────X─[H]─[M]─► Herald
```

**Quantum States During Execution:**
- **Initial**: |data⟩ ⊗ |0⟩
- **After H**: |data⟩ ⊗ (|0⟩ + |1⟩)/√2
- **After CNOT**: (|data⟩|0⟩ + |data⊕1⟩|1⟩)/√2
- **After Delay**: Vulnerable to decoherence
- **After final operations**: Herald measures error occurrence

#### 3. Professional Transpilation Pipeline
- **Stage 1**: Layout and basis translation using `SabreLayout` and `UnitarySynthesis`
- **Stage 2**: Schedule analysis with `ASAPScheduleAnalysis` and watchdog insertion
- **Stage 3**: Final optimization with `Optimize1qGatesDecomposition`
- **Circuit deflation**: Removes unused qubits for efficiency

#### 4. Comprehensive Benchmarking Framework
- **Ideal Reference**: Noise-free statevector calculation
- **Baseline**: Qiskit's standard optimization (level 3)
- **Watchdog Raw**: With watchdog but no post-selection
- **Watchdog Post-Selected**: Herald bit filtering for improved fidelity

### Advanced Features

#### Robust Duration Handling
- Automatic extraction from backend `InstructionDurations`
- Comprehensive fallbacks for all gate types and qubit combinations
- Support for both scheduled and explicit delay analysis

#### Intelligent Vulnerability Assessment
- T₂-based decoherence probability calculation
- Configurable vulnerability threshold (default: 0.001)
- Priority-based selection of most vulnerable idle periods

#### Consistent Data Analysis
- 5-qubit state space analysis (4 data + 1 herald)
- GHZ state probability calculations extracted directly from probability tables
- Perfect consistency between table values and analysis metrics

## 📊 Benchmark Results

### Test Circuit: 4-Qubit GHZ State
```python
qc.h(0)           # Superposition
qc.cx(0, 1)       # Entanglement
qc.cx(1, 2)       # Chain
qc.cx(2, 3)       # Complete GHZ
qc.delay(5000, 2) # Vulnerable period
qc.measure_all()  # Measurement
```

### Typical Performance Metrics
**Test Configuration:**
- Circuit: 4-qubit GHZ state with 5000 dt delay
- Noise Model: IBM fake backend with realistic parameters
- Shots: 8192 per experiment
- T₂ coherence time: ~100 μs (typical for superconducting qubits)

### Output Analysis
The benchmark produces:
1. **Comprehensive 5-qubit probability table** with herald and data bit separation
2. **Normalized probability plots** comparing all four approaches
3. **GHZ state analysis** with exact probability calculations
4. **Performance metrics** including fidelity improvements and discard rates

### Interpreting Results
- **Herald bit = '0'**: Keep measurement (no error detected)
- **Herald bit = '1'**: Discard measurement (error detected)
- **5-bit outcomes**: Format `HDDD` (Herald + 4 Data bits)
- **Post-selection**: Filter herald='0' and extract data bits

### Herald Bit Interpretation
- **Herald = '0'**: No error detected (keep measurement)
- **Herald = '1'**: Potential error detected (discard measurement)

### Circuit Encoding
- **5-bit outcomes**: `HDDD` (Herald + 4 Data bits)
- **Herald bit**: Leftmost position (index 0)
- **Data bits**: Rightmost 4 positions (indices 1-4)

### Post-Selection Process
1. Filter measurements where herald bit = '0'
2. Extract 4-bit data outcomes
3. Renormalize probability distribution
4. Calculate improved fidelity metrics

### Conclusion

### 1. Scheduling-Aware Analysis
Unlike static approaches, our transpiler uses actual circuit scheduling to identify real vulnerable periods, not just explicit delays.

### 2. Physics-Based Vulnerability Scoring
Vulnerability assessment based on actual T₂ times from backend properties, providing realistic decoherence probability estimates.

### Impact
The Ancilla-Assisted Decoherence Watchdog provides a practical approach to quantum error avoidance that:
- Requires minimal overhead (1 additional qubit)
- Integrates seamlessly with existing transpilation pipelines
- Scales efficiently with circuit size and complexity