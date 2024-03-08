#!/usr/bin/env python
# coding: utf-8


# Neatly drawn dynamic circuit for 5 or 10 wires, including the option to flip any number of the starting bits,
# and measure the state of any bit
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator

num_wires = 5
# Please change this value to change the number of wires. CNOT gates will generate automatically, connecting
# each wire.

# Create a quantum circuit with the specified number of qubits and classical bits
qc = QuantumCircuit(num_wires, num_wires)
# qc.x(0)
# qc.x(1)
# qc.x(2)

# Apply CNOT gates between each qubit pair
for control_wire in range(num_wires - 1):
    for target_wire in range(control_wire + 1, num_wires):
        qc.cx(control_wire, target_wire)

# Measure the qubit of your choice and store the result in the corresponding classical bit
measure_wire = 1  # Choose the qubit to measure (0-indexed)
output = qc.measure(measure_wire, measure_wire)
print(output)
# Draw the quantum circuit
qc.draw(output="mpl")


# Using Qiskit's unitary_simulator backend to calculate the unitary


num_wires = 3

# Create a quantum circuit with the specified number of qubits and classical bits
qc = QuantumCircuit(num_wires)

# Apply CNOT gates to all pairs of qubits
for control_wire in range(num_wires - 1):
    for target_wire in range(control_wire + 1, num_wires):
        qc.cx(control_wire, target_wire)  # Apply CNOT gates

# Transpile the circuit for the Aer backend
transpiled_circuit = transpile(qc, Aer.get_backend("unitary_simulator"))

# Print the transpiled circuit
print(qc)

# Run the transpiled circuit on the backend
job = Aer.get_backend("unitary_simulator").run(transpiled_circuit)
result = job.result()

# Get the unitary matrix
unitary = result.get_unitary(transpiled_circuit)

# Print the unitary matrix
# Print the unitary matrix using NumPy's formatting
print("Unitary Matrix:")
print(
    np.array2string(
        unitary,
        separator=", ",
        formatter={
            "complex_kind": lambda x: f'{x.real:.3f}{"" if x.imag==0 else f" + {x.imag:.3f}i"}'
        },
    )
)


# Using Qiskit's Operator to calculate the unitary, the output is seen to be the same as that from the unitary_simulator

from qiskit import QuantumCircuit, Aer

num_wires = 5

# Create a quantum circuit with the specified number of qubits and classical bits
qc = QuantumCircuit(num_wires)

# Apply CNOT gates to all pairs of qubits
for control_wire in range(num_wires - 1):
    for target_wire in range(control_wire + 1, num_wires):
        qc.cx(control_wire, target_wire)  # Apply CNOT gates

# Convert the circuit to an operator
op = Operator(qc)

# Print the operator
print(op)

# Compute the complex conjugate of the operator
op_conjugate = op.adjoint()

# Compute the product of the complex conjugate and the operator
product = op_conjugate.compose(op)

# Print the unitary matrix
print("Unitary Matrix:")
print(op.data)

# Print the product of the operator and its complex conjugate
print("Product of operator and its complex conjugate:")
print(product.data)

# Draw the quantum circuit
print("\nQuantum Circuit:")
print(qc.draw())


# Algebraic attempt 1: Failed because it is not easily scalable, as it does not account for entanglement well
# and also does not easily retrieve the unitary

import numpy as np


def apply_cnot(q0, q1):
    """Apply CNOT gate to q0 and q1."""
    # Define CNOT matrix
    cnot_matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

    # Compute Kronecker product of q0 and q1
    state_vector = np.kron(q0, q1)
    print("State vector after Kronecker product:", state_vector)

    # Apply CNOT gate by taking dot product with the CNOT matrix
    result_state = np.dot(cnot_matrix, state_vector)
    print("Result state after applying CNOT matrix:", result_state)

    # Extract the output state of q1
    output_q1 = result_state[2:4]  # Indices corresponding to q1

    # Check if the output in position 0 or position 2 is 0
    if result_state[1] == 1 or result_state[3] == 1:
        output_q1 = np.array([0, 1])  # Change output to [0, 1]
    else:
        output_q1 = np.array([1, 0])  # Change output to [1, 0]

    return output_q1


# Define qubits q0 and q1
q0 = np.array([1, 0])
q1 = np.array([0, 1])

# Apply CNOT gate and get the output state of q1
output_q1 = apply_cnot(q0, q1)
print("Output state of q1 after applying CNOT gate:", output_q1)


# Second try at attempt 1, resulting in a state vector output, from which it is difficult to return the Unitary

import numpy as np


def cnot_matrix():
    """Return the CNOT gate matrix."""
    return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def apply_cnot(qubits):
    """Apply CNOT gates to the qubits."""
    num_qubits = len(qubits)

    # Initialize the result state vector
    result_state = np.array(qubits)

    # Apply CNOT gates using a double for loop
    for control in range(num_qubits - 1):
        for target in range(control + 1, num_qubits):
            # Compute the Kronecker product of the control and target qubits
            control_state = qubits[control]
            target_state = qubits[target]
            kronecker_product = np.kron(target_state, control_state)

            # Apply the CNOT gate by taking the dot product with the CNOT matrix
            cnot_gate = cnot_matrix()
            cnot_result = np.dot(cnot_gate, kronecker_product)

            # Update the result state
            result_state[target] = cnot_result[2:]

    return result_state


# Initialize qubits q0 and q1 with initial states [0,1] and [1,0]
q0 = np.array([0, 1])
q1 = np.array([1, 0])
q2 = np.array([1, 1])
q3 = np.array([1, 0])

# Apply CNOT gates and get the final state of q1
final_state = apply_cnot([q0, q1, q2, q3])

# Print the final state of q1
print("Final state after applying CNOT gates:", final_state)


# Attempting to create the unitary by using the algebraic attempt, with taking the Krokecker product of parallel gates, after first
# SWAPping each wire until there are no non-native CNOTs, then padding each wire with an I matrix, then taking the dot product
# of each group of gates (i.e. each column where there is one CNOT, and num_qubits number of I matrices)

import numpy as np

# Define the CNOT gate matrix
CNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

# Define the identity matrix of appropriate size
I = np.eye(2)

# Define the SWAP gate matrix
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


# Function to apply CNOT gate with potential swaps
def apply_cnot_with_swaps(num_qubits, control, target):
    overall_matrix = np.kron(CNOT, np.eye(2 ** (num_qubits - 2)))

    # Calculate the distance between control and target qubits
    distance = abs(target - control)

    # Apply SWAP gates to bridge the gap if necessary
    for j in range(control + 1, target):
        overall_matrix = np.dot(overall_matrix, SWAP)

    return overall_matrix


# Function to multiply a matrix with its complex conjugate
def multiply_with_conjugate(matrix):
    # Compute the complex conjugate of the input matrix
    conjugate_matrix = np.conjugate(matrix)

    # Multiply the input matrix with its complex conjugate
    result = np.dot(matrix, conjugate_matrix)

    return result


# Example usage
num_qubits = 5

# Apply CNOT gate with potential swaps between qubits 3 and 0
unitary = apply_cnot_with_swaps(num_qubits, 3, 0)
print(unitary)

# Multiply the resulting matrix with its complex conjugate
result = multiply_with_conjugate(unitary)
print("Result of multiplying the matrix with its complex conjugate:")
print(result)
