# Data Science Project 

This repository contains the implementation of various matrix operations for both sparse and dense matrices. It's part of a data science project exploring matrix manipulations and performance profiling.

## Features

- Implementation of matrix classes `SparseMatrix` and `DenseMatrix`.
- Methods for performing scalar multiplication, matrix-vector multiplication, and matrix-matrix multiplication.
- Sparse matrix operations that handle addition, subtraction, and multiplication efficiently.
- Profiling and performance analysis for identifying bottlenecks in matrix operations.

## Installation

To get started with this project, clone the repository to your local machine using:

```sh
git clone https://github.com/alexisvannson/data_science_project_2.git
```
# Matrix Operations Library

## Overview

This Python library offers a range of matrix operations, including basic manipulations, eigenvalue and eigenvector computations, singular value decomposition (SVD), and performance analysis for both dense and sparse matrices. Designed for both practical application in matrix algebra and as an educational tool for numerical methods and optimization, this library is a versatile resource for developers and students alike.

## Project Structure

- `matrix_classes.py`: Implements the `Matrix`, `SparseMatrix`, and `DenseMatrix` classes, providing the foundational operations.
- `ui.py`: A command-line interface (CLI) to interactively explore the library's features and operations.
- `test_cases.py`: Contains unit tests to verify the correctness and reliability of the implemented matrix operations.
- `performance.py`: A utility script for profiling and analyzing the performance of matrix operations, emphasizing the comparison between dense and sparse matrix efficiencies.

## Getting Started

### Prerequisites

Ensure you have Python 3.6 or later installed. This library depends on `numpy` and `scipy` for numerical operations:

```sh
pip install numpy scipy
```

### Installation

Clone the repository to your local machine to start working with the matrix operations library:

```sh
git clone https://github.com/alexisvannson/data_science_project_2.git
```

### Using the Library

#### Through User Interface

For an interactive introduction to the library's capabilities, use the CLI provided by `ui.py`. This interface guides you through various matrix operations, offering hands-on examples:

```sh
python ui.py
```

#### Manual Operations

For more customized applications or integration into larger projects, you can import the classes directly into your Python scripts:

``` python
from matrix_classes import Matrix, SparseMatrix, DenseMatrix
import numpy as np

# Example: Creating dense matrices and performing an addition
A_dense = DenseMatrix(np.random.rand(5, 5))
B_dense = DenseMatrix(np.random.rand(5, 5))
addition_result = A_dense.add_matrices(B_dense.data)

print("Result of Dense Matrix Addition:\n", addition_result)
```

This approach allows for direct application of the library's functionalities to meet specific project needs.

### Running Tests

To ensure the reliability of matrix operations, run the comprehensive unit tests included in `test_cases.py`:

```
python -m unittest test_cases.py
```

### Performance Profiling

Analyze and compare the performance of dense and sparse matrix operations to optimize your use of the library:

```
python performance.py
```

This script highlights efficiency considerations, aiding in the selection and application of matrix operations.
