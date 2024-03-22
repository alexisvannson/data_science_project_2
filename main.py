import numpy as np
import scipy
from scipy import linalg
from scipy.linalg import svd
from matrix_classes import SparseMatrix, DenseMatrix 
from performance import generate_sparse_matrix_data


size = 50
non_zero_elements = 25  
data_sparse_a, indices_sparse_a = generate_sparse_matrix_data(size, non_zero_elements)
data_sparse_b, indices_sparse_b = generate_sparse_matrix_data(size, non_zero_elements)

A_sparse = SparseMatrix(data_sparse_a, indices_sparse_a, (size, size))
B_sparse = SparseMatrix(data_sparse_b, indices_sparse_b, (size, size))


data_dense_a = np.random.rand(50, 50)
data_dense_b = np.random.rand(50, 50)

A_dense = DenseMatrix(data_dense_a)
B_dense = DenseMatrix(data_dense_b)


# Addition
add_sparse_result = A_sparse.add_sparse_matrices(B_sparse)
print("Addition Result (Sparse):", add_sparse_result)

# Difference
diff_sparse_result = A_sparse.difference_of_sparse_matrices(B_sparse)
print("Difference Result (Sparse):", diff_sparse_result)

# Multiplication - Note: This might take some time due to the naive implementation
mult_sparse_result = A_sparse.multiply_sparse_matrices(B_sparse)
print("Multiplication Result (Sparse):", mult_sparse_result)

# Scalar multiplication
scalar_result = A_dense.multiply_by_scalar(2)
print("Scalar Multiplication Result (Dense):\n", scalar_result)

# Matrix-vector multiplication
vector = np.random.rand(50)
matrix_vector_result = A_dense.matrix_vector_multiplication(vector)
print("Matrix-Vector Multiplication Result (Dense):\n", matrix_vector_result)

# Matrix-matrix multiplication
matrix_matrix_result = A_dense.matrix_matrix_multiplication(B_dense.data)  # Pass numpy array directly
print("Matrix-Matrix Multiplication Result (Dense):\n", matrix_matrix_result)

# # Computing eigenvalues
eigenvalues = A_dense.get_eigenvalues()
print("\nEigenvalues of Dense Matrix:\n", eigenvalues)

# # Performing SVD
U, s, Vh = A_dense.SingularValueDecomposition()
print("\nSVD of Dense Matrix:\nU:\n", U, "\nS:\n", s, "\nVh:\n", Vh)
