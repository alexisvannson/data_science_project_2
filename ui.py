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

def display_menu():
    print("\nMatrix Operations Menu:")
    print("1. Add Sparse Matrices")
    print("2. Subtract Sparse Matrices")
    print("3. Multiply Sparse Matrices")
    print("4. Scalar Multiplication (Dense Matrix)")
    print("5. Matrix-Vector Multiplication (Dense Matrix)")
    print("6. Matrix-Matrix Multiplication (Dense Matrix)")
    print("7. Compute Eigenvalues (Dense Matrix)")
    print("8. Perform SVD (Dense Matrix)")
    print("9. Exit")
    choice = input("Enter your choice (1-9): ")
    return choice

def main():
    while True:
        choice = display_menu()
        if choice == '1':
            result = A_sparse.add_sparse_matrices(B_sparse)
            print("Addition Result (Sparse):\n", result)
        elif choice == '2':
            result = A_sparse.difference_of_sparse_matrices(B_sparse)
            print("Difference Result (Sparse):\n", result)
        elif choice == '3':
            result = A_sparse.multiply_sparse_matrices(B_sparse)
            print("Multiplication Result (Sparse):\n", result)
        elif choice == '4':
            result = A_dense.multiply_by_scalar(2)
            print("Scalar Multiplication Result (Dense):\n", result)
        elif choice == '5':
            vector = np.random.rand(50)
            result = A_dense.matrix_vector_multiplication(vector)
            print("Matrix-Vector Multiplication Result (Dense):\n", result)
        elif choice == '6':
            result = A_dense.matrix_matrix_multiplication(B_dense.data)
            print("Matrix-Matrix Multiplication Result (Dense):\n", result)
        elif choice == '7':
            eigenvalues = A_dense.get_eigenvalues()
            print("Eigenvalues of Dense Matrix:\n", eigenvalues)
        elif choice == '8':
            U, s, Vh = A_dense.SingularValueDecomposition()
            print("\nSVD of Dense Matrix:\nU:\n", U, "\nS:\n", s, "\nVh:\n", Vh)
        elif choice == '9':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()

