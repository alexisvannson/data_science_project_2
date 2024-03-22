import numpy as np
#from scipy.sparse import csr_matrix
import scipy
from scipy import linalg
from scipy.linalg import svd
import matplotlib.pyplot as plt
import time

# 1.1
class Matrix:
    def __init__(self,data):
        self.data = np.array(data)
# The following methods return a new Matrix instance with the result
    def multiply_by_scalar(self, scalar):
        result = []
        for row in self.data:
            new_row = []
            for element in row:
                new_row.append(element * scalar)
            result.append(new_row)
        return Matrix(result)
    
    def matrix_vector_multiplication(self, vector):
        result = []
        for row in self.data:
            row_sum = 0
            for i in range(len(row)):
                row_sum += row[i] * vector[i]
            result.append(row_sum)
        return Matrix(result)  # a single row matrix
   
    def matrix_matrix_multiplication(self, matrix_b):
        result = []
        num_b_cols = len(matrix_b[0])
        num_a_rows = len(self.data)
        for row_index in range(num_a_rows):
            result_row = []
            for col_index in range(num_b_cols):
                element_sum = 0
                for k in range(len(matrix_b)):
                    element_sum += self.data[row_index][k] * matrix_b[k][col_index]
                result_row.append(element_sum)
            result.append(result_row)
        return Matrix(result)

    def add_matrices(self, matrix_b):
        result = []
        for row_index in range(len(self.data)):
            new_row = []
            for col_index in range(len(self.data[row_index])):
                new_element = self.data[row_index][col_index] + matrix_b[row_index][col_index]
                new_row.append(new_element)
            result.append(new_row)
        return Matrix(result)
    
    def difference_of_matrices(self, matrix_b):
        result = []
        for row_index in range(len(self.data)):
            new_row = []
            for col_index in range(len(self.data[row_index])):
                new_element = self.data[row_index][col_index] - matrix_b[row_index][col_index]
                new_row.append(new_element)
            result.append(new_row) 
        return Matrix(result)

    def get_eigenvalues(self):
        eigenvalues, _ = scipy.linalg.eig(self.data)
        return eigenvalues
    
    def get_eigenvectors(self):
        _, eigenvectors = scipy.linalg.eig(self.data)
        return eigenvectors
    
    def SingularValueDecomposition(self):
        U, s, Vh = svd(self.data)
        return U, s, Vh   

    def __str__(self):
        return str(np.array(self.data))
    
class SparseMatrix(Matrix):
    def __init__(self, data, indices,shape):
        Matrix.__init__(self,data)
        self.sparse_matrix = {(row, col): value for value, (row, col) in zip(data, indices)}
        self.shape = shape

    def sparcity(self):
        non_zero_elements = len(self.sparse_matrix)
        rows,lines = self.shape
        total_elements = rows * lines
        return non_zero_elements/ total_elements

    def add_sparse_matrices(self, matrix_b):
        result = self.sparse_matrix.copy()  # Make a copy to ensure that the original matrix remains unchanged
        for key, value in matrix_b.sparse_matrix.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
        # Extract data and indices from dict for new SparseMatrix object
        data = list(result.values())
        indices = list(result.keys())
        return SparseMatrix(data, indices, self.shape)

    def difference_of_sparse_matrices(self, matrix_b):
        result = self.sparse_matrix.copy() 
        for key, value in matrix_b.sparse_matrix.items():
            if key in result:
                result[key] -= value
            else:
                result[key] = -value
        data = list(result.values())
        indices = list(result.keys())
        return SparseMatrix(data, indices, self.shape)

    def multiply_sparse_matrices(self, matrix_b):
        result = {}
        for (row_a, col_a), val_a in self.sparse_matrix.items():
            for (row_b, col_b), val_b in matrix_b.sparse_matrix.items():
                if col_a == row_b:
                    key = (row_a, col_b)
                    if key in result:
                        result[key] += val_a * val_b
                    else:
                        result[key] = val_a * val_b
        data = list(result.values())
        indices = list(result.keys())
        return SparseMatrix(data, indices, self.shape)

    def solve_system(self,target):
        result = scipy.sparse.linalg.spsolve(self.data, target, permc_spec=None, use_umfpack=True)
        return result

    def __str__(self):
        return str(self.sparse_matrix)



class DenseMatrix(Matrix):
    def __init__(self,data):
        Matrix.__init__(self,data)

    def multiply_by_scalar(self, scalar):
        result_data = self.data * scalar
        return DenseMatrix(result_data)
    
    def solve_system(self,target):
        try:
            result = linalg.solve(self.data, target)
            return result
        except np.linalg.LinAlgError as e:
            return str(e)
    def __str__(self):
        return str(self.data)
