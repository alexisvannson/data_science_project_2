import unittest
from matrix_classes import Matrix , SparseMatrix, DenseMatrix
import numpy as np

class TestMatrix(unittest.TestCase):
    
    def test_multiply_by_scalar(self):
        data = [[1, 2], [3, 4]]
        scalar = 2
        expected_result = [[2, 4], [6, 8]]
        matrix = Matrix(data)
        result_matrix = matrix.multiply_by_scalar(scalar)
        self.assertEqual(result_matrix.data.tolist(), expected_result)

    def test_matrix_vector_multiplication(self):
        data = [[1, 2], [3, 4]]
        vector = [1, 2]
        expected_result = [5, 11] 
        matrix = Matrix(data)
        result_matrix = matrix.matrix_vector_multiplication(vector)
        self.assertEqual(result_matrix.data.flatten().tolist(), expected_result)

    def test_matrix_matrix_multiplication(self):
        data_a = [[1, 2], [3, 4]]
        data_b = [[2, 0], [1, 2]]
        expected_result = [[4, 4], [10, 8]]
        matrix_a = Matrix(data_a)
        result_matrix = matrix_a.matrix_matrix_multiplication(data_b)
        self.assertEqual(result_matrix.data.tolist(), expected_result)

    def test_add_matrices(self):
        data_a = [[1, 2], [3, 4]]
        data_b = [[2, 3], [4, 5]]
        expected_result = [[3, 5], [7, 9]]
        matrix_a = Matrix(data_a)
        result_matrix = matrix_a.add_matrices(data_b)
        self.assertEqual(result_matrix.data.tolist(), expected_result)

    def test_difference_of_matrices(self):
        data_a = [[3, 5], [7, 9]]
        data_b = [[1, 2], [3, 4]]
        expected_result = [[2, 3], [4, 5]]
        matrix_a = Matrix(data_a)
        result_matrix = matrix_a.difference_of_matrices(data_b)
        self.assertEqual(result_matrix.data.tolist(), expected_result)

    def test_get_eigenvalues(self):
        data = [[4, 1], [2, 3]]
        matrix = Matrix(data)
        eigenvalues = matrix.get_eigenvalues()
        expected_eigenvalues = sorted(np.linalg.eigvals(data))# we sort eigenvalues to ensure a consistent order for comparison
        result_eigenvalues = sorted(eigenvalues)
        np.testing.assert_array_almost_equal(result_eigenvalues, expected_eigenvalues, decimal=5)

    

class TestSparseMatrix(unittest.TestCase):
    
    def test_add_sparse_matrices(self):
        data_a = [1, 2, 3]
        indices_a = [(0, 0), (1, 1), (2, 2)]
        shape = (3, 3)
        matrix_a = SparseMatrix(data_a, indices_a, shape)

        data_b = [4, 5, 6]
        indices_b = [(0, 0), (1, 1), (2, 2)]
        matrix_b = SparseMatrix(data_b, indices_b, shape)

        result_matrix = matrix_a.add_sparse_matrices(matrix_b)
        
        result_data_dict = result_matrix.sparse_matrix
        result_data = [result_data_dict[key] for key in sorted(result_data_dict)]
        expected_data = [5, 7, 9]  # Element-wise addition

        self.assertEqual(result_data, expected_data)


    
if __name__ == '__main__':
    unittest.main()

