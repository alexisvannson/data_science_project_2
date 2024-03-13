import numpy
from scipy.sparse import csr_matrix
import scipy
from scipy import linalg
from scipy.linalg import svd

# 1.1
class Matrix:
    def __init__(self,data):
        self.data = numpy.array(data)
# The following methods return a new Matrix instance with the result
    def multiply_by_scalar(self, scalar):
        result = self.data * scalar
        return Matrix(result)
    
    def matrix_vector_multiplication(self, vector):
        result = numpy.dot(self.data, vector)
        return Matrix(result) 
    
    def matrix_matrix_multiplication(self, matrix_b):
        matrix_b = numpy.array(matrix_b)
        result = numpy.dot(self.data, matrix_b)
        return Matrix(result)
    
    def add_matrices(self, matrix_b):
        matrix_b = numpy.array(matrix_b)
        result = numpy.add(self.data, matrix_b)
        return Matrix(result)
    
    def difference_of_matrices(self, matrix_b):
        matrix_b = numpy.array(matrix_b)
        result = numpy.subtract(self.data, matrix_b)
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
        return str(numpy.array(self.data))
    


class SparseMatrix(Matrix):
    def __init__(self,data,row_ind, col_ind, shape):
        Matrix.__init__(self,data)
        self.row_ind = row_ind
        self.col_ind = col_ind
        self.shape = shape
        self.sparse_matrix = csr_matrix((self.data, (row_ind, col_ind)), shape)

    def sparcity(self):
        non_zero_elements = self.sparse_matrix.size
        rows,lines = self.shape
        total_elements = rows * lines
        return non_zero_elements/ total_elements

    def matrix_matrix_multiplication(self,sparse_matrix_b):  #à refaire à la main 
        return self.sparse_matrix * sparse_matrix_b


    def add_matrices(self,sparse_matrix_b):#à refaire à la main
        return self.sparse_matrix + sparse_matrix_b

    def difference_of_matrices(self,sparse_matrix_b):#à refaire à la main
        return self.sparse_matrix - sparse_matrix_b
   
    def solve_system(self,target):
        result = scipy.sparse.linalg.spsolve(self.data, target, permc_spec=None, use_umfpack=True)
        return result

    def __str__(self):
        return str(csr_matrix((self.data, (self.row_ind, self.col_ind)), self.shape))



class DenseMatrix(Matrix):
    def __init__(self,data,row_ind, col_ind, shape):
        Matrix.__init__(self,data)
        self.row_ind = row_ind
        self.col_ind = col_ind
        self.shape = shape
        self.SparseMatrix = csr_matrix((self.data, (row_ind, col_ind)), shape).todense()
    
    def multiply_by_scalar(self, scalar):
        result_data = self.data * scalar
        return DenseMatrix(result_data)
    
    def solve_system(self,target):
        try:
            result = linalg.solve(self.data, target)
            return result
        except numpy.linalg.LinAlgError as e:
            return str(e)

    def __str__(self):
        return str(self.SparseMatrix)
    


data = [1,87,64,12,23]
data2 = [1,870,6,2,43]
row_ind = [0,0,2,3,4]
col_ind = [0,1,3,1,2]
shape= (5,5)
A = SparseMatrix(data,row_ind, col_ind, shape)
B = DenseMatrix(data2,row_ind, col_ind, shape)
#A.__str__()
#print(A.sparcity())
#print(A.SparseMatrix)
#print (B.multiply_by_scalar(1000).todense())
matrix_b = [[0,1,1,1,3],[4,2,2,1,3],[4,2,2,1,3],[4,2,2,1,3],[4,2,2,1,3]]
B.add_matrices(matrix_b)


#1.3
# eigenvalues squared matrix, SVD decompsition, using Scipy 

A1 = DenseMatrix(data2,row_ind, col_ind, shape)
print(A1.get_eigenvalues())
print(A1.get_eigenvectors())



