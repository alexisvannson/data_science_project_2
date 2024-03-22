import numpy as np
import time
import cProfile
import pstats
import matplotlib.pyplot as plt
from matrix_classes import SparseMatrix, DenseMatrix  # Import the classes from your module

def profile_method(method, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    method(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
    return stats

# This is just a placeholder; replace with the actual function from your module if it's defined there
def generate_sparse_matrix_data(size, num_elements):
    data = np.random.randint(1, 10, size=num_elements)
    indices = np.random.choice(np.arange(size*size), size=num_elements, replace=False)
    indices = [(index // size, index % size) for index in indices]
    return data, indices

def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result

def profile_matrix_operations(start_size, max_size, increase_factor):
    sizes = []
    sparse_times = []
    dense_times = []

    current_size = start_size
    while current_size <= max_size:
        # Generate sparse matrix data
        data, indices = generate_sparse_matrix_data(current_size, int(current_size * current_size * 0.1))
        sparse_matrix = SparseMatrix(data, indices, (current_size, current_size))

        # Generate dense matrix data
        dense_matrix = DenseMatrix(np.random.rand(current_size, current_size))

        # Time the sparse matrix operation (e.g., addition)
        time_taken_sparse, _ = time_function(sparse_matrix.add_sparse_matrices, sparse_matrix)
        sparse_times.append(time_taken_sparse)

        # Time the dense matrix operation (e.g., addition)
        time_taken_dense, _ = time_function(dense_matrix.add_matrices, dense_matrix.data)
        dense_times.append(time_taken_dense)

        sizes.append(current_size)
        current_size *= increase_factor  # Double the size for the next iteration

    return sizes, sparse_times, dense_times

def plot_results(sizes, sparse_times, dense_times):
    plt.plot(sizes, sparse_times, label='Sparse Matrix Operations')
    plt.plot(sizes, dense_times, label='Dense Matrix Operations')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Performance of Matrix Operations')
    plt.show()

# Set the start size, max size, and how much to increase the size each time
start_size = 10  # Starting from a 10x10 matrix
max_size = 160  # Set a maximum size to not exceed computational resources
increase_factor = 2  # Double the size in each step

# Run the profiling and plotting functions
sizes, sparse_times, dense_times = profile_matrix_operations(start_size, max_size, increase_factor)
plot_results(sizes, sparse_times, dense_times)


# Example usage
data, indices = generate_sparse_matrix_data(50, 100)  
sparse_matrix = SparseMatrix(data, indices, (50, 50))

# Profile the 'add_sparse_matrices' method
profiled_stats = profile_method(sparse_matrix.add_sparse_matrices, sparse_matrix)

# If you need to profile all methods of a class, you can write a loop to profile each one.
