import numpy as np
import time

def matrix_multiplication(size):
    """
    Perform CPU-bound matrix multiplication
    """
    # Initialize two random matrices of size NxN
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    print(f"Performing matrix multiplication for size {size}x{size}...")

    # Measure time for matrix multiplication
    start_time = time.time()
    C = np.dot(A, B)  # Matrix multiplication
    end_time = time.time()

    print(f"Matrix multiplication completed for size {size}x{size} in {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    # Matrix size (adjustable for larger/smaller workloads)
    matrix_size = 3000  
    matrix_multiplication(matrix_size)
