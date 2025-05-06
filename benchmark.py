import time
import numpy as np
import torch

VECTOR_SIZE = 1024
BATCH_SIZE = 64
NUM_ITERATIONS = 1


def benchmark_cpu_nlp():
    print("\n--- CPU NLP Benchmark ---")
    dummy_input = np.random.rand(BATCH_SIZE, VECTOR_SIZE).astype(np.float32)
    dummy_weight = np.random.rand(VECTOR_SIZE, VECTOR_SIZE).astype(np.float32)
    dummy_bias = np.random.rand(VECTOR_SIZE).astype(np.float32)

    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        output = np.matmul(dummy_input, dummy_weight) + dummy_bias
    end_time = time.time()
    duration = end_time - start_time
    print(f"CPU Time: {duration:.4f} seconds for {NUM_ITERATIONS} iterations")

def benchmark_gpu_nlp():
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping GPU benchmark.")
        return

    print("\n--- GPU NLP Benchmark ---")
    device = torch.device("cuda")
    dummy_input = torch.randn(BATCH_SIZE, VECTOR_SIZE, device=device)
    dummy_weight = torch.randn(VECTOR_SIZE, VECTOR_SIZE, device=device)
    dummy_bias = torch.randn(VECTOR_SIZE, device=device)

    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        output = torch.matmul(dummy_input, dummy_weight) + dummy_bias
    torch.cuda.synchronize()  # Ensure all GPU operations are finished
    end_time = time.time()
    duration = end_time - start_time
    print(f"GPU Time: {duration:.4f} seconds for {NUM_ITERATIONS} iterations")

def benchmark_cpu_visual():
    print("\n--- CPU Visual Processing Benchmark ---")
    image = np.random.rand(BATCH_SIZE, 256, 256, 3).astype(np.float32)
    kernel = np.random.rand(3, 3, 3, 3).astype(np.float32)

    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        output = np.zeros_like(image)
        for b in range(BATCH_SIZE):
            for c_out in range(3):
                for y in range(1, 255):
                    for x in range(1, 255):
                        for c_in in range(3):
                            for ky in range(3):
                                for kx in range(3):
                                    output[b, y, x, c_out] += image[b, y - 1 + ky, x - 1 + kx, c_in] * kernel[ky, kx, c_in, c_out]
    end_time = time.time()
    duration = end_time - start_time
    print(f"CPU Time: {duration:.4f} seconds for {NUM_ITERATIONS} iterations (Manual Convolution)")

def benchmark_gpu_visual():
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping GPU benchmark.")
        return

    print("\n--- GPU Visual Processing Benchmark ---")
    device = torch.device("cuda")
    image = torch.randn(BATCH_SIZE, 3, 256, 256, device=device)
    kernel = torch.randn(3, 3, 3, 3, device=device)

    import torch.nn.functional as F

    start_time = time.time()
    for _ in range(NUM_ITERATIONS):
        output = F.conv2d(image, kernel, padding=1)
    torch.cuda.synchronize()
    end_time = time.time()
    duration = end_time - start_time
    print(f"GPU Time: {duration:.4f} seconds for {NUM_ITERATIONS} iterations (PyTorch Conv2d)")

if __name__ == "__main__":
    print("Running benchmarks...")
    benchmark_cpu_nlp()
    benchmark_gpu_nlp()
    benchmark_cpu_visual()
    benchmark_gpu_visual()
    print("\nBenchmarks finished.")