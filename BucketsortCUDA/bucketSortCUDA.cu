#include <stdio.h>
#include <cuda.h>
#include <caliper/cali.h>

#define RANGE 100

// CUDA kernel for bucket sort
__global__ void bucket_sort(int *input, int *buckets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        atomicAdd(&buckets[input[idx]], 1);
    }
}

// Function to print array
void correctness_check(int *array, int size) {
    CALI_MARK_BEGIN("correctness_check");
    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            printf("Array is not sorted correctly.\n");
            CALI_MARK_END("correctness_check");
            return;
        }
    }
    printf("Array is sorted correctly.\n");
    CALI_MARK_END("correctness_check");
}

void bucketSort(int* input, int* output, int size, int blockSize) {
    // Allocate device memory
    int *d_input, *d_buckets;

    cudaEvent_t start, stop;
    float gpu_time = 0.0f;

    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_buckets, RANGE * sizeof(int));

    // Copy data to device
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_buckets, 0, RANGE * sizeof(int));
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    // Define grid and block dimensions
    dim3 dimBlock(blockSize);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Call the kernel
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    bucket_sort<<<dimGrid, dimBlock>>>(d_input, d_buckets, size);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU time: %.4f milliseconds\n", gpu_time);

    // Copy the buckets back to the host
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    int *buckets = new int[RANGE];
    cudaMemcpy(buckets, d_buckets, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Generate the sorted array
    int pos = 0;
    for(int i = 0; i < RANGE; ++i) {
        for(int j = 0; j < buckets[i]; ++j) {
            output[pos++] = i;
        }
    }

    // Clean up
    delete[] buckets;
    cudaFree(d_input);
    cudaFree(d_buckets);
}

int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN("main");
    int size, blockSize;

    blockSize = atoi(argv[1]);
    size = atoi(argv[2]);

    int *input = new int[size];
    int *output = new int[size];

    CALI_MARK_BEGIN("data_init");
    // Generate random input
    srand(time(NULL));
    for(int i = 0; i < size; i++) {
        input[i] = rand() % RANGE;
    }
    CALI_MARK_END("data_init");


    bucketSort(input, output, size, blockSize);

    correctness_check(output, size);

    // Clean up
    delete[] input;
    delete[] output;

    CALI_MARK_END("main");
    return 0;
}