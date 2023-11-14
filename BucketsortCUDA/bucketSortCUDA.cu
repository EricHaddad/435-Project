#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

#define RANGE 100

// generate input types
void generateInput(int *input, int size, char* type) {
    if (strcmp(type, "sorted") == 0) {
        for (int i = 0; i < size; i++) {
            input[i] = i;
        }
    } else if (strcmp(type, "random") == 0) {
        srand(time(NULL));
        for (int i = 0; i < size; i++) {
            input[i] = rand() % RANGE;
        }
        printf("\n");
    } else if (strcmp(type, "reverse") == 0) {
        for (int i = 0; i < size; i++) {
            input[i] = size - i;
        }
        printf("\n");
    } else if (strcmp(type, "perturbed") == 0) {
        for (int i = 0; i < size; i++) {
            input[i] = i;
        }
        // Perturb 1% of the elements
        int perturbCount = size / 100;
        srand(time(NULL));
        for (int i = 0; i < perturbCount; i++) {
            // Choose random index to perturb
            int idx = rand() % size;
            // Perturb the value
            input[idx] = rand() % RANGE;
        }
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

// CUDA kernel for bucket sort
__global__ void bucket_sort(int *input, int *buckets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        atomicAdd(&buckets[input[idx]], 1);
    }
}

void bucketSort(int* input, int* output, int size, int blockSize) {
    // Allocate device memory
    int *d_input, *d_buckets;

    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_buckets, RANGE * sizeof(int));

    // Copy data to device
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");

    CALI_MARK_BEGIN("cudaMemcpy_host_to_device");
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy_host_to_device");

    cudaMemset(d_buckets, 0, RANGE * sizeof(int));
    
    // Define grid and block dimensions
    dim3 dimBlock(blockSize);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);


    // Call the kernel
    CALI_MARK_BEGIN("bucket_sort_step_region");
    bucket_sort<<<dimGrid, dimBlock>>>(d_input, d_buckets, size);
    CALI_MARK_END("bucket_sort_step_region");
    

    // Copy the buckets back to the host
    int *buckets = new int[RANGE];
    
    CALI_MARK_BEGIN("cudaMemcpy_device_to_host");
    cudaMemcpy(buckets, d_buckets, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy_device_to_host");

    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    // Generate the sorted array
    int pos = 0;
    for(int i = 0; i < RANGE; ++i) {
        std::vector<int> bucket;
        for(int j = 0; j < buckets[i]; ++j) {
            bucket.push_back(i);
        }
        // Sort each bucket
        std::sort(bucket.begin(), bucket.end());
        // Add sorted bucket to output
        for (int j = 0; j < bucket.size(); ++j) {
            output[pos++] = bucket[j];
        }
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Clean up
    delete[] buckets;
    cudaFree(d_input);
    cudaFree(d_buckets);
}

int main(int argc, char *argv[]) {
    int size, blockSize;

    blockSize = atoi(argv[1]);
    size = atoi(argv[2]);
    char *input_type = argv[3];

    int *input = new int[size];
    int *output = new int[size];

    clock_t start, end;

    // Initialize data with user input type's values
    CALI_MARK_BEGIN("data_init");
    generateInput(input, size, input_type);
    CALI_MARK_END("data_init");

    start = clock();
    bucketSort(input, output, size, blockSize);
    end = clock();

    correctness_check(output, size);

    double whole_computation_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Clean up
    delete[] input;
    delete[] output;

    // Record metadata with Adiak
    std::string algorithm = "BucketSort"; 
    std::string programmingModel = "CUDA";
    std::string datatype = "int"; 
    int sizeOfDatatype = sizeof(int); 
    int inputSize = size; 
    std::string inputType = input_type; 
    int num_threads = blockSize;
    int num_blocks = (size + num_threads - 1) / num_threads;
    int group_number = 19;
    std::string implementation_source = "Handwritten";

    printf("THREADS: %d\n", num_threads);
    printf("NUM_VALS: %d\n", inputSize);
    printf("BLOCKS: %d\n", num_blocks);
    printf("Input Type: %s\n", input_type);
    printf("Whole computation time: %.3fs\n", whole_computation_time);

    // Initialize Adiak
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", inputSize);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", 1);
    adiak::value("num_threads", num_threads);
    adiak::value("num_blocks", num_blocks);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);
    adiak::value("Whole Computation Time", whole_computation_time);

    return 0;
}