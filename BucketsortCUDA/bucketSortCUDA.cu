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

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Whole computation time: %.3fs\n", elapsed);
}

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

    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");

    cudaMemset(d_buckets, 0, RANGE * sizeof(int));
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    // Define grid and block dimensions
    dim3 dimBlock(blockSize);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);


    // Call the kernel
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    bucket_sort<<<dimGrid, dimBlock>>>(d_input, d_buckets, size);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
    

    // Copy the buckets back to the host
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    int *buckets = new int[RANGE];
    
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(buckets, d_buckets, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");

    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

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

    // Initialize data with user input type's values
    CALI_MARK_BEGIN("data_init");
    generateInput(input, size, input_type);
    CALI_MARK_END("data_init");

    clock_t start, end;
    start = clock();
    bucketSort(input, output, size, blockSize);
    end = clock();

    print_elapsed(start, end);

    correctness_check(output, size);

    // Clean up
    delete[] input;
    delete[] output;

    // Record metadata with Adiak
    std::string algorithm = "BucketSort"; // replace with your algorithm name
    std::string programmingModel = "CUDA";
    std::string datatype = "int"; // replace with your data type
    int sizeOfDatatype = sizeof(int); // replace with your data type size
    int inputSize = size; // replace with your input size
    std::string inputType = input_type; // replace with your input type
    int num_threads = blockSize; // replace with your number of threads
    int num_blocks = (size + num_threads - 1) / num_threads; // replace with your number of CUDA blocks
    int group_number = 1; // replace with your group number
    std::string implementation_source = "Handwritten"; // replace with your source type

    printf("THREADS: %d\n", num_threads);
    printf("NUM_VALS: %d\n", inputSize);
    printf("BLOCKS: %d\n", num_blocks);
    printf("Input Type: %s\n", input_type);

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
    adiak::value("num_threads", num_threads);
    adiak::value("num_blocks", num_blocks);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    return 0;
}