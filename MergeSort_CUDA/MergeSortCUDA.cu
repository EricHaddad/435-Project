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

__host__ __device__ void merge(int *arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = new int[n1];
    int *R = new int[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        arr[k] = (L[i] <= R[j]) ? L[i++] : R[j++];
        k++;
    }

    while (i < n1) {
        arr[k] = L[i++];
        k++;
    }

    while (j < n2) {
        arr[k] = R[j++];
        k++;
    }

    delete[] L;
    delete[] R;
}

void mergeSort(int *arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

__global__ void merge_sort(int *input, int *temp, int n, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = tid * size;
    int right = left + size - 1;
    int mid = left + (right - left) / 2;

    if (left < n && mid < n && right < n) {
        merge(temp, left, mid, right);

        // Copy the sorted block back to the original array
        for (int i = left; i <= right; ++i) {
            input[i] = temp[i];
        }
    }
}

void mergeSortGPU(int *input, int *temp, int size, int blockSize) {
    // Copy data to device
    int *d_input, *d_temp;

    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_temp, size * sizeof(int));

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(blockSize);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    for (int curr_size = 1; curr_size <= size - 1; curr_size = 2 * curr_size) {
        for (int left_start = 0; left_start < size - 1; left_start += 2 * curr_size) {
            int mid = std::min(left_start + curr_size - 1, size - 1);
            int right_end = std::min(left_start + 2 * curr_size - 1, size - 1);

            merge_sort<<<dimGrid, dimBlock>>>(d_input, d_temp, size, curr_size);
        }
    }

    cudaMemcpy(input, d_input, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_temp);
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    
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
    mergeSort(input, 0, size - 1);
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