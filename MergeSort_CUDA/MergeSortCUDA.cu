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

typedef struct mergeSortResult {
    cudaError_t cudaStatus;
    char* msg;
} mergeSortResult_t;

__global__ void merge(int* arr, int* aux, unsigned int blockSize, const unsigned int last)
{
    int x = threadIdx.x;
    int start = blockSize * x;
    int end = start + blockSize - 1;
    int mid = start + (blockSize / 2) - 1;
    int l = start, r = mid + 1, i = start;

    if (end > last) { end = last; }
    if (start == end || end <= mid) { return; }

    while (l <= mid && r <= end) {
        if (arr[l] <= arr[r]) {
            aux[i++] = arr[l++];
        }
        else {
            aux[i++] = arr[r++];
        }
    }

    while (l <= mid) { aux[i++] = arr[l++]; }
    while (r <= end) { aux[i++] = arr[r++]; }

    for (i = start; i <= end; i++) {
        arr[i] = aux[i];
    }
}

inline mergeSortResult_t mergeSortError(cudaError_t cudaStatus, char* msg) {
    mergeSortResult_t error;
    error.cudaStatus = cudaStatus;
    error.msg = msg;
    return error;
}

inline mergeSortResult_t mergeSortSuccess() {
    mergeSortResult_t success;
    success.cudaStatus = cudaSuccess;
    return success;
}

inline mergeSortResult_t doMergeSortWithCuda(int* arr, unsigned int count, int* dev_arr, int* dev_aux) {
    const unsigned int last = count - 1;
    const unsigned size = count * sizeof(int);
    unsigned int threadCount;
    cudaError_t cudaStatus;
    char msg[1024];

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        return mergeSortError(cudaStatus, "cudaMemcpy failed!");
    }

    for (unsigned int blockSize = 2; blockSize < 2 * count; blockSize *= 2) {
        threadCount = count / blockSize;
        if (count % blockSize > 0) { threadCount++; }

        // Launch a kernel on the GPU with one thread for each block.
        merge<<<1, threadCount>>>(dev_arr, dev_aux, blockSize, last);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            sprintf(msg, "merge kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return mergeSortError(cudaStatus, msg);
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            sprintf(msg, "cudaDeviceSynchronize returned error code %d after launching merge kernel!\n", cudaStatus);
            return mergeSortError(cudaStatus, msg);
        }
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arr, dev_arr, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        return mergeSortError(cudaStatus, "cudaMemcpy failed!");
    }

    return mergeSortSuccess();
}

cudaError_t mergeSortWithCuda(int* arr, unsigned int count)
{
    const unsigned int size = count * sizeof(int);
    int* dev_arr = 0;
    int* dev_aux = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for two vectors (main and aux array).
    cudaStatus = cudaMalloc((void**)&dev_arr, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_aux, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_arr);
        return cudaStatus;
    }

    mergeSortResult_t result = doMergeSortWithCuda(arr, count, dev_arr, dev_aux);

    if (result.cudaStatus != cudaSuccess) {
        fprintf(stderr, result.msg);
    }

    cudaFree(dev_arr);
    cudaFree(dev_aux);

    return cudaStatus;
}

void mergeSortCUDA(int* input, int* output, int size, int blockSize) {
    const unsigned int dataSize = size * sizeof(int);
    int* d_input, * d_output, * d_aux;

    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);
    cudaMalloc(&d_aux, dataSize);

    cudaMemcpy(d_input, input, dataSize, cudaMemcpyHostToDevice);

    mergeSortResult_t result = doMergeSortWithCuda(input, size, d_input, d_aux);

    if (result.cudaStatus != cudaSuccess) {
        fprintf(stderr, result.msg);
    }

    cudaMemcpy(output, d_input, dataSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_aux);
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
    mergeSortCUDA(input, output, size, blockSize);
    end = clock();

    print_elapsed(start, end);

    correctness_check(output, size);

    // Clean up
    delete[] input;
    delete[] output;

    // Record metadata with Adiak
    std::string algorithm = "MergeSort"; // replace with your algorithm name
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