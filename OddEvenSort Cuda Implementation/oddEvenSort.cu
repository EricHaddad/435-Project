#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;
char* inputType;

float quicksort_time;
float cudaMemcpy_host_to_device_time;
float cudaMemcpy_device_to_host_time;

const char* quicksort_region = "quicksort";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

#define MAX_NUM 100

void array_fill(float* arr, int size, char* type) {
    if (strcmp(type, "sorted") == 0) {
        for (int i = 0; i < size; i++) {
            arr[i] = i;
        }
    } else if (strcmp(type, "random") == 0) {
        srand(time(NULL));
        for (int i = 0; i < size; i++) {
            arr[i] = rand() % MAX_NUM;
        }
    } else if (strcmp(type, "reverse") == 0) {
        for (int i = 0; i < size; i++) {
            arr[i] = size - i;
        }
    } else if (strcmp(type, "perturbed") == 0) {
        for (int i = 0; i < size; i++) {
            arr[i] = i;
        }
        // Perturb 1% of the elements
        int perturbCount = size / 100;
        srand(time(NULL));
        for (int i = 0; i < perturbCount; i++) {
            // Choose random index to perturb
            int idx = rand() % size;
            // Perturb the value
            arr[idx] = rand() % MAX_NUM;
        }
    }
}

__device__ void swap(float* a, float* b) 
{ 
    float t = *a; 
    *a = *b; 
    *b = t; 
} 

__global__ void oddEvenSortKernel(float *dev_values, int vals, bool odd) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx;
    if (odd) {
        idx = 2 * index + 1;
    } else {
        idx = 2 * index;
    }

    if (idx < vals - 1) {
        if (dev_values[idx] > dev_values[idx + 1]) {
            float temp = dev_values[idx];
            dev_values[idx] = dev_values[idx + 1];
            dev_values[idx + 1] = temp;
        }
    }
}

void oddEvenSort(float* values) {
    float *dev_values;
    size_t size = NUM_VALS * sizeof(float);
    
    cudaMalloc((void**) &dev_values, size);
      
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy_host_to_device");
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy_host_to_device");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */
      
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");

    for (int i = 0; i < NUM_VALS; ++i) {
        bool odd = i % 2;
        oddEvenSortKernel<<<blocks, threads>>>(dev_values, NUM_VALS, odd);
    }

    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
      
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy_device_to_host");
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy_device_to_host");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    cudaFree(dev_values);
} 

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    inputType = argv[3];
    BLOCKS = NUM_VALS / THREADS;
    
    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    cali::ConfigManager mgr;
    mgr.start();
  
    float *array = (float*) malloc( NUM_VALS * sizeof(float));
    CALI_MARK_BEGIN("data_init");
    array_fill(array, NUM_VALS, inputType);
    CALI_MARK_END("data_init");
    //array_print(array, NUM_VALS);

    clock_t start, stop;
    start = clock();
    oddEvenSort(array);
    stop = clock();
    print_elapsed(start, stop);
    
    bool sorted = true;
    CALI_MARK_BEGIN("correctness_check");
    for (int i = 1; i < NUM_VALS; i++) {
        if (array[i] < array[i - 1]) {
            sorted = false;
            break;
        }
    }
    CALI_MARK_END("correctness_check");
    
    if (sorted) {
        printf("Correctly Sorted!");
    } else {
        printf("Incorrectly Sorted.");
        //array_print(array, NUM_VALS);
    }
    
    char* typeOfInput;
    std::string final;
    typeOfInput = (char*)malloc(13 * sizeof(char));
    if (strcmp(inputType, "sorted") == 0) {
        std::string s = "Sorted";
        strcpy(typeOfInput, s.c_str()); 
    } else if (strcmp(inputType, "random") == 0) {
        std::string s = "Random";
        strcpy(typeOfInput, s.c_str()); 
    } else if (strcmp(inputType, "reverse") == 0) {
        std::string s = "ReverseSorted";
        strcpy(typeOfInput, s.c_str()); 
    } else if (strcmp(inputType, "perturbed") == 0) {
        std::string s = "1%perturbed";
        strcpy(typeOfInput, s.c_str()); 
    }
        
    final = typeOfInput;

    delete[] array;
    
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "OddEvenSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", final); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 1); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 19); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    mgr.stop();
    mgr.flush();
}