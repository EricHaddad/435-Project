#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string> 

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

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

void swap(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

void oddEvenSort(float* arr, int n) {
    int isSorted = 0;

    while (!isSorted) {
        isSorted = 1;

        for (int i = 1; i <= n - 2; i = i + 2) {
            if (arr[i] > arr[i + 1]) {
                swap(&arr[i], &arr[i + 1]);
                isSorted = 0;
            }
        }

        for (int i = 0; i <= n - 2; i = i + 2) {
            if (arr[i] > arr[i + 1]) {
                swap(&arr[i], &arr[i + 1]);
                isSorted = 0;
            }
        }
    }
}
 
float* merge(float* a1, int n1, float* a2, int n2) {
    float* result = (float*)malloc((n1 + n2) * sizeof(float));
    int i = 0;
    int j = 0;
    int k;
 
    for (k = 0; k < n1 + n2; k++) {
        if (i >= n1) {
            result[k] = a2[j];
            j++;
        } else if (j >= n2) {
            result[k] = a1[i];
            i++;
        } else if (a1[i] < a2[j]) {
            result[k] = a1[i];
            i++;
        } else {
            result[k] = a2[j];
            j++;
        }
    }
    return result;
}


// Driver Code
int main(int argc, char* argv[]) {
    CALI_CXX_MARK_FUNCTION;
    float* data = NULL;
    int chunk_size, own_chunk_size;
    float* chunk;
    MPI_Status status;
    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comm_large = "comm_large";
    const char* MPIScatter = "MPI_Scatter";
    const char* MPIBarrier = "MPI_Barrier";
    const char* correctness_check = "correctness_check";
    const char* MPIBcast = "MPI_Bcast";
    
    int NUM_VALS = atoi(argv[1]);
    char* inputType = argv[2];

    int number_of_process, rank_of_process;
    int rc = MPI_Init(&argc, &argv);
    

    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);

    cali::ConfigManager mgr;
    mgr.start();
    
    if (rank_of_process == 0) {
        chunk_size
            = (NUM_VALS % number_of_process == 0)
                  ? (NUM_VALS / number_of_process)
                  : (NUM_VALS / number_of_process
                     - 1);
 
        data = (float*)malloc(number_of_process * chunk_size
                            * sizeof(float));
        
        CALI_MARK_BEGIN(data_init);
        array_fill(data, NUM_VALS, inputType);
        CALI_MARK_END(data_init);
    }
    
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(MPIBarrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPIBarrier);
    CALI_MARK_BEGIN(comm_large);
 
    chunk_size
        = (NUM_VALS % number_of_process == 0)
              ? (NUM_VALS / number_of_process)
              : NUM_VALS
                    / (number_of_process - 1);
 
    chunk = (float*)malloc(chunk_size * sizeof(float));
    
    CALI_MARK_BEGIN(MPIScatter);
    MPI_Scatter(data, chunk_size, MPI_FLOAT, chunk,
                chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(MPIScatter);
    
    free(data);
    data = NULL;
    
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    own_chunk_size = (NUM_VALS
                      >= chunk_size * (rank_of_process + 1))
                         ? chunk_size
                         : (NUM_VALS
                            - chunk_size * rank_of_process);
 
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    oddEvenSort(chunk, own_chunk_size);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
    
    float* sorted = NULL;
    if (rank_of_process == 0) {
        sorted = (float*)malloc(NUM_VALS * sizeof(float));
    }
    
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(chunk, chunk_size, MPI_FLOAT, sorted, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    for (int step = 1; step < number_of_process; step = 2 * step) {
        if (rank_of_process % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_FLOAT,
                     rank_of_process - step, 0,
                     MPI_COMM_WORLD);
            break;
        }
 
        if (rank_of_process + step < number_of_process) {
            int received_chunk_size
                = (NUM_VALS
                   >= chunk_size
                          * (rank_of_process + 2 * step))
                      ? (chunk_size * step)
                      : (NUM_VALS
                         - chunk_size
                               * (rank_of_process + step));
            float* chunk_received;
            chunk_received = (float*)malloc(received_chunk_size * sizeof(float));
            MPI_Recv(chunk_received, received_chunk_size,
                     MPI_FLOAT, rank_of_process + step, 0,
                     MPI_COMM_WORLD, &status);
 
            data = merge(chunk, own_chunk_size, chunk_received, received_chunk_size);
 
            free(chunk);
            free(chunk_received);
            chunk = data;
            own_chunk_size = own_chunk_size + received_chunk_size;
            
        }
    }
    
    char* typeOfInput;
    std::string final;
    typeOfInput = (char*)malloc(13 * sizeof(char));
    if (rank_of_process == 0) {
        printf("Number of Processes: %d \n", number_of_process);
        printf("Number of Values: %d \n", NUM_VALS);
        printf("Input Type:%s \n", inputType);
        
        CALI_MARK_BEGIN(correctness_check);
        bool sortedCheck = true;
        for (int i = 1; i < NUM_VALS; i++) {
            if (chunk[i] < chunk[i - 1]) {
                sortedCheck = false;
            }
        }
        CALI_MARK_END(correctness_check);
        
        //array_print(chunk, own_chunk_size);
        if (sortedCheck) {
            printf("Correctly Sorted");
        } else {
            printf("Incorrectly Sorted");
        }
        

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
    }
    
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "OddEvenSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", final); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", number_of_process); // The number of processors (MPI ranks)
    adiak::value("num_threads", 1); // The number of CUDA or OpenMP threads
    adiak::value("group_num", 19); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Used lab 2 and https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/ for reference"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    mgr.stop();
    mgr.flush();
 
    MPI_Finalize();
}