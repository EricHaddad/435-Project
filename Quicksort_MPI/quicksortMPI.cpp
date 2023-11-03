#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

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

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

int partition(float* arr, int start, int end) {
    float pivot = arr[end];
    int i = start - 1;

    for (int j = start; j < end; j++) {
        if (arr[j] < pivot) {
            i++;
            float temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    float temp = arr[i + 1];
    arr[i + 1] = arr[end];
    arr[end] = temp;
    return i + 1;
}

void quicksort(float* arr, int start, int end) {
    if (start < end) {
        int pivotIndex = partition(arr, start, end);
        quicksort(arr, start, pivotIndex - 1);
        quicksort(arr, pivotIndex + 1, end);
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
    const char* whole_computation = "whole_computation";
    
    int NUM_VALS = atoi(argv[1]);

    int number_of_process, rank_of_process;
    int rc = MPI_Init(&argc, &argv);
    

    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);
    
    CALI_MARK_BEGIN(whole_computation);
    double whole_start_time, whole_end_time, whole_computation_time;
    whole_start_time = MPI_Wtime();
    
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
 
        array_fill(data, NUM_VALS);
        
        array_print(data, NUM_VALS);
    }

    MPI_Barrier(MPI_COMM_WORLD);
 
    MPI_Bcast(&NUM_VALS, 1, MPI_FLOAT, 0,
              MPI_COMM_WORLD);
 
    chunk_size
        = (NUM_VALS % number_of_process == 0)
              ? (NUM_VALS / number_of_process)
              : NUM_VALS
                    / (number_of_process - 1);
 
    chunk = (float*)malloc(chunk_size * sizeof(float));
 
    MPI_Scatter(data, chunk_size, MPI_FLOAT, chunk,
                chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    free(data);
    data = NULL;

    own_chunk_size = (NUM_VALS
                      >= chunk_size * (rank_of_process + 1))
                         ? chunk_size
                         : (NUM_VALS
                            - chunk_size * rank_of_process);
 
    quicksort(chunk, 0, own_chunk_size - 1);

    for (int step = 1; step < number_of_process;
         step = 2 * step) {
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
    
    CALI_MARK_END(whole_computation);
    whole_end_time = MPI_Wtime();
       
    whole_computation_time = whole_end_time - whole_start_time;
    
    
    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_procs", number_of_process);
    adiak::value("array_size", NUM_VALS);
    adiak::value("program_name", "quicksortMPI");
    adiak::value("array_datatype_size", sizeof(float));
     
    if (rank_of_process == 0) {
        array_print(chunk, NUM_VALS);
        printf("Number of Processes: %d \n", number_of_process);
        printf("Number of Values: %d \n", NUM_VALS);
        printf("Whole Computation Time: %f \n", whole_computation_time);
        adiak::value("whole_computation_time", whole_computation_time);  
    }
    
    mgr.stop();
    mgr.flush();
 
    MPI_Finalize();
}