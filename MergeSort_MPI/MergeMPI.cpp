#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <vector>
#include <algorithm>

#define MAX_NUM 100

using namespace std;

// generate input types
void generateInput(int *input, int size, char* type) {
    if (strcmp(type, "sorted") == 0) {
        for (int i = 0; i < size; i++) {
            input[i] = i;
        }
    } else if (strcmp(type, "random") == 0) {
        srand(time(NULL));
        for (int i = 0; i < size; i++) {
            input[i] = rand() % MAX_NUM;
        }
    } else if (strcmp(type, "reverse") == 0) {
        for (int i = 0; i < size; i++) {
            input[i] = size - i;
        }
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
            input[idx] = rand() % MAX_NUM;
        }
    }
}

// merge sort helper function
void merge(int *array, int left, int middle, int right) {
    int i, j, k;
    int n1 = middle - left + 1;
    int n2 = right - middle;

    // Create temporary arrays
    int L[n1], R[n2];

    // Copy data to temporary arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = array[left + i];
    for (j = 0; j < n2; j++)
        R[j] = array[middle + 1 + j];

    // Merge the temporary arrays back into array[left..right]
    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            array[k] = L[i];
            i++;
        } else {
            array[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        array[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        array[k] = R[j];
        j++;
        k++;
    }
}

// merge sort function
void mergesort(int *array, int left, int right) {
    if (left < right) {
        // Same as (left+right)/2, but avoids overflow for large left and right
        int middle = left + (right - left) / 2;

        // Sort first and second halves
        mergesort(array, left, middle);
        mergesort(array, middle + 1, right);

        // Merge the sorted halves
        merge(array, left, middle, right);
    }
}
// function to check the correctness of my code
bool isSorted(int *array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (array[i] > array[i + 1]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;

    int num_procs, rank, array_size, i;
    int *data = NULL; 
    double start_time, end_time, whole_computation_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3)
    {
        if (rank == 0) printf("Usage: mpirun -np <num_procs> %s <array_size>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    array_size = atoi(argv[1]);
    char *input_type = argv[2];

    cali::ConfigManager mgr;
    mgr.start();

    start_time = MPI_Wtime();

    if (rank == 0)
    {
        int remain = array_size % num_procs;

        CALI_MARK_BEGIN("data_init");
        // initialize array with random integers
        srand(time(NULL));
        data = (int *)malloc(sizeof(int) * array_size);

        generateInput(data, array_size, input_type);

        // add padding if array cannot be evenly divided
        if (remain != 0)
        {
            int size = array_size + num_procs - remain;
            data = (int *)realloc(data, sizeof(int) * size);
            for (i = array_size; i < size; i++)
                data[i] = 0;
        }
        CALI_MARK_END("data_init");
    }

    // start timer
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");

    CALI_MARK_BEGIN("MPI_Barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Barrier");

    // scatter array to all processes
    int scatter_size = array_size / num_procs;
    int *scatter_data = (int *)malloc(sizeof(int) * scatter_size);

    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(data, scatter_size, MPI_INT, scatter_data, scatter_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");

    // gather sorted arrays back to root process
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(scatter_data, scatter_size, MPI_INT, data, scatter_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");


    CALI_MARK_BEGIN("MPI_Barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Barrier");

    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (rank == 0)
    {
        // sort all data in root process
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        mergesort(data, 0, array_size - 1);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        CALI_MARK_BEGIN("correctness_check");
        if (isSorted(data, array_size)) {
            printf("The array is correctly sorted.\n");
        } else {
            printf("The array is not correctly sorted.\n");
        }
        CALI_MARK_END("correctness_check");

    }

    end_time = MPI_Wtime();
    whole_computation_time = end_time - start_time;

    if (data) free(data);
    if (scatter_data) free(scatter_data);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "MergeSort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", array_size);
    adiak::value("InputType", input_type);
    adiak::value("num_procs", num_procs);
    adiak::value("num_threads", 1); // The number of CUDA or OpenMP threads
    adiak::value("group_num", 19); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten");

    if(rank == 0)
    {
        printf("Algorithm: MergeSort\n");
        printf("ProgrammingModel: MPI\n");
        printf("Number of Processes: %d\n", num_procs);
        printf("Number of Values: %d\n", array_size);
        printf("Input Type: %s\n", input_type);
        printf("Whole computation time: %f seconds\n", whole_computation_time);
        adiak::value("Whole computation time", whole_computation_time);
    }


    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}