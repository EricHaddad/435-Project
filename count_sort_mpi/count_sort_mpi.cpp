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

// countsort helper function
void countsort(int *array, int n) {
    int max_element = 0;

    // Find the maximum element in the array
    for (int i = 0; i < n; i++) {
        if (array[i] > max_element) {
            max_element = array[i];
        }
    }

    // Create a count array to store the count of each element
    int *count = (int *)malloc(sizeof(int) * (max_element + 1));
    int *output = (int *)malloc(sizeof(int) * n);

    // Initialize count array with zeros
    for (int i = 0; i <= max_element; i++) {
        count[i] = 0;
    }

    // Store the count of each element in the count array
    for (int i = 0; i < n; i++) {
        count[array[i]]++;
    }

    // Modify the count array to store the sum of counts
    for (int i = 1; i <= max_element; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array by placing elements in their sorted order
    for (int i = n - 1; i >= 0; i--) {
        output[count[array[i]] - 1] = array[i];
        count[array[i]]--;
    }

    // Copy the sorted elements back to the original array
    for (int i = 0; i < n; i++) {
        array[i] = output[i];
    }

    // Free dynamically allocated memory
    free(count);
    free(output);
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
        countsort(data, array_size);
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
    adiak::value("Algorithm", "Countsort");
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
        printf("Algorithm: Countsort\n");
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