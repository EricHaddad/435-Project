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

    // Populate count array with 0s
    for (int i = 0; i <= max_element; i++) {
        count[i] = 0;
    }

    // Store the count of each element in the count array
    for (int i = 0; i < n; i++) {
        count[array[i]]++;
    }

    // Store sum of counts in count array
    for (int i = 1; i <= max_element; i++) {
        count[i] += count[i - 1];
    }

    // Put sorted values in output array
    for (int i = n - 1; i >= 0; i--) {
        output[count[array[i]] - 1] = array[i];
        count[array[i]]--;
    }

    // Populate original array with sorted array
    for (int i = 0; i < n; i++) {
        array[i] = output[i];
    }
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
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2)
    {
        if (rank == 0) printf("Usage: mpirun -np <num_procs> %s <array_size>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    array_size = atoi(argv[1]);

    if (rank == 0)
    {
        int remain = array_size % num_procs;

        CALI_MARK_BEGIN("data_init");
        // initialize array with random integers
        srand(time(NULL));
        data = (int *)malloc(sizeof(int) * array_size);
        printf("Unsorted array: ");
        for (i = 0; i < array_size; i++)
        {
            data[i] = rand() % MAX_NUM;
            printf("%d ", data[i]);
        }
        printf("\n");

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
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // scatter array to all processes
    int scatter_size = array_size / num_procs;
    int *scatter_data = (int *)malloc(sizeof(int) * scatter_size);
    MPI_Scatter(data, scatter_size, MPI_INT, scatter_data, scatter_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // sort each process's data
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    countsort(scatter_data, scatter_size);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // gather sorted arrays back to root process
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Gather(scatter_data, scatter_size, MPI_INT, data, scatter_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // end timer
    CALI_MARK_BEGIN("comm");
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    CALI_MARK_END("comm");

    if (rank == 0)
    {
        // sort all data in root process
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        countsort(data, array_size);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        printf("Sorted array: ");
        for (i = 0; i < array_size; i++)
            printf("%d ", data[i]);
        printf("\n");

        printf("Time to sort: %f seconds\n", end_time - start_time);

        CALI_MARK_BEGIN("correctness_check");
        if (isSorted(data, array_size)) {
            printf("The array is correctly sorted.\n");
        } else {
            printf("The array is not correctly sorted.\n");
        }
        CALI_MARK_END("correctness_check");
    }

    if (data) free(data);
    if (scatter_data) free(scatter_data);

    if (rank == 0) {
        adiak::init(NULL);
        adiak::launchdate();
        adiak::libraries();
        adiak::cmdline();
        adiak::clustername();

        adiak::value("Algorithm", "Count Sort");
        printf("Algorithm: Count Sort\n");

        adiak::value("ProgrammingModel", "MPI");
        printf("ProgrammingModel: MPI\n");

        adiak::value("Datatype", "int");
        printf("Datatype: int\n");

        adiak::value("SizeOfDatatype", sizeof(int));
        printf("SizeOfDatatype: %zu\n", sizeof(int));

        adiak::value("InputSize", array_size);
        printf("InputSize: %d\n", array_size);

        adiak::value("InputType", "Sorted");
        printf("InputType: Sorted\n");

        adiak::value("num_procs", num_procs);
        printf("num_procs: %d\n", num_procs);

        adiak::value("num_threads", 1);
        printf("num_threads: 1\n");

        adiak::value("num_blocks", 0);
        printf("num_blocks: 0\n");

        adiak::value("group_num", 19);
        printf("group_num: 19\n");

        adiak::value("implementation_source", "Handwritten & Online Source");
        printf("implementation_source: Handwritten, GeeksForGeeks: https://www.geeksforgeeks.org/counting-sort/\n");
    }

    MPI_Finalize();
    return 0;
}
