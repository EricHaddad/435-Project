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

/// mergesort helper function
void merge(int *array, int left, int middle, int right) {
    int i, j, k;
    int n1 = middle - left + 1;
    int n2 = right - middle;

    // Create temporary arrays
    int L[n1], R[n2];

    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = array[left + i];
    for (j = 0; j < n2; j++)
        R[j] = array[middle + 1 + j];

    // Merge the temp arrays back into array[left..right]
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

// mergesort function
void mergesort(int *array, int left, int right) {
    if (left < right) {
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
    mergesort(scatter_data, 0, scatter_size - 1);
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
        mergesort(data, 0, array_size - 1);
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

        adiak::value("Algorithm", "Bucketsort");
        printf("Algorithm: Bucketsort\n");

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

        adiak::value("group_num", 1);
        printf("group_num: 1\n");

        adiak::value("implementation_source", "Handwritten");
        printf("implementation_source: Handwritten\n");
    }

    MPI_Finalize();
    return 0;
}