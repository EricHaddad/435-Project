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

void sorted(int *array, int size)
{
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }
}

// random sort for the algorithm
void random_sort(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() % MAX_NUM;
    }
}

// reverse sort for the algorithm
void reverse_sort(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = size - i;
    }
}

// perturb sort for the algorithm
void perturb(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }
    int perturb_count = size / 100;
    for (int i = 0; i < perturb_count; i++) {
        array[rand() % size] = rand() % size;
    }
}

// bucketsort helper function
void bucketsort(int *array, int n) {
    // Find maximum value to know number of buckets
    int max_val = array[0];
    for (int i = 1; i < n; i++) {
        if (array[i] > max_val)
            max_val = array[i];
    }
    max_val++;

    // Create an array of vectors as buckets
    vector<int> *buckets = new vector<int>[max_val];

    // Put array elements in different buckets
    for (int i = 0; i < n; i++) {
        int bi = array[i]; 
        buckets[bi].push_back(array[i]);
    }

    // Sort individual buckets
    for (int i = 0; i < max_val; i++)
        sort(buckets[i].begin(), buckets[i].end());

    // Concatenate all buckets into array
    int index = 0;
    for (int i = 0; i < max_val; i++)
        for (int j = 0; j < buckets[i].size(); j++)
            array[index++] = buckets[i][j];

    delete[] buckets;
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

    int num_procs, rank, array_size;
    int *data = NULL; 
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3)
    {
        if (rank == 0) printf("Usage: mpirun -np <num_procs> %s <array_size> <input_type>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    array_size = atoi(argv[1]);
    char* input_type = argv[2];

    if (rank == 0)
    {
        int remain = array_size % num_procs;

        CALI_MARK_BEGIN("data_init");
        // initialize array with random integers
        srand(time(NULL));
        data = (int *)malloc(sizeof(int) * array_size);

        if (strcmp(input_type, "Random") == 0) {
         random_sort(data, array_size);
        } else if (strcmp(input_type, "Sorted") == 0) {
            sorted(data, array_size);
        } else if (strcmp(input_type, "Reverse") == 0) {
            reverse_sort(data, array_size);
        } else if (strcmp(input_type, "Perturbed") == 0) {
            perturb(data, array_size);
        }

        // add padding if array cannot be evenly divided
        if (remain != 0)
        {
            int size = array_size + num_procs - remain;
            data = (int *)realloc(data, sizeof(int) * size);
            for (int i = array_size; i < size; i++)
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
    bucketsort(scatter_data, scatter_size);
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
        bucketsort(data, array_size);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

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

        adiak::value("InputType", input_type);
        printf("InputType: %s\n", input_type);

        adiak::value("num_procs", num_procs);
        printf("num_procs: %d\n", num_procs);

        adiak::value("implementation_source", "Handwritten");
        printf("implementation_source: Handwritten\n");
    }

    MPI_Finalize();
    return 0;
}