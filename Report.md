# CSCE 435 Group project

## 1. Group members:
1. Sam Hirvilampi
2. Eric Haddad
3. Nhi Vu
4. Irving Salinas

---

## 2a. _due 10/25_ Project topic

Parallelizing quicksort and merge sort using MPI and CUDA. Comparing performance between sequential, MPI, and CUDA implementations.

Meeting/Communication Details:

-GroupMe for written communication

-Weekly meetings at Zach to work on the project and put things together.

## 2b. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

Implementing the following versions of each algorithm:
- Quicksort (MPI)
- Quicksort (CUDA)
- Quicksort (Sequential)
- Merge sort (MPI)
- Merge sort (CUDA)
- Merge sort (Sequential)
  
We will then compare performance metrics between these implementations.

Pseudocode:

---
Merge Sort (Sequential):

    if length of array is 1:
        return array
    
    leftArray = first half of array
    rightArray = second half of array

    left = MergeSort(leftArray)
    right = MergeSort(rightArray)
    result = []

    while left and right are not empty:
        if first element of left is less than first element of right:
            add first element of left to result
            remove first element from left
        else:
            add first element of right to result
            remove first element from right

    add rest of left to result
    add rest of right to result

    return result
---
---
Merge Sort (MPI):

    // Define the merge sort and merge functions
    function mergeSort(array, low, high)
        if low < high
            mid = low + (high - low) / 2
            mergeSort(array, low, mid)
            mergeSort(array, mid + 1, high)
            merge(array, low, mid, high)
    
    function merge(array, low, mid, high)
        // Merge two sorted subarrays array[low..mid] and array[mid+1..high]
        // Implementation details are omitted for brevity
    
    // Start of main program
    
    // Initialize MPI
    MPI_Init(&argc, &argv)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank)
    MPI_Comm_size(MPI_COMM_WORLD, &size)
    
    // Split the data among processes
    if rank == 0
        // Master process: distribute data to other processes
        for i = 1 to size - 1
            data_to_send = split_data(i)
            MPI_Send(data_to_send, data_to_send_size, MPI_INT, i, tag, MPI_COMM_WORLD)
    else
        // Worker process: receive data from master process
        MPI_Recv(data, data_size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status)
    
    // Start computation time
    start_time = getCurrentTime()
    
    // Perform the merge sort
    mergeSort(data, 0, data.length - 1)
    
    // End computation time
    end_time = getCurrentTime()
    computation_time = end_time - start_time
    
    // If not master process, send the sorted data back to master
    if rank != 0
        MPI_Send(data, data_size, MPI_INT, 0, tag, MPI_COMM_WORLD)
    
    // If master process, receive sorted data from all worker processes and merge
    if rank == 0
        for i = 1 to size - 1
            MPI_Recv(sorted_data, data_size, MPI_INT, i, tag, MPI_COMM_WORLD, &status)
            // Merge current sorted data into the final sorted array
            merge(final_sorted_array, 0, final_sorted_array.length - 1, sorted_data.length)
    
    // Start communication time
    start_comm_time = getCurrentTime()
    
    // Finalize MPI
    MPI_Finalize()
    
    // End communication time
    end_comm_time = getCurrentTime()
    communication_time = end_comm_time - start_comm_time
    
    // Amount of data sent
    amount_of_data_sent = sizeof(data) * data_size
---
---
Merge Sort (CUDA):

    function mergeSortGPU(array, size){
        //Start recording time
    
        //Allocate device memory and copy the input array
        device_array = allocateDeviceMemory(size)
    
        //Launch quicksort kernel
        mergeSortKernel<<<1, 1>>>(device_array...)
        synchronizeDevice()
    
        //Copy sorted array to host
        copyArrayToHost()
    
        //Stop recording time
    }
    
    function mergeSortKernel(array, left value, right value){
        if length of array is 1:
            return array
        
        leftArray = first half of array
        rightArray = second half of array
    
        left = mergeSortKernel(leftArray)
        right = mergeSortKernel(rightArray)
        result = []
    
        while left and right are not empty:
            if first element of left is less than first element of right:
                add first element of left to result
                remove first element from left
            else:
                add first element of right to result
                remove first element from right
    
        add rest of left to result
        add rest of right to result

    return result

    }
    
    function main(){
        //Call mergeSortGPU() with the array and size
        mergeSortOnGPU(array, size)
        //Output the array
    }
---
---
Quick Sort (Sequential):

    if length of array is 1:
        return array

    select a pivot element from the array

    array1 = elements less than pivot
    array2 = elements greater than pivot

    array1 = quicksort(array1)
    array2 = quicksort(array2)

    return array1 + array2
---
---
Quick Sort (MPI):

    // Define the quick sort and partition functions
    function quickSort(array, low, high)
        if (low < high)
            pi = partition(array, low, high)
            quickSort(array, low, pi - 1)
            quickSort(array, pi + 1, high)
        
    function partition(array, low, high)
        pivot = array[high]
        i = (low - 1)
        for j = low to high - 1
            if array[j] <= pivot
                i++
                swap array[i] with array[j]
        swap array[i + 1] with array[high]
        return i + 1
    
    // Start of main program
    
    // Initialize MPI
    MPI_Init(&argc, &argv)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank)
    MPI_Comm_size(MPI_COMM_WORLD, &size)
    
    // Split the data among processes
    if rank == 0
        // Master process: distribute data to other processes
        for i = 1 to size - 1
            data_to_send = split_data(i)
            MPI_Send(data_to_send, data_to_send_size, MPI_INT, i, tag, MPI_COMM_WORLD)
    else
        // Worker process: receive data from master process
        MPI_Recv(data, data_size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status)
    
    // Start computation time
    start_time = getCurrentTime()
    
    // Perform the quick sort
    quickSort(data, 0, data.length - 1)
    
    // End computation time
    end_time = getCurrentTime()
    computation_time = end_time - start_time
    
    // If not master process, send the sorted data back to master
    if rank != 0
        MPI_Send(data, data_size, MPI_INT, 0, tag, MPI_COMM_WORLD)
    
    // If master process, receive sorted data from all worker processes
    if rank == 0
        for i = 1 to size - 1
            MPI_Recv(sorted_data, data_size, MPI_INT, i, tag, MPI_COMM_WORLD, &status)
            merge sorted_data into final_sorted_array
    
    // Start communication time
    start_comm_time = getCurrentTime()
    
    // Finalize MPI
    MPI_Finalize()
    
    // End communication time
    end_comm_time = getCurrentTime()

    communication_time = end_comm_time - start_comm_time
    
    // Amount of data sent
    amount_of_data_sent = sizeof(data) * data_size
---
---
Quick Sort (CUDA):

    function quicksortGPU(array, size){
        //Start recording time
    
        //Allocate device memory and copy the input array
        device_array = allocateDeviceMemory(size)
    
        //Launch quicksort kernel
        quicksortKernel<<<1, 1>>>(device_array...)
        synchronizeDevice()
    
        //Copy sorted array to host
        copyArrayToHost()
    
        //Stop recording time
    }
    
    function quicksortKernel(array, left value, right value){
        pivot = array[(left + right) / 2]
        i = left
        j = right
        
        //Partition array into segments
        while i <= j
            while array[i] < pivot
                i++
            while array[j] > pivot
                j--
    
            if i <= j
                swap(array[i], array[j])
                i++
                j--
    
        //Recursively sort the segments
            if left < j
                quicksortKernel<<<1, 1>>>(array, left, j)
            if i < right
                quicksortKernel<<<1, 1>>>(array, i, right)
    }
    
    function main(){
        //Call quicksortGPU() with the array and size
        quicksortOnGPU(array, size)
        //Output the array
    }
---

## 2c. Evaluation plan - what and how will you measure and compare
- Input sizes: 20
- Input types: integer
- Strong scaling (same problem size, increase number of processors/nodes): we will increase the number of processors for the same array size
- Weak scaling (increase problem size, increase number of processors): We will increase both the array size and the number of processors
- Number of threads in a block on the GPU: We will test to run 2,4,8,16 threads in CUDA

## 3. Project Implementation

# Bucketsort MPI
---
    In the parallel version of this algorithm implemented with MPI, the process is divided among multiple processes. Here's how it works:

    Data Initialization: The root process (rank 0) initializes an array with random integers and adds padding if the array cannot be evenly divided among the available processes.

    Data Distribution: The root process scatters the array to all processes using MPI's scatter function. Each process receives a chunk of the array.

    Local Sorting: Each process sorts its local chunk of data using the bucket sort algorithm. The maximum value in each local chunk is used to determine the number of buckets. Each element is placed into a bucket, and then each bucket is sorted.

    Data Gathering: The sorted data from each process is gathered back in the root process using MPI's gather function.

    Final Sorting: The root process then sorts the gathered data using the bucket sort algorithm again to ensure complete sorting.

    Correctness Check: A function checks if the final array is sorted correctly by verifying that each element is less than or equal to the one that follows it.

    Throughout the algorithm, Caliper is used to time the computation and communication parts separately. The "comm" regions cover the MPI communication functions (scatter and gather), and the "comp" regions cover the computation function (bucket sort). This allows detailed performance measurements to be taken.
---


