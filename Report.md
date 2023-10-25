# CSCE 435 Group project

## 1. Group members:
1. Sam Hirivlampi
2. Eric Haddad
3. Nhi Vu
4. Irving Salinas

---

## 2. _due 10/25_ Project topic

Parallelizing quicksort and merge sort using MPI and CUDA. Comparing performance between sequential, MPI, and CUDA implementations

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

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
