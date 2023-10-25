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

For example:
- Quicksort (MPI)
- Quicksort (CUDA)
- Quicksort (Sequential)
- Merge sort (MPI)
- Merge sort (CUDA)
- Merge sort (Sequential)

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
