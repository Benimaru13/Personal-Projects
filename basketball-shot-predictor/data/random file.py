def QuickSort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return QuickSort(left) + middle + QuickSort(right)

arr = [3, 6, 8, 10, 1, 2, 1]    
print(QuickSort(arr))
