# DSA Demo File for TRX-AI
# Problem 1: Bubble Sort (inefficient + bug)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n):   # ❌ inefficient + bug
            if arr[j] > arr[j+1]:  # ❌ index error
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
# Problem 2: Fibonacci (inefficient recursion)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # ❌ exponential time
# Problem 3: Linear Search (inefficient usage)
def search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return True
    return False
# Problem 4: Missing edge case
def find_max(arr):
    max_val = arr[0]   
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val
# Sample run
if __name__ == "__main__":
    arr = [5, 2, 9, 1]
    print(bubble_sort(arr))
    print(fibonacci(5))
    print(search(arr, 9))
    print(find_max(arr))