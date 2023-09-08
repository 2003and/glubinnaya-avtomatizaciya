def find_inconsistency(arr):
    indexes: list = []
    for i in range(1, len(arr)):
        if (arr[i] != arr[i-1]+1):
            indexes.append(i)
    
    if indexes:
        return indexes
    else:
        return "Не найдено"

inp = list(map(int, input().split()))
# [1, 2, 4, 5, 6, 10, 15]
print(*find_inconsistency(inp))