def find_inconsistency(arr):
    indexes: list = []
    next_num: int = arr[0]
    for i in range(len(arr)-1):
        if (arr[i+1] != next_num+1):
            indexes.append(i+1)
        next_num = arr[i+1]
    if indexes:
        return indexes
    else:
        return "Не найдено"

print(*find_inconsistency([1, 2, 3, 4]))