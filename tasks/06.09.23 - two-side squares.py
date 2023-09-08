def generate_array_of_squares(N):
    res: list = []
    for i in range(-N,N+1):
        res.append(i**2)
    return res

print(generate_array_of_squares(4))