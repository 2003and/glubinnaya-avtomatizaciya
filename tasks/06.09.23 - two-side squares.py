def generate_array_of_squares(N):
    res: list = []
    for i in range(N,0,-1):
        res.append(-1*i**2)
    res.append(0)
    for i in range(1,N+1):
        res.append(i**2)
    return res

print(generate_array_of_squares(4))