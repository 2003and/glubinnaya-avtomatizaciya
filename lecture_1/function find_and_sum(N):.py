def find_arithm_progr_sum(a,N):
    last_member = N-N%a
    S = ((a+last_member)*(N//a))/2
    return S

def find_and_sum(N):
    S3 = find_arithm_progr_sum(3,N)
    S5 = find_arithm_progr_sum(5,N)
    S15 = find_arithm_progr_sum(15,N)

    return S3 + S5 - 2*S15


print(find_and_sum(int(input())))