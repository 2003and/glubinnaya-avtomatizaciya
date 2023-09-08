

def sum(*nums, **kwargs):
    result = 0
    for i in nums:
        if kwargs.get("neg", False) == True:
            result -= i
        else:
            result += i
    return result


print(sum(1,2))
print(sum(5))
print(sum(6,6,6))
print(sum(1,3,3,7))

print(sum(1,2,3,neg=True))

# def discriminant(a, b, c) -> int|float:
#     D = b**2 - 4*a*c
#     return D

# def greeting_asya():
#     return "Hello, Asya!"


# def greet_asya():
#     print(greeting_asya())


# print(greeting_asya().lower()+" How are you today?")

# print(discriminant(1,2,3))