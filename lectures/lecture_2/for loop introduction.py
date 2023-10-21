# numbers: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15]

# for n in numbers:
#     if n % 3 == 0 and n % 5 == 0:
#         print(f"Число {n} делится и на 3 и на 5")
#     elif n % 5 == 0:
#         print(f"Число {n} делится на 5")
#     elif n % 3 == 0:
#         print(f"Число {n} делится на 3")
#     else:
#         print(f"Число {n} не делится ни на 3, ни на 5")


# word = input("Enter word")
# vowels = 'aeiouy'
# vowel_count = 0

# for ch in word:
#     print(ch)
#     if ch in vowels:
#         vowel_count += 1

# print(vowel_count)


n: int = int(input("N: "))

# for i in range(n+1):
#     print(i)

array: list = list(range(n))
i = 0
while True:
    if array[i] % 123 == 0:
        print(f"{array[i]} is divisible by 123")
        break
    i+=1
