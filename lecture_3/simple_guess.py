# ввод: любое целое число
# вывод: корень этого числа, если он целый
# если такого корня нет - "трудно, не могу найти"

def guess(num: int) -> int:
    ans = "Трудно"
    for i in range(max(num//2,2)):
        if i*i == num:
            ans = i
            break
    return ans

print(guess(int(input("Введите число: "))))
