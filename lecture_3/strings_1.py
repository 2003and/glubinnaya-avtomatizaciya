import string

# new_name: str = input("Type name: ")
# greet_message = "hello bob"

# # greet_message = (greet_message[:-3] +new_name)

# greet_message = \
#     greet_message.replace("bob", new_name)

# # print(greet_message)

# ===========================================

# river: str = "mmmmmmississippi"

# print(
#     "m" + river.strip("mi")
# )

# ===========================================

# words: str = "<!--das das das--!>"

# print(words.split())
# print(words.strip("<>!-").split())

# ===========================================

# numbers: str = string.digits
# word: str = "hell1234o b32ob ho312w ar3e y231ou _0123456789_"

# # for number in numbers:
# #     word = word.replace(number,'')

# new_word: str = ""
# for ch in word:
#     if ch in numbers:
#         continue
#     else:
#         new_word += ch

# word = new_word
# del new_word

# print(word)

# ===========================================

# words: str = "Hello Bob, are you a bob??? BOB!!!"
# # words = words.lower().replace("bob", "Gregory")
# _words = ""
# while True:
#     bob_index = words.lower().find("bob")
#     if bob_index == -1:
#         break
#     else:
#         words = words[:bob_index]+"Gregory"+words[bob_index+3:]


# print(words)
# # print(_words)
# # print(words.capitalize())
# # print(words.casefold())
# # print(words.upper())
# # print(words.lower())

# ===========================================

# _tuple: tuple = (
#     [1,2],
#     [1,2],
#     [1,2],
#     [1,2],
# )
# _tuple[1][1] = 5
# print(_tuple)

# ===========================================

# def my_sum(x:int|float, y:int) -> int:
#    return x+y


# def my_sum(l:list) -> int:
#     answer = 0
#     for i in l:
#         answer += i
#     return answer

def my_sum(*args, **kwargs) ->int|float:
    # answer = 0
    # for i in args:
    #     answer += i
    # return answer
    print(args)
    print(kwargs)

# print(my_sum(3,5))