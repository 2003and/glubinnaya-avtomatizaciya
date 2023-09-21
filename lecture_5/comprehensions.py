a: list[int] = [1,2,3,3,5]
b: list[int] = [0,0,1,0,1]

for i,val in enumerate(a):
    print(f"Ind:{i}; Val:{val}")


answer = [val*b[i] for i,val in enumerate(a)]
print(answer)
# # Identical to:
# answer = []
# for i in a:
#     answer.append(i)