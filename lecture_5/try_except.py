def convert_ab_int(a:str, b:str) -> tuple[int,int]:
    a, b = int(a), int(b)
    return a, b

def divide_ab(a: int, b:int) -> float:
    if 3 in [a, b]:
        raise AttributeError("I hate the number 3 >:(")
    return a/b

if __name__ == "__main__":
    while True:
        a,b = input("").split()
        try:
            a, b = convert_ab_int(a,b)
            division_score = divide_ab(a,b)
            print(division_score)
        # except Exception as e:
        #     print(f"Error: {e}")
        except (ValueError, ZeroDivisionError) as e:
            print("Please, enter two numbers!")
            print("Also, please don't try dividing by zero :(")
            print()
            continue
        except AttributeError as exc:
            print(f"Developer is a bad penis because, I quote, \"{e}\"")
        finally:
            print("this is a final block!")