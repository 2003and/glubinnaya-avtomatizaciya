def decide_winner(player1, player2):
    if (player1 == "бумага" and player2 == "камень") or \
       (player1 == "камень" and player2 == "ножницы") or \
       (player1 == "ножницы" and player2 == "бумага"):
        return "игрок 1 выиграл"
    elif player1 == player2:
        return "ничья"
    else:
        return "игрок 2 выиграл"


pl1,pl2 = input().lower().split()
print(decide_winner(pl1, pl2))