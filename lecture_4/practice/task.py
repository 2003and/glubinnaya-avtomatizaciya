from import_this import generate_race_data
from time_conversion import seconds_conversion


if __name__  == "__main__":
    generated_race_data = generate_race_data(5)
    # empty winner description string declaration
    winner: str = ""
    for i in generated_race_data.values():
        # finding the first-placed racer
        if i.get("FinishedPlace", -1) == 1:
            winner = f"Выиграл - {i.get('RacerName','лох какойто)').upper()}!!! Поздравляем!"
            break
    # winner += "\n"+
    print(winner)
    print("_"*len(winner))

    print("\nПервые три места:\n")
    for curr_racist in range(1, 4):
        racist_desc = f"\tГонщик на {'первом' if curr_racist == 1 else 'втором' if curr_racist == 2 else 'третьем'} месте:\n"
        for i in generated_race_data.values():
            if i.get("FinishedPlace",-1) == curr_racist:
                racist_desc += f"\t\tИмя:{i.get('RacerName','лох какойто)')}\n"
                racist_desc += f"\t\tКоманда:{i.get('RacerTeam','лохи какието)')}\n"
                racist_time = seconds_conversion(i.get('FinishedTimeSeconds',0))
                racist_desc += f"\t\tВремя:{racist_time[0]}:{racist_time[1]}:{racist_time[2]}\n"
        print(racist_desc)  