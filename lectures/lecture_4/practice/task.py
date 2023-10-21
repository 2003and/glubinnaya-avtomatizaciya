from import_this import generate_race_data, RaceInfo
from time_conversion import seconds_conversion


if __name__  == "__main__":
    # generate random input data
    generated_race_data: RaceInfo = generate_race_data(5)

    # empty winner description string declaration
    winner: str = ""

    # generating winner description string contents
    for i in generated_race_data.values():
        # finding the first-placed racer
        if i.get("FinishedPlace", -1) == 1:
            winner = f"Выиграл - {i.get('RacerName','лох какойто)').upper()}!!! Поздравляем!"
            break
    print(winner) # printing winner description
    print("_"*len(winner)) # printing separator

    # Describing first three places
    print("\nПервые три места:\n")
    for curr_racist in range(1, 4):
        racist_desc: str = f"\tГонщик на {'первом' if curr_racist == 1 else 'втором' if curr_racist == 2 else 'третьем'} месте:\n"
        # finding the [curr_racist]-placed racer
        for i in generated_race_data.values():
            if i.get("FinishedPlace",-1) == curr_racist:
                racist_desc       += f"\t\tИмя:{i.get('RacerName','лох какойто)')}\n"                  # adding racer name to desc
                racist_desc       += f"\t\tКоманда:{i.get('RacerTeam','лохи какието)')}\n"             # adding team name to desc
                racist_time: tuple = seconds_conversion(i.get('FinishedTimeSeconds',0))                # processing time
                racist_desc       += f"\t\tВремя:{racist_time[0]}:{racist_time[1]}:{racist_time[2]}\n" # adding timestamp to desc
        print(racist_desc) # printing racer description