from import_this import RACE_DATA
from time_conversion import seconds_conversion

winner: str = ""
for i in RACE_DATA.values():
    if i.get("FinishedPlace", -1) == 1:
        winner = f"Выиграл - {i.get('RacerName','лох какойто)').upper()}!!! Поздравляем!"
        break
winner += "\n"+"_"*len(winner)
print(winner)

print("\nПервые три места:\n")
for curr_racist in range(1, 4):
    racist_desc = f"\tГонщик на {'первом' if curr_racist == 1 else 'втором' if curr_racist == 2 else 'третьем'} месте:\n"
    for i in RACE_DATA.values():
        if i.get("FinishedPlace",-1) == curr_racist:
            racist_desc += f"\t\tИмя:{i.get('RacerName','лох какойто)')}\n"
            racist_desc += f"\t\tКоманда:{i.get('RacerTeam','лохи какието)')}\n"
            racist_time = seconds_conversion(i.get('FinishedTimeSeconds',0))
            racist_desc += f"\t\tВремя:{racist_time[0]}:{racist_time[1]}:{racist_time[2]}\n"
    print(racist_desc)