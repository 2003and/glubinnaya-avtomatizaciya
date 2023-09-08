def seconds_conversion(sec):
    h = sec // 3600
    m = (sec % 3600) // 60
    return str(h) + " час(а/ов) и " + str(m) + " минут(а/ы)"


sec_arr = [3600, 3601, 3500, 323500]
for i in sec_arr:
    print(i,"-->",seconds_conversion(i))