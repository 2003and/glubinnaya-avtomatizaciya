def seconds_conversion(sec):
    h = str(sec // 3600)
    if len(h)<2:
        h = "0"+h
    m = str((sec % 3600) // 60)
    if len(m)<2:
        m = "0"+m
    s = str(sec%60)
    if len(s)<2:
        s = "0"+s
    return (h,m,s)

if __name__ == "__main__":
    sec_arr = [3600, 3601, 3500, 323500]
    for i in sec_arr:
        print(i,"-->",seconds_conversion(i))