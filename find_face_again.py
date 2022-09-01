j = 0
for p in range(1, 380011, 2):
    j = j + 1
    k = 0
    coordSort = []
    coordSort_5_final = []
    coordSort_5 = []

    coordSort_5_judge = [False, False, False, False, False]
    coordSort_6_final = []
    coordSort_6 = []
    coordSort_6_judge = [False, False, False, False, False, False]
    coordList = []
    with open("./runs/detect_11/front_" + str(p) + '_' + str(j) + '.txt', 'r') as f:
        txt = f.read().splitlines()
        for str1 in txt:
            coord = str.split(str1, ' ')
            coordList.append(coord)
    f.close()
    if len(coordList) == 11:
        for coord in coordList:
            with open("./runs/detect_11_f/front_" + str(p) + '_' + str(j) + '.txt', 'a') as f:
                f.write(str(int(coord[0])) + ' ' + str(int(coord[1])) + ' ' + str(int(coord[2])) + ' ' + str(int(coord[3])) + '\n')
        f.close()
        continue

    for coord in coordList:
        if int(coord[1]) in range(300, 400, 1):
            coordSort_5.append(coord)
        else:
            coordSort_6.append(coord)

    for coord in coordSort_5:
        if not coordSort_5_final:
            coordSort_5_final.append(coord)
        else:
            k = 0
            m = 0
            for i in coordSort_5_final:
                m = m + 1
                if int(coord[0]) <= int(i[0]):
                    coordSort_5_final.insert(k, coord)
                    k = 0
                    break
                else:
                    k = k + 1
                    if m == len(coordSort_5_final):
                        coordSort_5_final.append(coord)
                        break

    for coord in coordSort_6:
        if not coordSort_6_final:
            coordSort_6_final.append(coord)
        else:
            k = 0
            m = 0
            for i in coordSort_6_final:
                m = m + 1
                if int(coord[0]) <= int(i[0]):
                    coordSort_6_final.insert(k, coord)
                    k = 0
                    break
                else:
                    k = k + 1
                    if m == len(coordSort_6_final):
                        coordSort_6_final.append(coord)
                        break
    print(coordSort_5_final)
    print(coordSort_6_final)


    if len(coordSort_5_final) < 5:
        for coord in coordSort_5_final:
            if int(coord[0]) in range(1, 101, 1):
                coordSort_5_judge[0] = True
            elif int(coord[0]) in range(300, 401, 1):
                coordSort_5_judge[1] = True
            elif int(coord[0]) in range(500, 601, 1):
                coordSort_5_judge[2] = True
            elif int(coord[0]) in range(700, 801, 1):
                coordSort_5_judge[3] = True
            elif int(coord[0]) in range(810, 1000, 1):
                coordSort_5_judge[4] = True
        print(coordSort_5_judge)
        o = 0
        for x in coordSort_5_judge:
            if x is False:
                coordSort_5_final.insert(o, [0, 0, 0, 0])
            o = o + 1

    if len(coordSort_6_final) < 6:
        for coord in coordSort_6_final:
            if int(coord[0]) in range(1, 101, 1):
                coordSort_6_judge[0] = True
            elif int(coord[0]) in range(100, 201, 1):
                coordSort_6_judge[1] = True
            elif  int(coord[0]) in range(300, 401, 1):
                coordSort_6_judge[2] = True
            elif  int(coord[0]) in range(400, 501, 1):
                coordSort_6_judge[3] = True
            elif  int(coord[0]) in range(600, 701, 1):
                coordSort_6_judge[4] = True
            elif  int(coord[0]) in range(710, 811, 1):
                coordSort_6_judge[5] = True
        print(coordSort_6_judge)
        q = 0
        for x in coordSort_6_judge:
            if x is False:
                coordSort_6_final.insert(q, [0, 0, 0, 0])
            q = q + 1

    coordSort_5_final.extend(coordSort_6_final)
    coordSort =coordSort_5_final
    print("sort", len(coordSort))
    if len(coordSort) > 11:
        with open('count大于11_again.txt', 'a') as f1:
            f1.write('txt' + str(j) + '\n')
        f1.close()
    for coord in coordSort:
        with open("./runs/detect_11_f/front_" + str(p) + '_' + str(j) + '.txt', 'a') as f:
            f.write(str(int(coord[0])) + ' ' + str(int(coord[1])) + ' ' + str(int(coord[2])) + ' ' + str(int(coord[3])) + '\n')
    f.close()