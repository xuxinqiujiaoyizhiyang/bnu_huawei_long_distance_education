import cv2

video = cv2.VideoCapture("./data/videos/front.mp4")

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("test.mp4", fourcc, 30, (int(width), int(height)))
p = 1
l = 1
count = [] # 小于11个人脸的txt文件
while 1:
    ret, frame = video.read()
    if l % 2 != 0:
        print(str(l) + '/' + str(38010))
        coordSort = []
        coordSort_5 = []
        coordSort_6 = []
        coordList = []
        with open("./labels/front_" + str(l) + '.txt', 'r') as f:
            txt = f.read().splitlines()
            for str1 in txt:
                coord = str.split(str1, ' ')[1: 5]
                coordList.append(coord)
        f.close()

        for coord in coordList:
            if int(coord[1]) > 200:
                if not coordSort:
                    coordSort.append(coord)
                else:
                    k = 0
                    m = 0
                    for i in coordSort:
                        m = m + 1
                        if int(coord[1]) >= int(i[1]):
                            coordSort.insert(k, coord)
                            k = 0
                            break
                        else:
                            k = k + 1
                            if m == len(coordSort):
                                coordSort.append(coord)
                                break
            else:
                continue
        if len(coordSort) < 11:
            with open('count小于11.txt', 'a') as f1:
                f1.write('txt' + str(l) + '\n')
            f1.close()

        if len(coordSort) > 11:
            with open('count大于11.txt', 'a') as f2:
                f2.write('txt' + str(l) + '\n')
            f2.close()

        print(len(coordSort))
        print(len(coordList))

        for coord in coordSort[0: 5]:
            if not coordSort_5:
                coordSort_5.append(coord)
            else:
                k = 0
                m = 0
                for i in coordSort_5:
                    m = m + 1
                    if int(coord[0]) <= int(i[0]):
                        coordSort_5.insert(k, coord)
                        k = 0
                        break
                    else:
                        k = k + 1
                        if m == len(coordSort_5):
                            coordSort_5.append(coord)
                            break

        for coord in coordSort[5: 11]:
            if not coordSort_6:
                coordSort_6.append(coord)
            else:
                k = 0
                m = 0
                for i in coordSort_6:
                    m = m + 1
                    if int(coord[0]) <= int(i[0]):
                        coordSort_6.insert(k, coord)
                        k = 0
                        break
                    else:
                        k = k + 1
                        if m == len(coordSort_6):
                            coordSort_6.append(coord)
                            break
        print(coordSort_5)
        print(coordSort_6)
        coordSort_5.extend(coordSort_6)
        coordSort =coordSort_5
        print("sort", coordSort)
        j = 0
    if ret:
        if l % 2 != 0:
            for coord in coordSort:
                j = j + 1
                cv2.rectangle(frame, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 0),
                              lineType=cv2.LINE_AA, thickness=2)
                cv2.putText(frame, str(j), (int(coord[0]), int(coord[1]) - 2), 0, 1, [225, 255, 255], thickness=1,
                            lineType=cv2.LINE_AA)
                with open("./runs/detect_11/front_" + str(l) + '_' + str(p) + '.txt', 'a') as f:
                    f.write(str(int(coord[0])) + ' ' + str(int(coord[1])) + ' ' + str(int(coord[2])) + ' ' + str(
                        int(coord[3])) + '\n')
                f.close()
            p = p + 1
            framecopy = frame.copy()
            cv2.imshow("result", frame)
            writer.write(frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
        else:
            cv2.imshow("result", framecopy)
            writer.write(framecopy)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
    else:
        break
    l = l + 1
video.release()
cv2.destroyAllWindows()