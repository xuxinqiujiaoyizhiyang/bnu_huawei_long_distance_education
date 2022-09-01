import json
from functools import reduce

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json

with open("./mark/front-activity.json") as f:
    mark = json.load(f)
f.close()

with open("./recognition/result1.txt") as f:
    recognition = json.load(f)
f.close()

result = {"Listen": {},
          "Experiment / Practice": {},
          "Take notes": {},
          "Exercises": {},
          "Operate computers / pad": {},
          "Raise hands": {},
          "Stand up": {},
          "Read": {},
          "Talk with teacher": {},
          "Feedback to teacher": {},
          "Peer discussion": {},
          "Hands-on collaboration": {},
          "Disengage": {}
          }

for i in range(1, 19006, 3):
    print(i)
    for stu in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08']:
        if stu == 'S10':
            stu1 = 'S010'
        else:
            stu1 = stu
        split =  str.split(recognition.get(str(i)).get(stu1), ',')
        split[0] = split[0].replace('{', '', 1)
        list_str = list(split[3])
        list_str.pop(-1)
        split[3] = ''.join(list_str)
        for action in split:
            if action.lstrip() in result[mark.get(str(i)).get(stu)].keys():
                result[mark.get(str(i)).get(stu)][action.lstrip()] = result[mark.get(str(i)).get(stu)][action.lstrip()] + 1
            else:
                result[mark.get(str(i)).get(stu)][action.lstrip()] = 1
print(result)

resultall = {}
for recog in result:
    key = list(result[recog].keys())
    x = range(1, len(key)+1, 1)
    result[recog] = sorted(result[recog].items(), key=lambda x: x[1], reverse=True)
    key = []
    value = []
    for i in x:
        key.append(result[recog][i-1][0])
        value.append(result[recog][i-1][1])
    value1 = []
    for i in x:
        value1.append(value[i-1] / sum(value))
    ax = plt.gca()
    x_major_locator = MultipleLocator(1)
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.title(recog)
    plt.xlim(0, len(x))
    plt.bar(x, value1)
    r = 0
    u = 0
    for v in value1:
        u = u + 1
        r = v + r
        if r > 0.9:
            break
    print(u)
    result1 = {}
    result2 = []
    for i in x:
        result1[str(i)] = key[i - 1] + ":" + str(value1[i - 1])
        if i < u + 1:
            result2.append(key[i - 1])
    if result1 == {}:
        continue
    else:
        plt.show()
    with open("./result/" + recog + "_action.txt", "w") as f:
        f.write(json.dumps(result1))
    f.close()
    resultall[recog] = result2
with open("./result/" + "all_action.txt", "w") as f:
    f.write(json.dumps(resultall))
