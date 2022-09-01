import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

with open("./mark/back-activity.json") as f:
    mark = json.load(f)
f.close()

with open("./recognition/result4.txt") as f:
    recognition = json.load(f)
f.close()

result = {"Ask": {},
          "Teach / Lead in": {},
          "Praise": {},
          "Criticize": {},
          "Manage": {},
          "Demo": {},
          "Individual Guide": {},
          "Observe": {},
          "Visit": {},
          "Blackboard": {},
          }

for i in range(1, 19006, 1):
    if recognition.get(str(i)) == None:
        continue
    split =  str.split(recognition.get(str(i)).get('T'), ',')
    split[0] = split[0].replace('{', '', 1)
    list_str = list(split[3])
    list_str.pop(-1)
    split[3] = ''.join(list_str)
    for action in split:
        if action.strip() in result[mark.get(str(i)).get('T')].keys():
            result[mark.get(str(i)).get('T')][action.strip()] = result[mark.get(str(i)).get('T')][action.strip()] + 1
        else:
            result[mark.get(str(i)).get('T')][action.strip()] = 1
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
        value1.append(value[i - 1] / sum(value))
    print(value1)
    ax = plt.gca()
    x_major_locator = MultipleLocator(1)
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.title(recog)
    plt.xlim(0, len(x))
    plt.bar(x, value1)
    plt.show()
    result1 = {}
    result2 = []
    for i in x:
        result1[str(i)] = key[i - 1] + ":" + str(value1[i - 1])
        if i < 7:
            result2.append(key[i - 1])
    if result1 == {}:
        continue
    if recog == 'Teach / Lead in':
        recog = "Teach and Lead in"
    with open("./result_T/" + recog + "_action.txt", "w") as f:
        f.write(json.dumps(result1))
    f.close()
    resultall[recog] = result2
with open("./result_T/" + "all_action.txt", "w") as f:
    f.write(json.dumps(resultall))