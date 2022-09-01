import json
import os

import cv2
import torch

from model import vgg
from utils.torch_utils import select_device
from torchvision import transforms
from PIL import Image

video = cv2.VideoCapture("./data/videos/front.mp4")

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("test.mp4", fourcc, 30, (int(width), int(height)))
p = 0
l = 1
device = select_device('0')
# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

json_file = open(json_path, "r")
class_indict = json.load(json_file)

model1 = vgg(model_name="vgg16", num_classes=7).to(device)
# load model weights
weights_path = "./vgg16Net16200epoch.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model1.load_state_dict(torch.load(weights_path, map_location=device))
resultfinal = {}

while 1:
    j = 0
    ret, frame = video.read()
    coordList = []
    p = p + 1
    result = {}

    if ret:
        expression = {"face": int(len(coordList)),
                      "anger": 0,
                      "disgust": 0,
                      "fear": 0,
                      "happy": 0,
                      "normal": 0,
                      "sad": 0,
                      "surprised": 0,
                      "no expression detected": 0,
                      "no face": 0}
        if p % 2 != 0:
            with open("./runs/detect_11_f/front_" + str(p) + '_' + str(l) + '.txt', 'r') as f:
                txt = f.read().splitlines()
                for str1 in txt:
                    coord = str.split(str1, ' ')
                    coordList.append(coord)
            f.close()
            for coord in coordList:
                if int(coord[0]) != 0:
                    framecopy1 = frame.copy()
                    data_transform = transforms.Compose(
                        [transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    img = data_transform(Image.fromarray(
                        cv2.resize(framecopy1[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])], (224, 224))))
                    # expand batch dimension
                    img = torch.unsqueeze(img, dim=0)
                    # expression detect
                    model1.eval()
                    with torch.no_grad():
                        # predict class
                        output = torch.squeeze(model1(img.to(device))).cpu()
                        predict = torch.softmax(output, dim=0)
                        print(predict)
                        predict_cla = torch.argmax(predict).numpy()
                    s1, prof = class_indict[str(predict_cla)], predict[predict_cla].numpy()

                    if prof > 0.1:
                        label = f'{s1}'
                        # label = f'{s1} {prof:.3f}'
                        expression[s1] = expression[s1] + 1
                    else:
                        label = "no expression"
                        expression[label] = expression[label] + 1

                    j = j + 1
                    if j > 11:
                        break
                    if j in range(0, 10, 1):
                        result['S0'+str(j)] = label
                    else:
                        result['S'+str(j)] = label

                    cv2.rectangle(frame, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 0),
                                  lineType=cv2.LINE_AA, thickness=2)
                    cv2.putText(frame, str(j) + ' ' + label, (int(coord[0]), int(coord[1]) - 2), 0, 1, [225, 255, 255], thickness=2,
                                lineType=cv2.LINE_AA)
                    framecopy2 = frame.copy()
                else:
                    label = "no face"
                    expression[label] = expression[label] + 1
                    j = j + 1
            resultfinal[str(l)] = result
            print(json.dumps(resultfinal))
            if l == 19005:
                with open('./result_person.txt', 'w') as f:
                    f.write(json.dumps(resultfinal))
                f.close()
            cv2.imshow("result", frame)
            writer.write(frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            with open("result1.txt", "a") as f:
                f.write('第' + str(l) + '帧:' +json.dumps(expression)+"\n")
            l = l + 1
        else:
            print("in")
            cv2.imshow("result", framecopy2)
            writer.write(framecopy2)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
    else:
        break
video.release()
cv2.destroyAllWindows()