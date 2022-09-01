import os,cv2,time,torch,random,pytorchvideo,warnings,argparse,math
import numpy as np

warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort
import json

def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(clip, boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, i, img, color=[100,100,100], text_info="None", text="None",
                 velocity=None,thickness=1,fontsize=0.5,fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info+text, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    if i == 0:
        cv2.rectangle(img, c1, (c1[0] + 300, c1[1] + int(t_size[1]*6)), color, -1)
        cv2.putText(img, text_info+text, (c1[0], c1[1]+(i+1)*t_size[1]+1),
                    cv2.FONT_HERSHEY_TRIPLEX, fontsize, [0,0,0], 1)
    else:
        cv2.putText(img, text, (c1[0], c1[1] + (i+1) * t_size[1]+i*5),
                    cv2.FONT_HERSHEY_TRIPLEX, fontsize, [0, 0, 0], 1)
    return img

def deepsort_update(Tracker,pred,xywh,np_img):
    print(pred)
    print(xywh)
    print(np_img)
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    print(outputs)
    return outputs

def save_yolopreds_tovideo(yolo_preds,id_to_ava_labels,score,scoreAll,color_map,output_video):

    for i, (im, pred) in enumerate(zip(yolo_preds["img"], yolo_preds["pred"])):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, cls, trackid) in enumerate(pred):
                if int(cls.item()) != 0:
                    ava_label = ''
                    s = ''
                elif trackid.item() in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid.item()]
                    # ava_label = id_to_ava_labels[trackid].split(' ')[0]
                    s = "con:"+str(score[trackid.item()])
                    # print("ava_label", ava_label)
                else:
                    ava_label = 'Unknow'
                    s = 'con:Unknow'
                #text = '{} {} {}'.format(yolo_preds.names[int(cls)],ava_label,s)
                for index, value in enumerate(ava_label.split('\n')):
                    text = '{} {}'.format(int(trackid.item()), value)
                    color = color_map[int(cls)]
                    im = plot_one_box(box,index,im,color,str(int(trackid.item())), value)

            cv2.putText(im, "Class concentration:"+str(scoreAll), (30,30), cv2.FONT_HERSHEY_TRIPLEX, 1, [255, 255, 255], 1)
        output_video.write(im.astype(np.uint8))

def main(config):
    result = {}
    resultall = {}
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 1000
    model.classes = config.classes
    device = config.device
    imsize = config.imsize
    video_model = slowfast_r50_detection(True).eval().to(device)
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp2.pbtxt")
    print(ava_labelnames)
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    with open("../new/front-pos.json", "r") as f:
        load_dict = json.load(f)
    f.close()

    vide_save_path = config.output
    video=cv2.VideoCapture(config.input)
    width,height = int(video.get(3)),int(video.get(4))
    video.release()
    outputvideo = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))
    print("processing...")
    
    video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(config.input)
    a=time.time()
    m = 0
    for i in np.arange(0,video.duration,1/15):
        m = m + 1
        video_clips=video.get_clip(i, i+1/15)
        video_clips=video_clips['video']
        if video_clips is None:
            continue
        img_num=video_clips.shape[1]
        imgs=[]
        for j in range(img_num):
            imgs.append(tensor_to_numpy(video_clips[:,j,:,:]))

        print(str(m) + '/' + str(int(video.duration*15))+ ':')

        img = imgs[0]/255
        h, w, _ = img.shape
        xywh = [load_dict.get(str(m)).get("S01"),
                load_dict.get(str(m)).get("S02"),
                load_dict.get(str(m)).get("S03"),
                load_dict.get(str(m)).get("S04"),
                load_dict.get(str(m)).get("S05"),
                load_dict.get(str(m)).get("S06"),
                load_dict.get(str(m)).get("S07"),
                load_dict.get(str(m)).get("S08"),
                load_dict.get(str(m)).get("S09"),
                load_dict.get(str(m)).get("S10"),
                load_dict.get(str(m)).get("S11"),
                ]
        xyxy = []
        pred = []

        for index in range(0, 11, 1):
            x_, y_, w_, h_ = xywh[index][0], xywh[index][1], xywh[index][2], xywh[index][3]

            x1 = w * x_ - 0.5 * w * w_
            x2 = w * x_ + 0.5 * w * w_
            y1 = h * y_ - 0.5 * h * h_
            y2 = h * y_ + 0.5 * h * h_
            xyxy.append([x1, y1, x2, y2])
            # 坐标，识别实体类被， trackid
            pred.append([x1, y1, x2, y2, 0, index+1])

        pred1 = []
        pred1.append(pred)

        pred_tensor = torch.tensor(pred1)
        yolo_preds = {"pred": pred_tensor,
                      "img": imgs}

        # yolo_preds=model(imgs, size=imsize)
        # with open("result.txt", 'a') as f:
        #     f.write("\n片段第"+ str(m) + '/' + str(int(video.duration*30)) + '帧' + ':')
        # f.close()
        #
        # with open("result1.txt", 'a') as f:
        #     f.write("\n片段第" + str(m) + '/' + str(int(video.duration * 30)) + '帧' + ':\n')
        # f.close()
        # print(yolo_preds.pred[0])
        # print(len(yolo_preds.imgs))
        # yolo_preds.files=[f"img_{i*25+k}.jpg" for k in range(img_num)]

        # print(i,video_clips.shape,img_num)
        # deepsort_outputs=[]
        # for j in range(len(yolo_preds.pred)):
        # # for j in range(len(pred_tensor)):
        #     temp=deepsort_update(deepsort_tracker,yolo_preds.pred[j],yolo_preds.xywh[j][:,0:4].cpu(),yolo_preds.imgs[j])
        # # temp=deepsort_update(deepsort_tracker, pred_tensor, pred_tensor[:, 0:4].cpu(), imgs[0])
        #     # print("xy", yolo_preds.xywh[j][:, 0:4])
        #     # print("pr", yolo_preds.pred[j][:, 0:4])
        #     if len(temp)==0:
        #         temp=np.ones((0,8))
        #     deepsort_outputs.append(temp.astype(np.float32))
        # yolo_preds.pred=deepsort_outputs
        # print(yolo_preds)
        # print("videp_clip", video_clips.size())
        id_to_ava_labels={}
        id_to_ava_labels_file={}
        score = {}
        scoreAll = 0
        if yolo_preds.get("pred")[0].shape[0]:
            inputs,inp_boxes,_=ava_inference_transform(video_clips, yolo_preds.get("pred")[0][:,0:4], crop_size=imsize)
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)
            with torch.no_grad():
                slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                slowfaster_preds = slowfaster_preds.cpu()
            for tid,avalabel in zip(yolo_preds.pred[img_num//2][:,5].tolist(),np.argmax(slowfaster_preds,axis=1).tolist()):
                id_to_ava_labels[tid]=ava_labelnames[avalabel+1]
        save_yolopreds_tovideo(yolo_preds,id_to_ava_labels,coco_color_map,outputvideo)
    print("total cost: {:.3f}s, video clips length: {}s".format(time.time()-a,video.duration))
        
    outputvideo.release()
    print('saved video to:', vide_save_path)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="front-00.00.00.000-00.00.03.862-00.00.00.000-00.00.01.016.mp4", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="output2.mp4", help='folder to save result imgs, can not use input folder')
    # object detect config
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', default='0', type=int, help='filter by class: --class 0, or --class 0 2 3')
    config = parser.parse_args()
    
    print(config)
    main(config)
