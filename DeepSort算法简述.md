## DeepSort算法简述

1、DeepSort算法原理部分学习路线：卡尔曼滤波算法/匈牙利算法==>Sort算法==>DeepSort算法

2、源码阅读：yolo_slowfast-master/deep_sort下为deepSort算法源码

3、DeepSort算法以目标检测为基础进行实现，项目中使用yolov5进行目标检测的任务，无需了解，项目中使用远程hub加载yolov5模型

![image-20220313010859550](C:\Users\98187\AppData\Roaming\Typora\typora-user-images\image-20220313010859550.png)

***1，2均可Google搜索学习，如只需使用该项目，简单了解原理后，更换目标检测权重和目标追踪权重即可提升效果，进而对自己的视频进行检测,代码中链接包含last.pt用于yolov5的目标检测，ckpt.t7用于deepsort的目标追踪**

