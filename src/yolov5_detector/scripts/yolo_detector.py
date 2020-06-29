#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from yolov5_detector.msg import Detection
from sensor_msgs.msg import Image
import numpy as np
import torch
import utils.utils
import utils.datasets

class YoloV5Detector():
    def __init__(self):
        self.node_name = 'yolov5_detector'
        rospy.init_node(self.node_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f'Using device {self.device}')

        rospy.loginfo('Loading yolov5s.pt')
        self.yolo = torch.load('src/yolov5_detector/scripts/weights/yolov5s.pt')['model'].float()
        self.yolo.to(self.device).eval()

        self.names = self.yolo.names if hasattr(self.yolo, 'names') else self.yolo.modules.names

        self.image_sub = rospy.Subscriber('/camera/rgb/image_color', Image, self.image_callback, queue_size=1)
        rospy.loginfo('Waiting for image topics...')

        self.pub = rospy.Publisher(self.node_name, Detection, queue_size=10)
    
    def image_callback(self, ros_data):
        cols, rows = ros_data.width, ros_data.height
        frame = np.frombuffer(ros_data.data, dtype=np.uint8).reshape(rows, cols, -1)
        img_size = min(frame.shape[:2])
        im0 = frame.copy()
        frame = utils.datasets.letterbox(im0, new_shape=img_size)[0]
        frame = frame[:, :, ::-1].transpose(2, 0, 1)
        frame = np.ascontiguousarray(frame)
        img = torch.from_numpy(frame).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        with torch.no_grad():
            pred = self.yolo(img)[0]
            pred = utils.utils.non_max_suppression(pred, 0.4, 0.5, agnostic=False)

        for det in pred:
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = utils.utils.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, confiance, detection_class in det:
                    xywh = (utils.utils.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    object_class = self.names[int(detection_class)]
                    
                    detection = Detection()
                    detection.x = xywh[0]
                    detection.y = xywh[1]
                    detection.w = xywh[2]
                    detection.h = xywh[3]
                    detection.confiance = confiance
                    detection.object_class = object_class

                    self.pub.publish(detection)


if __name__ == '__main__':
    YoloV5Detector()
    rospy.spin()
