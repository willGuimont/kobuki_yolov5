#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from yolov5_detector.msg import Detection
from sensor_msgs.msg import Image
import numpy as np
import cv2

class YoloV5Detector():
    def __init__(self):
        self.node_name = 'yolov5_detector'
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.frame = None

        self.image_sub = rospy.Subscriber('/camera/rgb/image_color', Image, self.image_callback)

        rospy.loginfo('Waiting for image topics...')

        # self.pub = rospy.Publisher(, Detection, queue_size=10)
        # rospy.init_node('yolov5_talker', anonymous=True)
    
    def image_callback(self, ros_data):
        cols, rows = ros_data.width, ros_data.height
        self.frame = np.frombuffer(ros_data.data, dtype=np.uint8).reshape(rows, cols, -1)

    def cleanup(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        YoloV5Detector()
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
