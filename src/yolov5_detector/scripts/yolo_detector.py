#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from yolov5_detector.msg import Detection
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

class YoloV5Detector():
    def __init__(self):
        self.node_name = 'yolov5_detector'
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)
        self.cv_window_name = self.node_name

        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.cv_window_name, 25, 75)
        self.bridge = CvBridge()

        self.frame = None

        self.image_sub = rospy.Subscriber('/camera/rgb/image_color', Image, self.image_callback)

        rospy.Timer(rospy.Duration(0.03), self.show_image)
        rospy.loginfo('Waiting for image topics...')

        # self.pub = rospy.Publisher(, Detection, queue_size=10)
        # rospy.init_node('yolov5_talker', anonymous=True)
    
    def image_callback(self, ros_data):
        # print(ros_data)
        try:
            self.frame = self.bridge.imgmsg_to_cv2(ros_data, desired_encoding='passthrough')
        except CvBridgeError:
            ...

    def show_image(self, event):
        if self.frame is not None:
            cv2.imshow(self.cv_window_name, self.frame)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        YoloV5Detector()
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
