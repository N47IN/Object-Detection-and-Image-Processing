#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		    Cosmo Logistic (CL) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script should be used to implement Task 1A of Cosmo Logistic (CL) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          CL_1874
# Author List:		Keerthi Vasan M ,Navin Sriram, Shobhith
# Filename:		    task1a.py
# Functions:
# calculate_rectangle_area,detect_aruco,depthimagecb,colorimagecb,process_image,main
# Nodes:		    aruco_tf_process
#
# Publishing Topics  - [ /tf ]
#                   Subscribing Topics - /camera/color/image_raw,'/camera/aligned_depth_to_color/image_raw'


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
# from realsense2_camera_msgs.msg import RGBD

import time

def calculate_rectangle_area(coordinates):

    area = None
    width = None
    area = cv2.contourArea(coordinates)
    width = np.sqrt(area)

    return area, width


def detect_aruco(image, camera_matrix, dist_matrix):

    aruco_area_threshold = 1500
    cam_mat = camera_matrix

    dist_mat = dist_matrix
    # We are using 150x150 aruco marker size
    size_of_aruco_m = 0.15

    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids = []


    id = []
    # print(image)
    image_bgr = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, id, rejected) = detector.detectMarkers(
        image)
    if id is not None:
        cv2.aruco.drawDetectedMarkers(image_bgr, corners, id)
        objPoints = np.zeros((4, 1, 3), dtype=np.float32)
        objPoints[0][0] = [-size_of_aruco_m/2.0, size_of_aruco_m/2.0, 0]
        objPoints[1][0] = [size_of_aruco_m/2.0, size_of_aruco_m/2.0, 0]
        objPoints[2][0] = [size_of_aruco_m/2.0, -size_of_aruco_m/2.0, 0]
        objPoints[3][0] = [-size_of_aruco_m/2.0, -size_of_aruco_m/2.0, 0]
        for i in range(len(id)):
            area, width = calculate_rectangle_area(corners[i][0])
            # print(area)
            # if area < aruco_area_threshold:
            #     continue
            shifted_array = np.roll(corners[i][0], -2, axis=0)
            rtval, rvec_world, tvec = cv2.solvePnP(
                objPoints, shifted_array, cam_mat, dist_mat)
            rtval, rvec, tvec = cv2.solvePnP(
                objPoints, corners[i][0], cam_mat, dist_mat)
            cv2.drawFrameAxes(image_bgr, cam_mat, dist_mat,
                              rvec, tvec, size_of_aruco_m)
            centerX = (corners[i][0][0][0] + corners[i][0][2][0] +
                       corners[i][0][1][0] + corners[i][0][3][0]) / 4
            centerY = (corners[i][0][0][1] + corners[i][0][2][1] +
                       corners[i][0][1][1] + corners[i][0][3][1]) / 4
            distance = cv2.norm(tvec)
            center_aruco_list.append([centerX, centerY])
            ids.append(id[i])
            distance_from_rgb_list.append(distance)
            rotCamerMatrix1, _ = cv2.Rodrigues(rvec_world)
            eulerAngles = R.from_matrix(rotCamerMatrix1).as_euler('xyz')
            angle_aruco_list.append(eulerAngles)
            width_aruco_list.append(width)
            cv2.circle(image_bgr, (int(centerX), int(centerY)),
                       10, (255, 0, 255), 5)
    cv2.imshow("Image", image_bgr)
    cv2.waitKey(1)
    # print(center_aruco_list)
    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids


##################### CLASS DEFINITION #######################

class aruco_tf(Node):

    def __init__(self):

        super().__init__('aruco_tf_publisher')

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depthimagecb, 10)

        self.info_cam_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_intel_realsense, 10)
        ############ Constructor VARIABLES/OBJECTS ############

        # rate of time to process image (seconds)
        image_processing_rate = 0.2
        # initialise CvBridge object for image conversion
        self.bridge = CvBridge()
        # buffer time used for listening transforms
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # object as transform broadcaster to send transform wrt some frame_id
        self.br = tf2_ros.TransformBroadcaster(self)
        # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        self.cv_image = None
        self.depth_image = None 
        self.cam_mat = None
        self.dist_mat = None

        time.sleep(4)
        self.timer = self.create_timer(
            image_processing_rate, self.process_image)

    def depthimagecb(self, data):
 
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data)

        except CvBridgeError as e:
            print(e)

    def colorimagecb(self, data):

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data)
            # print(np.shape(self.cv_image))
            # self.cv_image = cv2.rotate(self.cv_image,cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)

    def camera_info_intel_realsense(self, info):
        '''
        Description: callback function taking information from the camera_info topic and storing it as a member variable for future use

        Args:
        Returns:
        '''
        # info  = CameraInfo()
        # distortion model is plumb bomb
        self.sizeCamX = info.width
        self.sizeCamY = info.height
        self.centerCamX = info.k[2]
        self.centerCamY = info.k[5]
        self.focalX = info.k[0]
        self.focalY = info.k[4]
        self.cam_mat = np.asarray(info.k).reshape(3, 3)
        # right now info.d is empty but in realsense it should be correct , also it is a 1x6 array
        self.dist_mat = np.asarray(info.d)

    def process_image(self):

        sizeCamX = self.sizeCamX
        sizeCamY = self.sizeCamY
        centerCamX = self.centerCamX
        centerCamY = self.centerCamY
        focalX = self.focalX
        focalY = self.focalY
        
        if self.cv_image is not None and self.depth_image is not None:
            center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids = detect_aruco(
                self.cv_image, self.cam_mat, self.dist_mat)
            for i in range(len(width_aruco_list)):
          
                quaternion = None
                transform_stamped = TransformStamped()
                transform_stamped.header.stamp = self.get_clock().now().to_msg()
                transform_stamped.header.frame_id = 'camera_link'
                transform_stamped.child_frame_id = f'camera_link_cv'
                transform_stamped.transform.translation.x = 0.0
                transform_stamped.transform.translation.y = 0.0
                transform_stamped.transform.translation.z = 0.0
                # this part is officially given in ros
                r = R.from_euler('xzy', [-math.pi/2, -math.pi/2, 0])
                quaternion = r.as_quat()
                transform_stamped.transform.rotation.x = quaternion[0]
                transform_stamped.transform.rotation.y = quaternion[1]
                transform_stamped.transform.rotation.z = quaternion[2]
                transform_stamped.transform.rotation.w = quaternion[3]
                self.br.sendTransform(transform_stamped)
                # int(center_aruco_list[i][0]))
                cv2.circle(self.depth_image, (int(center_aruco_list[i][1]), int(center_aruco_list[i][0])),
                           2, (255, 255, 255), 5)
                depth_value = self.depth_image[int(center_aruco_list[i][1]),
                                               int(center_aruco_list[i][0])]
                depth_value = depth_value/1000
                '''uncomment bottom to start the logger'''
                self.get_logger().info(f"{depth_value}")
                x = depth_value * \
                    (sizeCamX -
                     int(center_aruco_list[i][0]) - centerCamX) / focalX
                y = depth_value * \
                    (sizeCamY -
                     int(center_aruco_list[i][1]) - centerCamY) / focalY
                z = depth_value
                # print(f"{ids[i][0]} {angle_aruco_list[i][0]}")
                r = R.from_euler('xyz', [
                                 math.pi+angle_aruco_list[i][0], angle_aruco_list[i][1], angle_aruco_list[i][2]])
                quaternion = r.as_quat()
                transform_stamped.header.stamp = self.get_clock().now().to_msg()
                transform_stamped.header.frame_id = 'camera_link_cv'
                # transform_stamped.child_frame_id = f'cam_cv_{ids[i][0]}'
                transform_stamped.child_frame_id = f'cl_1874_cam_{ids[i][0]}'
                transform_stamped.transform.translation.x = -x
                transform_stamped.transform.translation.y = -y
                transform_stamped.transform.translation.z = z
                transform_stamped.transform.rotation.x = quaternion[0]
                transform_stamped.transform.rotation.y = quaternion[1]
                transform_stamped.transform.rotation.z = quaternion[2]
                transform_stamped.transform.rotation.w = quaternion[3]
                self.br.sendTransform(transform_stamped)

                from_frame_rel = f"cl_1874_cam_{ids[i][0]}"
                to_frame_rel = 'camera_link'
                try:
                    t = self.tf_buffer.lookup_transform(
                        to_frame_rel,
                        from_frame_rel,
                        rclpy.time.Time())
                except tf2_ros.TransformException as ex:
                    self.get_logger().info(
                        f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                    return

                transform_stamped_base = TransformStamped()
                transform_stamped_base.header.frame_id = "camera_link"
                transform_stamped_base.child_frame_id = f'cl#1874_cam_{ids[i][0]}'
                transform_stamped_base.transform.translation.x = t.transform.translation.x
                transform_stamped_base.transform.translation.y = t.transform.translation.y
                transform_stamped_base.transform.translation.z = t.transform.translation.z
                transform_stamped_base.transform.rotation.x = t.transform.rotation.x
                transform_stamped_base.transform.rotation.y = t.transform.rotation.y
                transform_stamped_base.transform.rotation.z = t.transform.rotation.z
                transform_stamped_base.transform.rotation.w = t.transform.rotation.w
                transform_stamped_base.header.stamp = self.get_clock().now().to_msg()
                self.br.sendTransform(transform_stamped_base)

                from_frame_rel = f"cl#1874_cam_{ids[i][0]}"
                to_frame_rel = 'base_link'
                try:
                    t = self.tf_buffer.lookup_transform(
                        to_frame_rel,
                        from_frame_rel,
                        rclpy.time.Time())
                except tf2_ros.TransformException as ex:
                    self.get_logger().info(
                        f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                    return

                transform_stamped_base = TransformStamped()
                transform_stamped_base.header.frame_id = "base_link"
                transform_stamped_base.child_frame_id = f'cl#1874_base_{ids[i][0]}'
                transform_stamped_base.transform.translation.x = t.transform.translation.x
                transform_stamped_base.transform.translation.y = t.transform.translation.y
                transform_stamped_base.transform.translation.z = t.transform.translation.z
                transform_stamped_base.transform.rotation.x = t.transform.rotation.x
                transform_stamped_base.transform.rotation.y = t.transform.rotation.y
                transform_stamped_base.transform.rotation.z = t.transform.rotation.z
                transform_stamped_base.transform.rotation.w = t.transform.rotation.w
                transform_stamped_base.header.stamp = self.get_clock().now().to_msg()
                self.br.sendTransform(transform_stamped_base)

def main():

    rclpy.init(args=sys.argv)

    # creating ROS node
    node = rclpy.create_node('aruco_tf_process')

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    # creating a new object for class 'aruco_tf'
    aruco_tf_class = aruco_tf()

    # spining on the object to make it alive in ROS 2 DDS
    rclpy.spin(aruco_tf_class)

    # destroy node after spin ends
    aruco_tf_class.destroy_node()

    # shutdown process
    rclpy.shutdown()


if __name__ == '__main__':

    main()
