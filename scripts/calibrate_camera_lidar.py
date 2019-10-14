#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.2.1
Date    : Jan 20, 2019

Description:
Script to find the transformation between the Camera and the LiDAR

Example Usage:
1. To perform calibration using the GUI to pick correspondences:

    $ rosrun lidar_camera_calibration calibrate_camera_lidar.py --calibrate

    The point correspondences will be save as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/img_corners.npy
    - PKG_PATH/calibration_data/lidar_camera_calibration/pcl_corners.npy

    The calibrate extrinsic are saved as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/extrinsics.npz
    --> 'euler' : euler angles (3, )
    --> 'R'     : rotation matrix (3, 3)
    --> 'T'     : translation offsets (3, )

2. To display the LiDAR points projected on to the camera plane:

    $ roslaunch lidar_camera_calibration display_camera_lidar_calibration.launch

Notes:
Make sure this file has executable permissions:
$ chmod +x calibrate_camera_lidar.py

References: 
http://wiki.ros.org/message_filters
http://wiki.ros.org/cv_bridge/Tutorials/
http://docs.ros.org/api/image_geometry/html/python/
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing

# External modules
import cv2
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

# ROS modules
PKG = 'lidar_camera_calibration'
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import tf2_ros
import ros_numpy
import image_geometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_matrix
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import yaml

# Global variables
PAUSE = False
FIRST_TIME = True
KEY_LOCK = threading.Lock()
TF_BUFFER = None
TF_LISTENER = None
CV_BRIDGE = CvBridge()
CAMERA_MODEL = image_geometry.PinholeCameraModel()

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'

'''
Keyboard handler thread
Inputs: None
Outputs: None
'''
def handle_keyboard():
    global KEY_LOCK, PAUSE
    key = raw_input('Press [ENTER] to pause and pick points\n')
    with KEY_LOCK: 
        print("Pause true")
        PAUSE = True


'''
Start the keyboard handler thread
Inputs: None
Outputs: None
'''
def start_keyboard_handler():
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()


'''
Save the point correspondences and image data
Points data will be appended if file already exists

Inputs:
    data - [numpy array] - points or opencv image
    filename - [str] - filename to save
    folder - [str] - folder to save at
    is_image - [bool] - to specify whether points or image data

Outputs: None
'''
def save_data(data, filename, folder, is_image=False):
    # Empty data
    if not len(data): return

    # Handle filename
    filename = os.path.join(PKG_PATH, os.path.join(folder, filename))
    
    # Create folder
    try:
        os.makedirs(os.path.join(PKG_PATH, folder))
    except OSError:
        if not os.path.isdir(os.path.join(PKG_PATH, folder)): raise

    # Save image
    if is_image:
        cv2.imwrite(filename, data)
        return

    # Save points data
    if os.path.isfile(filename):
        rospy.logwarn('Updating file: %s' % filename)
        data = np.vstack((np.load(filename), data))
    np.save(filename, data)


'''
Runs the image point selection GUI process

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    now - [int] - ROS bag time in seconds
    rectify - [bool] - to specify whether to rectify image or not

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/img_corners.npy
'''
def extract_points_2D(img_msg, now, rectify=False):
    print("extract_points_2D")
    # Log PID
    rospy.loginfo('2D Picker PID: [%d]' % os.getpid())

    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e: 
        rospy.logerr(e)
        return

    # Rectify image
    if rectify: CAMERA_MODEL.rectifyImage(img, img)
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select 2D Image Points - %d' % now.secs)
    ax.set_axis_off()
    ax.imshow(disp)

    # Pick points
    picked, corners = [], []
    def onclick(event):
        x = event.xdata
        y = event.ydata
        if (x is None) or (y is None): return

        # Display the picked point
        picked.append((x, y))
        corners.append((x, y))
        rospy.loginfo('IMG: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]
        print(len(corners))

    def onpress(event):
        if event.key == "escape":
            if len(corners) > 0:
                del corners[-1]
                del picked[-1]
                if len(corners) > 0:
                    picked.insert(0, corners[-1])
                    del ax.lines[-1]
                    ax.figure.canvas.draw_idle()
            print("The number of picked corners = ", corners)

    # Display GUI
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onpress)
    plt.show()

    # Save corner points and image
    rect = '_rect' if rectify else ''
    if len(corners) == 5: 
        del corners[-1] # Remove last duplicate
        save_data(corners, 'img_corners%s.npy' % (rect), CALIB_PATH)
        save_data(img, 'image_color%s-%d.jpg' % (rect, now.secs), 
        os.path.join(CALIB_PATH, 'images'), True)

'''
Runs the LiDAR point selection GUI process

Inputs:
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    now - [int] - ROS bag time in seconds

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/pcl_corners.npy
'''
def extract_points_3D(velodyne, now):
    print("extract_points_3D")
    # Log PID
    rospy.loginfo('3D Picker PID: [%d]' % os.getpid())

    # Extract points data
    points = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne)
    print(points)
    points = np.asarray(points.tolist())
    print(points.shape)

    # Select points within chessboard range
    inrange = np.where((points[:, :, 0] < -2.0) &
                       (points[:, :, 0] > -5.0) &
                      #(np.abs(points[:, :, 1]) < 5.0) &
                       (points[:, :, 1] <  1.5) &
                       (points[:, :, 1] > -1.5) &
                       (points[:, :, 2] <  1.0) &
                       (points[:, :, 2] > -2.0))
    
    points = points[inrange[0], inrange[1]]
    print(points.shape)
    if points.shape[0] > 5:
        rospy.loginfo('PCL points available: %d', points.shape[0])
    else:
        rospy.logwarn('Very few PCL points available in range')
        return

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points[:, -1] / np.max(points[:, -1]))

    # Setup matplotlib GUI
    fig = plt.figure()
   #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.set_title('Select 3D LiDAR Points - %d' % now.secs, color='white')
    ax.set_axis_off()
   #ax.set_facecolor((0, 0, 0))
    ax.set_axis_bgcolor((0, 0, 0))
    xs = points[:, 0]
    xs_flat = xs.flatten()
    ys = points[:, 1]
    ys_flat = ys.flatten()
    zs = points[:, 2]
    zs_flat = zs.flatten()
    ax.scatter(xs_flat, ys_flat, zs_flat, c=colors, picker=5)

    # Equalize display aspect ratio for all axes
    max_range = (np.array([points[:, 0].max() - points[:, 0].min(), 
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    print(points[:, 0].max(), points[:, 0].min())
    print(points[:, 1].max(), points[:, 1].min())
    print(points[:, 2].max(), points[:, 2].min())
    print(max_range, mid_x, mid_y, mid_z)

    # Pick points
    picked, corners = [], []
    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d

        # Ignore if same point selected again
        if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
            return
        
        # Display picked point
        picked.append((x[ind], y[ind], z[ind]))
        corners.append((x[ind], y[ind], z[ind]))
        rospy.loginfo('PCL: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]
        print(len(corners))

    def onpress(event):
        if event.key == "escape":
            if len(corners) > 0:
                del corners[-1]
                del picked[-1]
                if len(corners) > 0:
                    picked.insert(0, corners[-1])
                    del ax.lines[-1]
                    ax.figure.canvas.draw_idle()
            print("The number of picked corners = ", len(corners))

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('key_press_event', onpress)
    plt.show()

    # Save corner points
    if len(corners) == 5: 
        del corners[-1] # Remove last duplicate
        save_data(corners, 'pcl_corners.npy', CALIB_PATH)

'''
Calibrate the LiDAR and image points using OpenCV PnP RANSAC
Requires minimum 5 point correspondences

Inputs:
    points2D - [numpy array] - (N, 2) array of image points
    points3D - [numpy array] - (N, 3) array of 3D points

Outputs:
    Extrinsics saved in PKG_PATH/CALIB_PATH/extrinsics.npz
'''
def calibrate(points2D=None, points3D=None):
    # Load corresponding points
    folder = os.path.join(PKG_PATH, CALIB_PATH)
    img_npy_file = os.path.join(folder, 'img_corners.npy')
    pcl_npy_file = os.path.join(folder, 'pcl_corners.npy')
    if os.path.isfile(img_npy_file) is False: 
        rospy.logwarn("img corners are not exist.")
        return
    if os.path.isfile(pcl_npy_file) is False: 
        rospy.logwarn("pcl corners are not exist.")
        return
    if points2D is None: points2D = np.load(os.path.join(folder, 'img_corners.npy'))
    if points3D is None: points3D = np.load(os.path.join(folder, 'pcl_corners.npy'))
    
    print(points2D.shape[0], points3D.shape[0])
    # Check points shape
    assert(points2D.shape[0] == points3D.shape[0])
    if not (points2D.shape[0] >= 12):
        rospy.logwarn('PnP RANSAC Requires minimum 5 points.')
        rospy.logwarn('But 12 points are required for this tool.')
        return

    # Obtain camera matrix and distortion coefficients
    camera_matrix = CAMERA_MODEL.intrinsicMatrix()
   #dist_coeffs = CAMERA_MODEL.distortionCoeffs()
    dist_coeffs = np.zeros((4, 1), dtype='f8')

    # Estimate extrinsics
    success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(points3D, 
        points2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: rospy.logwarn('Optimization unsuccessful')

    # Convert rotation vector
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    euler = euler_from_matrix(rotation_matrix)

    # Save extrinsics
    np.savez(os.path.join(folder, 'extrinsics.npz'),
        euler=euler, R=rotation_matrix, T=translation_vector.T)

    # Display results
    print('Euler angles (RPY):', euler)
    print('Rotation Matrix:', rotation_matrix)
    print('Translation Offsets:', translation_vector.T)

    # LiDAR to camera -> camera to LiDAR
    quaternion = quaternion_from_euler(-euler[2], -euler[1], -euler[0], 'szyx')
    translation = [translation_vector.T[0, 2], -translation_vector.T[0, 0], translation_vector.T[0, 1]]
    
    yaml_stream = open(folder+'/'+TF_NAME+'.yaml', 'w')
    yaml_stream.write("%s: \n" % TF_NAME) 
    yaml_stream.write("    child_frame_id: %s\n" % CHILD_FRAME) 
    yaml_stream.write("    header: \n") 
    yaml_stream.write("        frame_id: %s\n" % PARENT_FRAME) 
    yaml_stream.write("    transform: \n") 
    yaml_stream.write("        translation: \n") 
    yaml_stream.write("            x: %f\n" % translation[0])
    yaml_stream.write("            y: %f\n" % translation[1])
    yaml_stream.write("            z: %f\n" % translation[2])
    yaml_stream.write("        rotation: \n") 
    yaml_stream.write("            x: %f\n" % quaternion[0])
    yaml_stream.write("            y: %f\n" % quaternion[1])
    yaml_stream.write("            z: %f\n" % quaternion[2])
    yaml_stream.write("            w: %f\n" % quaternion[3])
    yaml_stream.close()

'''
Projects the point cloud on to the image plane using the extrinsics

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs:
    Projected points published on /sensors/camera/camera_lidar topic
'''
def project_point_cloud(velodyne, img_msg, image_pub):
    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e: 
        rospy.logerr(e)
        return

    # Transform the point cloud
    try:
        transform = TF_BUFFER.lookup_transform('os1_lidar', 'front_fhd_cam', rospy.Time())
        print(transform)
        euler = euler_from_quaternion([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w], axes='sxyz')
        print(euler[0]/3.141592*180.0, euler[1]/3.141592*180.0, euler[2]/3.141592*180.0)
        t_x = -transform.transform.translation.y
        t_y = transform.transform.translation.z
        t_z = transform.transform.translation.x
        r = quaternion_from_euler(-euler[2], -euler[1], -euler[0], axes='szyx')
        transform.transform.rotation.x = r[0]
        transform.transform.rotation.y = r[1]
        transform.transform.rotation.z = r[2]
        transform.transform.rotation.w = r[3]
        transform.transform.translation.x = t_x
        transform.transform.translation.y = t_y
        transform.transform.translation.z = t_z
        velodyne = do_transform_cloud(velodyne, transform)
    except tf2_ros.LookupException:
        return

    # Extract points from message
    points3D = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne)
    points3D = np.asarray(points3D.tolist())
    print("check1 ", points3D.shape)
    
    # Filter points in front of camera
    inrange = np.where((points3D[:, 2] > 0) &
                       (points3D[:, 2] < 5) &
                       (np.abs(points3D[:, 0]) < 5) &
                       (np.abs(points3D[:, 1]) < 5))
   #inrange = np.where((points3D[:, 0] < -2.0) &
   #                   (points3D[:, 0] > -5.0) &
   #                  #(np.abs(poinD[:, :, 1]) < 5.0) &
   #                   (points3D[:, 1] <  1.5) &
   #                   (points3D[:, 1] > -1.5) &
   #                   (points3D[:, 2] <  1.0) &
   #                   (points3D[:, 2] > -2.0))
    max_intensity = np.max(points3D[:, -1])
    points3D = points3D[inrange[0]]
    print("check2 ", points3D.shape)

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('jet')
    colors = cmap(points3D[:, -1] / max_intensity) * 255

    # Project to 2D and filter points within image boundaries
    points2D = [ CAMERA_MODEL.project3dToPixel(point) for point in points3D[:, :3] ]
    points2D = np.asarray(points2D)
    print("check3 ", points2D.shape)
    print("check4 ", img.shape)
    inrange = np.where((points2D[:, 0] >= 0) &
                       (points2D[:, 1] >= 0) &
                       (points2D[:, 0] < img.shape[1]) &
                       (points2D[:, 1] < img.shape[0]))
    points2D = points2D[inrange[0]].round().astype('int')

    # Draw the projected 2D points
    for i in range(len(points2D)):
        cv2.circle(img, tuple(points2D[i]), 2, tuple(colors[i]), -1)

    # Publish the projected points image
    try:
        image_pub.publish(CV_BRIDGE.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e: 
        rospy.logerr(e)


'''
Callback function to publish project image and run calibration

Inputs:
    image - [sensor_msgs/Image] - ROS sensor image message
    camera_info - [sensor_msgs/CameraInfo] - ROS sensor camera info message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs: None
'''
def callback(image, camera_info, velodyne, image_pub=None):
    print("waiting [Enter] to start picking points ...")
    global CAMERA_MODEL, FIRST_TIME, PAUSE, TF_BUFFER, TF_LISTENER

    # Setup the pinhole camera model
    if FIRST_TIME:
        FIRST_TIME = False

        # Setup camera model
        rospy.loginfo('Setting up camera model')
        CAMERA_MODEL.fromCameraInfo(camera_info)

        # TF listener
        rospy.loginfo('Setting up static transform listener')
        TF_BUFFER = tf2_ros.Buffer()
        TF_LISTENER = tf2_ros.TransformListener(TF_BUFFER)

    # Projection/display mode
    if PROJECT_MODE:
        project_point_cloud(velodyne, image, image_pub)

    # Calibration mode
    elif PAUSE:
        print("calibrate")
        # Create GUI processes
        now = rospy.get_rostime()
        img_p = multiprocessing.Process(target=extract_points_2D, args=[image, now])
        pcl_p = multiprocessing.Process(target=extract_points_3D, args=[velodyne, now])
        img_p.start(); pcl_p.start()
        img_p.join(); pcl_p.join()

        # Calibrate for existing corresponding points
        calibrate()

        # Resume listener
        with KEY_LOCK: PAUSE = False
        start_keyboard_handler()


'''
The main ROS node which handles the topics

Inputs:
    camera_info - [str] - ROS sensor camera info topic
    image_color - [str] - ROS sensor image topic
    velodyne - [str] - ROS velodyne PCL2 topic
    camera_lidar - [str] - ROS projected points image topic

Outputs: None
'''
def listener(camera_info, image_color, velodyne_points, camera_lidar=None):
    # Start node
    rospy.init_node('calibrate_camera_lidar', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    rospy.loginfo('Projection mode: %s' % PROJECT_MODE)
    rospy.loginfo('TF name: %s' % TF_NAME)
    rospy.loginfo('CameraInfo topic: %s' % camera_info)
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('PointCloud2 topic: %s' % velodyne_points)
    rospy.loginfo('Output topic: %s' % camera_lidar)

    # Subscribe to topics
    info_sub = message_filters.Subscriber(camera_info, CameraInfo)
    image_sub = message_filters.Subscriber(image_color, Image)
    velodyne_sub = message_filters.Subscriber(velodyne_points, PointCloud2)

    # Publish output topic
    image_pub = None
    if camera_lidar: image_pub = rospy.Publisher(camera_lidar, Image, queue_size=5)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, info_sub, velodyne_sub], queue_size=5, slop=0.1)
    ats.registerCallback(callback, image_pub)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':

    # Calibration mode, rosrun
    TF_NAME = None
    CHILD_FRAME = None
    PARENT_FRAME = None
    camera_lidar = None
    PROJECT_MODE = False
    if sys.argv[1] == '--calibrate':
        print('argc=', len(sys.argv))
       #if len(sys.argv) < 5: 
       #    print('rosrun lidar_camera_calibration calibrate_camera_lidar.py --calibrate <tf_name> <child_frame_id> <parent_frame_id>')
       #    exit()
       #camera_info = '/usb_cam/camera_info'
       #image_color = '/usb_cam/image_raw'
       #velodyne_points = '/os1_cloud_node/points'
       #TF_NAME = sys.argv[2]
       #CHILD_FRAME = sys.argv[3]
       #PARENT_FRAME = sys.argv[4]
        camera_info = rospy.get_param('camera_info_topic')
        image_color = rospy.get_param('image_rect_topic')
        velodyne_points = rospy.get_param('velodyne_points_topic')
        TF_NAME = rospy.get_param('tf_name')
        CHILD_FRAME = rospy.get_param('child_frame')
        PARENT_FRAME = rospy.get_param('parent_frame')
    # Projection mode, run from launch file
    else:
        camera_info = rospy.get_param('camera_info_topic')
        image_color = rospy.get_param('image_rect_topic')
        velodyne_points = rospy.get_param('velodyne_points_topic')
        camera_lidar = rospy.get_param('camera_lidar_topic')
        PROJECT_MODE = bool(rospy.get_param('project_mode'))

    # Start keyboard handler thread
    if not PROJECT_MODE: start_keyboard_handler()

    # Start subscriber
    listener(camera_info, image_color, velodyne_points, camera_lidar)
