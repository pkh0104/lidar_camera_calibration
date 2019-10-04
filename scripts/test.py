#!/usr/bin/env python
# -*- coding: utf-8 -*-

PKG = 'lidar_camera_calibration'
import roslib; roslib.load_manifest(PKG)
import threading
import rospy
import os

KEY_LOCK = threading.Lock()

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'

def handle_keyboard():
    global KEY_LOCK, PAUSE
    key = raw_input('Press [ENTER] to pause and pick points\n')
    with KEY_LOCK: 
        print("Pause true")
        PAUSE = True

def start_keyboard_handler():
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()

if __name__ == '__main__':
    start_keyboard_handler()
    rospy.init_node('calibrate_camera_lidar', anonymous=True)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')
