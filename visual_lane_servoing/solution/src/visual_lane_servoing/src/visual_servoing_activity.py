#!/usr/bin/env python
# coding: utf-8

# In[18]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def get_steer_matrix_left_lane_markings(shape, image):
    
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_left_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                    using the masked left lane markings (numpy.ndarray)
    """
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = 1
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    mask_ground = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    mask_mag = (Gmag > 250)
    
    mask_left = np.ones((image.shape[0], image.shape[1]))
    width = img.shape[1]
    mask_left[:,int(np.floor(width/2)):width + 1] = 0

    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_neg = (sobely < 0)

#     white_lower_hsv = np.array([0, 0, 150])
#     white_upper_hsv = np.array([179, 40, 254])
    yellow_lower_hsv = np.array([10, 0, 170])
    yellow_upper_hsv = np.array([35, 254, 220]) 
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
#     print("mask_yellow.shape==", mask_yellow.shape)
#     print("mask_left.shape==", mask_left.shape)
#     print("mask_mag.shape==", mask_mag.shape)
#     print("mask_sobelx_neg.shape==", mask_sobelx_neg.shape)
#     print("mask_sobely_neg.shape==", mask_sobely_neg.shape)
#     print("mask_yellow.shape==", mask_yellow.shape)
    steer_matrix_left_lane = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow


    return steer_matrix_left_lane


# In[34]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK


def get_steer_matrix_right_lane_markings(shape, image):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_right_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                     using the masked right lane markings (numpy.ndarray)
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = 1
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    mask_ground = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    mask_mag = (Gmag > 250)
    
    mask_right = np.ones((image.shape[0], image.shape[1]))
    width = img.shape[1]
    mask_right[:,0:int(np.floor(width/2))] = 0
    mask_sobelx_pos = (sobelx > 0)
    mask_sobely_neg = (sobely < 0)

    white_lower_hsv = np.array([0, 0, 150])
    white_upper_hsv = np.array([179, 40, 254])
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)

#     print("mask_right.shape==", mask_right.shape)
#     print("mask_mag.shape==", mask_mag.shape)
#     print("mask_sobelx_pos.shape==", mask_sobelx_pos.shape)
#     print("mask_sobely_neg.shape==", mask_sobely_neg.shape)
#     print("mask_white.shape==", mask_white.shape)
    steer_matrix_right_lane = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return steer_matrix_right_lane


# In[28]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def detect_lane_markings(image):
    """
        Args:
            image: An image from the robot's camera in the BGR color space (numpy.ndarray)
        Return:
            left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
            right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """
    
    h, w, _ = image.shape
    
    mask_left_edge = get_steer_matrix_left_lane_markings((h, w), image)
    mask_right_edge = get_steer_matrix_right_lane_markings((h, w), image)
    
    return (mask_left_edge, mask_right_edge)

