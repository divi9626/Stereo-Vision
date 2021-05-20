# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:17:55 2021

@author: divyam
"""

import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import copy

img1 = cv.imread("im0.png")
img2 = cv.imread("im1.png")
img1 = cv.resize(img1, (600,400))
img2 = cv.resize(img2, (600,400))

#cv.imshow("image",cap1)
#cv.waitKey(0)
#cv.destroyAllWindows

############### feature matching ###############
orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matching
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

size = len(matches)
print(size)
match = matches[:int(size/10)]

# Initialize lists
list_kp1 = []
list_kp2 = []

for mat in match:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append((x1, y1))
    list_kp2.append((x2, y2))

#matching_result = cv.drawMatches(img1, kp1, img2, kp2, match, None, flags=2)
#cv.imshow("matching results", matching_result)
#cv.waitKey(0)
#cv.destroyAllWindows
##################################################

########### getting feature index ################
def get_8_rand():        # get 8 random index to fit in Af = 0
    key_X1 = []
    
    rand_ind_list = []
    for i in range(8):
#        if i not in rand_ind_list:
        rand_ind_list.append(random.randint(0,int(size/10)-1))
    
    for ind in rand_ind_list:
        key_X1.append(ind)
    return key_X1

def get_index(key_x):
    index1 = []
    index2 = []
    for i in key_x:
        index1.append((int(list_kp1[i][0]),int(list_kp1[i][1])))
        index2.append((int(list_kp2[i][0]),int(list_kp2[i][1])))
    return index1, index2
###################################################
    
########## 8 point algorithms ##################
 
 # -----------------A_matrix = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
#A_matrix = []
#key_x = get_8_rand()
#for iter in key_x:
#    A_matrix.append([list_kp2[iter][0]*list_kp1[iter][0],list_kp2[iter][0]*list_kp1[iter][1],list_kp2[iter][0],list_kp2[iter][1]*list_kp1[iter][0],list_kp2[iter][1]*list_kp1[iter][1],list_kp2[iter][1],list_kp1[iter][0],list_kp1[iter][1],1])
#U,sig,V_T = np.linalg.svd(A_matrix)
#K_H = V_T[-1,:]/V_T[-1,-1]
#F = K_H.reshape(3,3)


################## RANSAC ###############
initial_F = 10000
F_mat = None
for i in range(2000):
    A_matrix = []
    key_x = get_8_rand()
    for iter in key_x:                                     # making A matrix for Af = 0
        A_matrix.append([list_kp2[iter][0]*list_kp1[iter][0],list_kp2[iter][0]*list_kp1[iter][1],list_kp2[iter][0],list_kp2[iter][1]*list_kp1[iter][0],list_kp2[iter][1]*list_kp1[iter][1],list_kp2[iter][1],list_kp1[iter][0],list_kp1[iter][1],1])
    U,sig,V_T = np.linalg.svd(A_matrix)
    K_H = V_T[-1,:]/V_T[-1,-1]
    F = K_H.reshape(3,3)
    list_kp1_new = list(list_kp1[0])
    list_kp2_new = list(list_kp2[0])
    list_kp1_new.append(1)
    list_kp2_new.append(1)
    X1 = np.array(list_kp1_new)
    X2 = np.array(list_kp2_new)
    ans_F = np.matmul(np.transpose(X2),np.matmul(F,X1))   # Fundamental matrix equation xT*F*x = 0
    if ans_F < 0:
        continue
    if abs(ans_F) < initial_F:
        initial_F = abs(ans_F)
        index1, index2 = get_index(key_x)
        F_mat = F

#### Enforcing Ransac #####
U_f , sig_f, V_t_f = np.linalg.svd(F_mat)
sig_f[-1] = 0
F_mat = np.matmul(U_f,np.matmul(np.diag(sig_f),V_t_f))
###########################

#F_mat = np.array([[ 2.56502805e-19,2.54139887e-17,-5.07574184e-15],[-4.80711816e-31,-2.43696301e-18,1.00000000e+00],
#                  [9.53904157e-29,-1.00000000e+00,7.53666453e-17]])      ## seeded
print(initial_F)
print(index1)
index1 = np.array(index1)
index2 = np.array(index2)
    
##################  Esential Matrix #####################
## dataset 1    
K1 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
K2 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])

## dataset 2    
#K1 = np.array([[4396.869, 0, 1353.072], [0, 4396.869, 989.702], [0, 0, 1]])
#K2 = np.array([[4396.869, 0, 1538.86], [0, 4396.869, 989.702], [0, 0, 1]])

## dataset 3 
#K1 = np.array([[5806.559, 0, 1429.219], [0, 5806.559 ,  993.403], [0, 0, 1]])
#K2 = np.array([[5806.559, 0, 1538.86], [0, 5806.559,  993.403], [0, 0, 1]])

E = np.matmul(np.transpose(K2),np.matmul(F_mat,K1))


############ Estimating pose using Esential Matrix ##########
def get_pose(E, K):               ## E is essential matrix and K is intrinsic parameter
    print("--Getting rotational and translational matrices")
    U, S, VT = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = np.matmul(np.matmul(U, np.linalg.inv(W)), VT)
    T = np.matmul(np.matmul(np.matmul(U, W), S), U.T)
    print("Rotation\n", R)
    print("Translation\n", T)
    return R, T

print("Pose of Camera one ")
print("########")
R, T = get_pose(E,K1)
print("Pose of Camera two ")
print("########")
R, T = get_pose(E,K2)

############ Rectification ####################
h1, w1, c = img1.shape
h2, w2, c = img2.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(index1), np.float32(index2), F_mat, imgSize=(w1, h1))
print("H1/n ", H1)
print ("H2/n ", H2)

img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
img_rec1 = copy.deepcopy(img1_rectified)
img_rec2 = copy.deepcopy(img2_rectified)

X_1 = np.linalg.inv(H2) 
F_new = np.matmul(np.transpose(X_1),np.matmul(F_mat,np.linalg.inv(H1)))
print("Rectified F/n :",F_new)
############ fixing initial points #################

new_kp1 = []
new_kp2 = []

for i in range(len(list_kp1)):                             # transforming keypoints to new warped image
    X1 = np.array([list_kp1[i][0],list_kp1[i][1],1])
    X2 = np.array([list_kp2[i][0],list_kp2[i][1],1])
    new_pt1 = np.dot(H1,np.transpose(X1))
    new_pt2 = np.dot(H2,np.transpose(X2))
    new_kp1.append((int(new_pt1[0]/new_pt1[2]),int(new_pt1[1]/new_pt1[2])))
    new_kp2.append((int(new_pt2[0]/new_pt2[2]),int(new_pt2[1]/new_pt2[2])))

count = 0         
for mat in match:                                          # setting up new keypoints according to feature transform

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    kp1[img1_idx].pt = new_kp1[count]
    kp2[img2_idx].pt = new_kp2[count]

    count += 1
    

#################### Epipolar lines ######################
def drawlines(img1,img2,lines,pts1,pts2):                             # for drawing epipolar lines
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c, h = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(100,200,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), (0,0,255),2)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

new_kp2 = np.array(new_kp2)
new_kp1 = np.array(new_kp1)
lines1 = cv.computeCorrespondEpilines(new_kp2.reshape(-1,1,2), 2,F_new)
lines2 = cv.computeCorrespondEpilines(new_kp1.reshape(-1,1,2), 2,F_new)
lines1 = lines1.reshape(-1,3)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img1_rectified,img2_rectified,lines1,new_kp1,new_kp2)
img5,img6 = drawlines(img2_rectified,img1_rectified,lines2,new_kp2,new_kp1)
cv.imshow("image", img5)
cv.imshow("image2", img6)
cv.waitKey(0)
cv.destroyAllWindows
        
################ correspondance (SSD) ########################
window = (7,7)


d = 50
img_gray1 = cv.cvtColor(img_rec1, cv.COLOR_BGR2GRAY)
img_gray2 = cv.cvtColor(img_rec2, cv.COLOR_BGR2GRAY)

def check_diff(img1,img2,x,y,count):
    sum = 0
    for i in range(window[0]):
        for j in range(window[1]):
            diff = (img1[x+i][y+j] - img2[x+i][y+j+count])**2
            sum = sum + diff
#    min_diff = min(diff_list)
#    ind = diff_list.index(min_diff)
#    index = (int(str(ind)[0]),ind%10)
    return sum
#
def diff_disp(img1,img2,x,y,count):
    
    winblock1 = img1[x:x + window[0],y:x + window[1]]
    winblock2 = img2[x:x + window[0],y + count:y + count + window[1]]
    diff = (winblock2.sum() - winblock1.sum())
    #print(np.sum(winblock1))
    return diff

h, w = img_gray1.shape
disparity_image = np.zeros(img_gray1.shape)
h = h - 10
w = w - 50  
t = math.ceil(window[0]/2)
for i in range(h):
    for j in range(w):
        diff_list = []
        for k in range(50):
            sum = diff_disp(img_gray1,img_gray2,i,j,k)
            diff_list.append(sum)
        disparity_image[i+t][j+t] = diff_list.index(min(diff_list))

plt.imshow(disparity_image)
colormap = plt.get_cmap('jet')
heatmap = (colormap(disparity_image) * 2**16).astype(np.uint16)[:,:,:3]
heatmap = cv.cvtColor(heatmap, cv.COLOR_RGB2BGR)

#########################################################################

###### inbuilt disparity function
#win_size = 2
#min_disp = -4
#max_disp = 9
#num_disp = max_disp - min_disp  # Needs to be divisible by 16
#stereo = cv.StereoSGBM_create(
#    minDisparity=min_disp,
#    numDisparities=num_disp,
#    blockSize=5,
#    uniquenessRatio=5,
#    speckleWindowSize=5,
#    speckleRange=5,
#    disp12MaxDiff=2,
#    P1=8 * 3 * win_size ** 2,
#    P2=32 * 3 * win_size ** 2,
#)
#disparity_SGBM = stereo.compute(img_rec1, img_rec2)
#plt.imshow(disparity_SGBM, "gray")
#plt.colorbar()
#plt.show()
########################################################################

###################### Depth Map #######################################
def normalize(matrix):
    maxvalue = matrix.max()  
    minvalue = matrix.min()  
    span = maxvalue - minvalue
    
    matrix = (matrix - minvalue)/span
    matrix = matrix*255         
    matrix = matrix.astype(np.uint8)
    
    return matrix

disp = disparity_image.astype(np.float32)
disp[disp == 0] = 0.01
depth = 1./(5*np.copy(disp))      # 1/5 because image was resized by 0.2
B = 177.288
f = 5299.313
depth = depth*B*f
depth = normalize(depth)
depthmap = cv.applyColorMap(depth, cv.COLORMAP_JET)

########################################################################

#################### Results ##########################################
plt.imshow(disparity_image)
matching_result = cv.drawMatches(img1_rectified, kp1, img2_rectified, kp2, match, None, flags=2)
cv.imshow("matching results", matching_result)
cv.imshow("disparity_image", disparity_image)
cv.imshow('heatmap', heatmap)
cv.imshow('depthmap', depthmap)
cv.imshow('depth', depth)
#cv.imshow("gray", img_gray1)
#cv.imshow("image", img5)
#cv.imshow("image2", img6)
cv.waitKey(0)
cv.destroyAllWindows

