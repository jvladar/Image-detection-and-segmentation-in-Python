import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1

f = 'Zostera'
f1= 'Mytilus'
# def resize (f):
#     for file in os.listdir(f):
#         if file != '.DS_Store':
#             f_img = f+"/"+file
#             print (file)
#             img = Image.open(f_img)
#             img = img.resize((1024,1024))
#             img.save(f_img)
# def task1():
#     resize(f)
#     resize(f1)

# 2
# def match_image(img, template):
#     methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
#         cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] 
#     result_list = []
#     w, h = template.shape[::-1]
#     match_method = cv2.TM_CCOEFF_NORMED
#     res = cv2.matchTemplate(img, template, match_method)
#     minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
#     topleft = maxloc
#     btm_right = (topleft[0] + w, topleft[1] + h)
#     cv2.rectangle(img, topleft, btm_right, 255, 2)
#     return maxval

# def congo(etalons,images,my_input,my):
#     for et in os.listdir(etalons):
#         if et != '.DS_Store':
#             e_img = etalons+"/"+et
#             # print(e_img)
#             img_teplate= cv2.imread(e_img, 0) 
#             my_input=[]
#             for image in os.listdir(images):
#                 if image != '.DS_Store':
#                     f_img = images+"/"+image
#                     img = cv2.imread(f_img, 0)
#                     result = match_image(img, img_teplate)
#                     my_input.append(result)
#             my.append(my_input)
#     averages=[]
#     for i in my :
#         avg=0
#         for o in i:
#             avg=avg+o
#         averages.append(avg/(len(my[0])))
#     print("Precision:", sum(averages)/len(averages))
#     return averages

# def getFolderNameAndEtalonFolderName(classFolderName):
#     etalonFolderName = classFolderName + 'Etalons'
#     print('Folder name:', classFolderName, '\r\nEtalon folder name:', etalonFolderName)
#     return classFolderName, etalonFolderName

# def task2(classFolderName):
#     my_input = []
#     my = []
#     folderName, etalonFolderName = getFolderNameAndEtalonFolderName(classFolderName)
#     pele = congo(etalonFolderName, folderName, my_input, my)
#     print (pele)


# # Mytilus & Zostera
# task2(f1)
# task2(f)

import cv2 as cv
def match_image(img, template):
    methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED,
        cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED] 
    result_list = []
    w, h = template.shape[::-1]
    match_method = cv.TM_CCOEFF_NORMED
    res = cv.matchTemplate(img, template, match_method)
    minval, maxval, minloc, maxloc = cv.minMaxLoc(res)
    topleft = maxloc
    btm_right = (topleft[0] + w, topleft[1] + h)
    cv.rectangle(img, topleft, btm_right, 255, 2)
    return maxval


def congo(etalons,images,my_input,my):
    for et in os.listdir(etalons):
        if et != '.DS_Store':
            e_img = etalons+"/"+et
            print(e_img)
            img_teplate= cv.imread(e_img, 0) 
            my_input=[]
            for image in os.listdir(images):
                if image != '.DS_Store':
                    f_img = images+"/"+image
                    img = cv.imread(f_img, 0)
                    result = match_image(img, img_teplate)
                    my_input.append(result)
            my.append(my_input)
    averages=[]
    for i in my :
        avg=0
        for o in i:
            avg=avg+o
        averages.append(avg/(len(my[0])))
    print(sum(averages)/len(averages))
    return averages

images = 'Mytilus'
etalons = 'MytilusEtalons'
my_input = [] 
my = [] 
pele = congo(etalons,images,my_input,my)
print("Mytilus")
print(pele)

images = 'Zostera'
etalons = 'ZosteraEtalons'
my_input = [] 
my = [] 
pele = congo(etalons,images,my_input,my)
print("Zostera")
print(pele)


# 3 task - SIFT

# MIN_MATCH_COUNT = 2

# def get_matched_coordinates(temp_img, map_img):

#     # initiate SIFT detector
#     sift = cv2.SIFT_create()

#     # find the keypoints and descriptors with SIFT
#     kp1, des1 = sift.detectAndCompute(temp_img,None)
#     kp2, des2 = sift.detectAndCompute(map_img, None)

#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=50)
#     search_params = dict(checks=1000)

#     flann = cv2.FlannBasedMatcher(index_params, search_params)

#     # find matches by knn which calculates point distance in 128 dim
#     matches = flann.knnMatch(des1, des2, k=2)
#     # store all the good matches as per Lowe's ratio test.
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75*n.distance:
#             good.append(m)

#     print(len(good))

#     if len(good) > MIN_MATCH_COUNT:
#         src_pts = np.float32(
#             [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32(
#             [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

#         # find homography
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         matchesMask = mask.ravel().tolist()

#         h, w = temp_img.shape
#         pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
#                           [w-1, 0]]).reshape(-1, 1, 2)
#         dst = cv2.perspectiveTransform(pts, M)  # matched coordinates

#         map_img = cv2.polylines(
#             map_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

#     else:
#         print("Not enough matches are found - %d/%d" %
#               (len(good), MIN_MATCH_COUNT))
#         matchesMask = None

#     draw_params = dict(matchColor=(200, 0, 200),  # draw matches in green color
#                        singlePointColor=None,
#                        matchesMask=matchesMask,  # draw only inliers
#                        flags=2)

#     # draw template and map image, matches, and keypoints
#     img3 = cv2.drawMatches(temp_img, kp1, map_img, kp2,
#                            good, None, **draw_params)

#     # show result image
#     plt.imshow(img3, 'gray'), plt.axis('off'), plt.show()

#     # result image path
#     cv2.imwrite(os.path.join('result2.png'), img3)
#     return img3


# # # read images
# temp_img_gray = cv2.imread('MytilusEtalons/10.png', 0)
# map_img_gray = cv2.imread('Mytilus/musla9.png', 0)
# get_matched_coordinates(temp_img_gray, map_img_gray)








#3 task - ORB

# img1 = cv2.imread('MytilusEtalons/10.png',0)
# img2 = cv2.imread('Mytilus/musla9.png',0)

# orb = cv2.ORB_create(nfeatures=500)
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
# # draw first 50 matches
# match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None)
# plt.imshow(match_img),
# plt.axis('off')
# plt.show()

#3 task - 
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt






# 3 task - SIFT Brute force

# MIN_MATCH_COUNT = 2

# def get_matched_coordinates(temp_img, map_img):

#     # initiate SIFT detector
#     sift = cv2.SIFT_create()

#     # find the keypoints and descriptors with SIFT
#     kp1, des1 = sift.detectAndCompute(temp_img,None)
#     kp2, des2 = sift.detectAndCompute(map_img, None)


#     bf = cv2.BFMatcher()

#     # find matches by knn which calculates point distance in 128 dim
    
#     matches = bf.knnMatch(des1, des2, k=2)
#     # store all the good matches as per Lowe's ratio test.
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75*n.distance:
#             good.append(m)

#     print(len(good))

#     if len(good) > MIN_MATCH_COUNT:
#         src_pts = np.float32(
#             [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32(
#             [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

#         # find homography
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         matchesMask = mask.ravel().tolist()

#         h, w = temp_img.shape
#         pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
#                           [w-1, 0]]).reshape(-1, 1, 2)
#         dst = cv2.perspectiveTransform(pts, M)  # matched coordinates

#         map_img = cv2.polylines(
#             map_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

#     else:
#         print("Not enough matches are found - %d/%d" %
#               (len(good), MIN_MATCH_COUNT))
#         matchesMask = None

#     draw_params = dict(matchColor=(200, 0, 200),  # draw matches in green color
#                        singlePointColor=None,
#                        matchesMask=matchesMask,  # draw only inliers
#                        flags=2)

#     # draw template and map image, matches, and keypoints
#     img3 = cv2.drawMatches(temp_img, kp1, map_img, kp2,
#                            good, None, **draw_params)

#     # show result image
#     plt.imshow(img3, 'gray'), plt.axis('off'), plt.show()

#     # result image path
#     cv2.imwrite(os.path.join('resultBF.png'), img3)
#     return img3


# # # read images
# temp_img_gray = cv2.imread('MytilusEtalons/10.png', 0)
# map_img_gray = cv2.imread('Mytilus/musla9.png', 0)
# get_matched_coordinates(temp_img_gray, map_img_gray)