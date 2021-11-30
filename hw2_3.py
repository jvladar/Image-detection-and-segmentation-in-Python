import os
# from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Task 3
# 1) SIFT with Flann based matching

MIN_MATCH_COUNT = 2

def get_matched_coordinates(temp_img, map_img):

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(temp_img,None)
    kp2, des2 = sift.detectAndCompute(map_img, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=50)
    search_params = dict(checks=1000)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # find matches by knn which calculates point distance in 128 dim
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(len(good))

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = temp_img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  

        map_img = cv2.polylines(
            map_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches were found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(200, 0, 200),  
                       singlePointColor=None,
                       matchesMask=matchesMask,  
                       flags=2)

    img3 = cv2.drawMatches(temp_img, kp1, map_img, kp2,
                           good, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.axis('off'), plt.show()

    cv2.imwrite(os.path.join('resultFB.png'), img3)
    return img3

temp_img_gray = cv2.imread('MytilusEtalons/10.png', 0)
map_img_gray = cv2.imread('Mytilus/musla9.png', 0)
get_matched_coordinates(temp_img_gray, map_img_gray)


# 2) ORB with BruteForce matching

img1 = cv2.imread('MytilusEtalons/10.png',0)
img2 = cv2.imread('Mytilus/musla9.png',0)

orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# draw first 30 matches
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None)
cv2.imwrite(os.path.join('resultORB.png'), match_img)
plt.imshow(match_img),
plt.axis('off')
plt.show()


# 3) SIFT with BruteForce matching

MIN_MATCH_COUNT = 2

def get_matched_coordinates(temp_img, map_img):
    # initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(temp_img,None)
    kp2, des2 = sift.detectAndCompute(map_img, None)

    bf = cv2.BFMatcher()

    # find matches by knn which calculates point distance in 128 dim
    matches = bf.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    print(len(good))

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = temp_img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  # matched coordinates

        map_img = cv2.polylines(
            map_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    # draw template and map image, matches, and keypoints
    img3 = cv2.drawMatches(temp_img, kp1, map_img, kp2,
                           good, None, **draw_params)

    # show and save result image
    plt.imshow(img3, 'gray'), plt.axis('off'), plt.show()
    cv2.imwrite(os.path.join('resultBF.png'), img3)
    return img3

# read images
temp_img_gray = cv2.imread('MytilusEtalons/10.png', 0)
map_img_gray = cv2.imread('Mytilus/musla9.png', 0)
get_matched_coordinates(temp_img_gray, map_img_gray)