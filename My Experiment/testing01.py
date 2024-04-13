import cv2
import numpy as np


def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / \
            np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array(
            [first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True


fx = 2.9545e+03
fy = 2.9621e+03
cx = 1.5024e+03
cy = 2.0827e+03
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
K_inv = np.linalg.inv(K)

points1 = [[368, 743], [298, 1016], [496, 654], [295, 937], [309, 793], [322, 747], [290, 916], [301, 996], [300, 711], [295, 937], [309, 793], [498, 996], [469, 717], [295, 937], [442, 1000], [459, 984], [505, 996], [295, 936], [373, 929], [468, 717], [867, 686], [
    466, 997], [398, 783], [317, 748], [296, 937], [443, 1001], [400, 984], [935, 705], [445, 918], [398, 783], [308, 791], [298, 1017], [316, 746], [310, 788], [444, 916], [313, 719], [400, 982], [301, 710], [295, 937], [308, 788], [376, 985], [315, 745], [232, 1594]]
points2 = [[351, 678], [271, 941], [409, 601], [271, 864], [291, 723], [306, 679], [267, 843], [274, 921], [285, 643], [271, 864], [291, 723], [468, 931], [452, 658], [271, 864], [413, 933], [430, 918], [549, 927], [271, 864], [349, 860], [451, 658], [831, 653], [
    437, 929], [378, 719], [300, 680], [271, 864], [414, 933], [373, 914], [893, 673], [420, 853], [378, 719], [291, 721], [271, 943], [301, 677], [292, 719], [421, 851], [298, 653], [373, 913], [286, 644], [271, 865], [293, 716], [347, 913], [300, 677], [301, 1436]]
points1_np = np.array(points1)
points2_np = np.array(points2)

E, mask = cv2.findEssentialMat(points1_np, points2_np, cameraMatrix=K,
                               method=cv2.RANSAC, prob=0.999, threshold=1.0, mask=None)

print('Essential Matrix:\n', E)

# decompose Essential matrix into R and t
U, S, Vt = np.linalg.svd(E)
W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

# iterate over all point correspondences used in the estimation of the fundamental matrix
first_inliers = []
second_inliers = []
for i in range(len(mask)):
    if mask[i]:
        # normalize and homogenize the image coordinates
        first_inliers.append(
            K_inv.dot([points1_np[i, 0], points1_np[i, 1], 1.0]))
        second_inliers.append(
            K_inv.dot([points2_np[i, 0], points2_np[i, 1], 1.0]))

# Determine the correct choice of second camera matrix
# only in one of the four configurations will all the points be in front of both cameras
# First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
R = U.dot(W).dot(Vt)
T = U[:, 2]
if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):

    # Second choice: R = U * W * Vt, T = -u_3
    T = - U[:, 2]
    if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):

        # Third choice: R = U * Wt * Vt, T = u_3
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]

        if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):

            # Fourth choice: R = U * Wt * Vt, T = -u_3
            T = - U[:, 2]

print('Rotation Matrix:\n', R, 'Translational Vector:', T)
