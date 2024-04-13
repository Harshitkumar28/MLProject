import numpy as np
import cv2

fx = 2.9545e+03
fy = 2.9621e+03
cx = 1.5024e+03
cy = 2.0827e+03
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

Proj_Mat1 = np.concatenate((K, np.ones((3, 1))), axis=1)
points1 = [[368, 743], [298, 1016], [496, 654], [295, 937], [309, 793], [322, 747], [290, 916], [301, 996], [300, 711], [295, 937], [309, 793], [498, 996], [469, 717], [295, 937], [442, 1000], [459, 984], [505, 996], [295, 936], [373, 929], [468, 717], [867, 686], [
    466, 997], [398, 783], [317, 748], [296, 937], [443, 1001], [400, 984], [935, 705], [445, 918], [398, 783], [308, 791], [298, 1017], [316, 746], [310, 788], [444, 916], [313, 719], [400, 982], [301, 710], [295, 937], [308, 788], [376, 985], [315, 745], [232, 1594]]
points2 = [[351, 678], [271, 941], [409, 601], [271, 864], [291, 723], [306, 679], [267, 843], [274, 921], [285, 643], [271, 864], [291, 723], [468, 931], [452, 658], [271, 864], [413, 933], [430, 918], [549, 927], [271, 864], [349, 860], [451, 658], [831, 653], [
    437, 929], [378, 719], [300, 680], [271, 864], [414, 933], [373, 914], [893, 673], [420, 853], [378, 719], [291, 721], [271, 943], [301, 677], [292, 719], [421, 851], [298, 653], [373, 913], [286, 644], [271, 865], [293, 716], [347, 913], [300, 677], [301, 1436]]
points1_np = np.array(points1)
points2_np = np.array(points2)
E, mask = cv2.findEssentialMat(points1_np, points2_np, cameraMatrix=K,
                               method=cv2.RANSAC, prob=0.999, threshold=1.0, mask=None)


def _form_transf(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def sum_z_cal_relative_scale(q1, q2, R, t):
    # Get the transformation matrix
    T = _form_transf(R, t)
    # Make the projection matrix
    '''
    Proj_Mat1 --> 3x4
    T --> 4x4
    Proj_Mat2 --> 3x4
    '''
    Proj_Mat2 = np.matmul(Proj_Mat1, T)

    # Triangulate the 3D points
    hom_Q1 = cv2.triangulatePoints(
        Proj_Mat1, Proj_Mat2, q1.T, q2.T).astype(np.float64)
    # Also seen from cam 2
    hom_Q2 = np.matmul(T, hom_Q1)

    # # removing zero elements from homogeneous component
    epsilon = 1e-8
    hom_Q1[3, hom_Q1[3, :] == 0] = epsilon
    hom_Q2[3, hom_Q2[3, :] == 0] = epsilon

    # Un-homogenize
    uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
    uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

    # Find the number of points there has positive z coordinate in both cameras
    sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
    sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

    # Form point pairs and calculate the relative scale
    relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) /
                             np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
    return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale


def decom_Essential_Matrix(E, q1, q2):
    # Decompose the essential matrix
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = np.squeeze(t)
    # Make a list of the different possible pairs
    pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
    # Check which solution there is the right one
    z_sums = []
    relative_scales = []
    for R, t in pairs:
        z_sum, scale = sum_z_cal_relative_scale(q1, q2, R, t)
        z_sums.append(z_sum)
        relative_scales.append(scale)
    # Select the pair there has the most points with positive z coordinate
    right_pair_idx = np.argmax(z_sums)
    right_pair = pairs[right_pair_idx]
    relative_scale = relative_scales[right_pair_idx]
    R1, t = right_pair
    t = t * relative_scale
    return [R1, t]


R, t = decom_Essential_Matrix(E, points1_np, points2_np)
print(R, t)
