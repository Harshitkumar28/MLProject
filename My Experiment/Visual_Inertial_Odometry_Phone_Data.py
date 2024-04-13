import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import pandas as pd
from MahonyAHRS import MahonyAHRS
from quaternion_operations import quatern2euler, quaternConj, quatern2rotMat, quaternRotate
from scipy.signal import butter, filtfilt


def generate_depth_map(input_img):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # Apply input transforms
    input_batch = transform(input_img).to(device)
    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=input_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    return depth_map


def generate_rgbd(input_img, depth_map):
    color_raw = o3d.geometry.Image(np.asarray(input_img))
    depth_raw = o3d.geometry.Image(np.asanyarray(depth_map))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)
    return rgbd_image


def generate_point_cloud(rgbd_image, camera_intrinsic_o3d):
    # Create the point cloud from images and camera intrisic parameters
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsic_o3d)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def invert_depth_map(depth_map):
    return 255 - depth_map


def project_Kp_in_3D(Kp, depth, fx, fy, cx, cy):
    Kp_3D = []
    for k in Kp:
        u, v = k
        d = 0.1 * np.mean(depth[int(v), int(u), :])
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d
        Kp_3D.append((x, y, z))
    return Kp_3D


def findHammingDistance(v1, v2):
    return np.sum(np.abs(v1 - v2))


def visual_comparison_of_reprojection(_Kp, Kp, height, width):
    img = np.ones((height, width, 3), dtype=np.uint8)*255
    dot_size = 10
    for point, _point in zip(Kp, _Kp):
        col, row = point
        col = int(col)
        row = int(row)
        _col, _row = _point
        _col = int(_col)
        _row = int(_row)
        img[row - dot_size // 2:row + dot_size // 2, col - dot_size //
            2:col + dot_size // 2, :] = [0, 0, 255]  # original --> blue
        if (0 < _col < width and 0 < _row < height):
            img[_row - dot_size // 2:_row + dot_size // 2, _col - dot_size //
                2:_col + dot_size // 2, :] = [255, 0, 0]  # reprojected --> red
    return img


def get_IMU_rotaiton_and_translation(ahrs, gyro, accel, time):

    sample_period = 0
    for t in range(1, len(time)):
        sample_period += time[t] - time[t - 1]
    sample_period /= len(time)

    accX, accY, accZ = accel[:, 0], accel[:, 1], accel[:, 2]
    acc_mag = np.sqrt(accX*accX + accY*accY + accZ*accZ)

    # HP filter to accel data
    filtCutOff = 0.0001
    [b, a] = butter(1, (2*filtCutOff)/(1/sample_period), 'high')
    acc_magFilt = filtfilt(b, a, acc_mag)
    # compute abs value
    acc_magFilt = np.abs(acc_magFilt)
    # LP filter accel data
    filtCutOff = 5
    [b, a] = butter(1, (2*filtCutOff)/(1/sample_period), 'high')
    acc_magFilt = filtfilt(b, a, acc_mag)
    # threshold detection
    stationary = acc_magFilt < 0.055

    R = np.zeros(shape=(3, 3, len(time)))
    quaternion = np.zeros((len(time), 4))

    gyro_copy = np.copy(gyro)
    accel_copy = np.copy(accel)

    for t in range(len(time)):
        if (stationary[t]):
            ahrs.Kp = 0
        else:
            ahrs.Kp = 0.5
        ahrs.UpdateIMU(gyro_copy[t], accel_copy[t])
        quaternion[t] = ahrs.Quaternion
        # total rotation in this window
        R[:, :, t] = quatern2rotMat(ahrs.Quaternion).T

    # # calculating tilt-compensated accelerometer
    tc_Acc = np.zeros_like(accel)
    for t in range(len(time)):
        tc_Acc[t] = R[:, :, t] @ accel[t].T
    # tc_Acc = quaternRotate(accel, quaternConj(quaternion))     ## linear accel relative to earth frame

    # calculate linear acceleration and convert them into m/s^2
    lin_Acc = tc_Acc - np.array([0, 0, 9.8])

    # calculate linear velocity (integrating acceleration)
    lin_Vel = np.zeros_like(lin_Acc)
    for t in range(1, len(time)):
        lin_Vel[t] = lin_Vel[t - 1] + lin_Acc[t] * (time[t] - time[t - 1])

    # high pass filter to linear velocity to remove drift
    order = 2
    filtCutOff = 1.5
    b, a = butter(order, (2 * filtCutOff) / (1/sample_period), 'high')
    lin_Vel_HP = filtfilt(b, a, lin_Vel, axis=0)
    # calculate linear position (integrating velocity)

    lin_Pos = np.zeros_like(lin_Vel_HP)
    for t in range(1, len(time)):
        lin_Pos[t] = lin_Pos[t - 1] + lin_Vel_HP[t] * (time[t] - time[t - 1])

    # High-pass filter linear position to remove drift
    order = 2
    filtCutOff = 1.5
    b, a = butter(order, (2*filtCutOff)/(1/sample_period), 'high')
    lin_Pos_HP = filtfilt(b, a, lin_Pos, axis=0)
    return R[:, :, 0], R[:, :, -1], lin_Pos_HP[-1, :], lin_Acc, lin_Vel_HP, lin_Pos_HP


# midas model loading
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDas", model_type)
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDas", "transforms")
if (model_type == "DPT_Large" or model_type == "DPT_Hybrid"):
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# loading IMU data
accel = pd.read_csv('my_data/VID_20240110_143256accel.csv').to_numpy()
gyro = pd.read_csv('my_data/VID_20240110_143256gyro.csv').to_numpy()
time = pd.read_csv(
    'my_data/VID_20240110_143256_imu_timestamps.csv').to_numpy()
accel = accel[:-2, :]
time = time[:-2, :]
init_time = np.copy(time[0])
for t in range(len(time)):
    time[t] = time[t] - init_time
time = time * 1e-9

sample_period = 0

for t in range(1, len(time)):
    sample_period += time[t] - time[t - 1]

sample_period /= len(time)

AHRS = MahonyAHRS(SamplePeriod=sample_period, Kp=1, Ki=0)

# initial convergance
initPeriod = 2
indexSel = np.arange(0, np.argmax(time > time[0] + initPeriod) + 1)
for i in range(2000):
    AHRS.UpdateIMU(np.array([0, 0, 0]), np.array([np.mean(
        accel[indexSel, 0]), np.mean(accel[indexSel, 1]), np.mean(accel[indexSel, 2])]))

# video path
video_path = "my_data\VID_20240110_143256.mp4"
# timestep
timestep = 0.5  # extract frames in every 0.01s
# open the video file
cap = cv2.VideoCapture(video_path)
# get the fps of the video
fps = cap.get(cv2.CAP_PROP_FPS)
# calculate the frame interval between the timestep
frame_interval = int(fps * timestep)
# get the intial frame
_, frame1 = cap.read()
height, width, _ = frame1.shape
frame1_depth = generate_depth_map(frame1)
frame1_depth_inv = invert_depth_map(frame1_depth)
t1 = time[0]
# Camera Parameters
fx = 2.9545e+03
fy = 2.9621e+03
cx = 1.5024e+03
cy = 2.0827e+03
# Intrinsic Matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
# Creating ORB object
orb = cv2.ORB_create()
# detecting keypoints from the first frame
Kp1, des1 = orb.detectAndCompute(frame1, None)
frame1_ORB = cv2.drawKeypoints(frame1, Kp1, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
Kp1_np = np.array([K.pt for K in Kp1])
Kp1_3D = project_Kp_in_3D(Kp1_np, frame1_depth, fx, fy, cx, cy)

current_frame = 1
time_tolerance = 1e-2

idx1 = np.where(abs(time - t1) <= time_tolerance)[0][0]

cv2.namedWindow('KeyPoint Matching', cv2.WINDOW_NORMAL)
cv2.resizeWindow('KeyPoint Matching', width=1080, height=1920)

while True:
    ret, frame2 = cap.read()
    if (not ret):
        break
    if (current_frame % frame_interval == 0):
        t2 = current_frame / fps
        frame2_depth = generate_depth_map(frame2)
        frame2_depth_inv = invert_depth_map(frame2_depth)
        Kp2, des2 = orb.detectAndCompute(frame2, None)
        frame2_ORB = cv2.drawKeypoints(frame2, Kp2, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        Kp2_np = np.array([K.pt for K in Kp2])
        Kp2_3D = project_Kp_in_3D(Kp2_np, frame2_depth, fx, fy, cx, cy)
        idx2 = np.where(abs(time - t2) <= time_tolerance)[0][0]
        _gyro = gyro[idx1:idx2 + 1, :]
        _accel = accel[idx1:idx2 + 1, :]
        _time = time[idx1:idx2 + 1, :]
        print(idx2 - idx1 + 1)
        R1, R2, t, lin_Acc, lin_Vel_HP, lin_Pos_HP = get_IMU_rotaiton_and_translation(
            AHRS, _gyro, _accel, _time)
        # transforming (rotating and translating) 3D keypoints
        transformed_points = np.zeros_like(Kp1_np)
        _Kp2_3D = []
        T1 = np.eye(4)
        T2 = np.eye(4)
        for i in range(len(Kp1_3D)):
            # _Kp2_3D.append(R @ Kp1_3D[i] + t)
            T1[0:3, 0:3] = R1
            T2[0:3, 3] = np.zeros(shape=(3,))
            T2[0:3, 0:3] = R2
            T2[0:3, 3] = t
            P1 = np.append(Kp1_3D[i], 1)
            # P2 = np.linalg.inv(T) @ P1
            P2 = T2 @ np.linalg.inv(T1) @ P1
            P2 = P2[:-1]
            _Kp2_3D.append(P2)
        # back project the keypoints to the second frame
        _Kp2 = []
        for point in _Kp2_3D:
            pt = K @ point
            pt = pt / abs(pt[-1])
            pt = pt[:-1]
            _Kp2.append(pt)
        # Performing matches within neighborhood
        neighborhood_radius = 400
        matches = []
        found_matches = 0
        for i, (_p, _des) in enumerate(zip(_Kp2, des1)):
            nearby_original_Kp_idxs = np.where(
                np.linalg.norm(Kp2_np - _p, axis=1) < neighborhood_radius
            )[0]
            best_matched_index = -1
            if (len(nearby_original_Kp_idxs)):
                # match with nearby keypoints
                nearby_matches = []
                min_dis = 1500
                for j, desc2 in zip(nearby_original_Kp_idxs, des2[nearby_original_Kp_idxs]):
                    hamming_dis = findHammingDistance(_des, desc2)
                    if (hamming_dis < min_dis):
                        min_dis = hamming_dis
                        best_matched_index = j
            if (best_matched_index != -1):
                found_matches += 1
            matches.append(best_matched_index)
        # performing keypoint matches between Kp1 and Kp2
        matching_visual = np.zeros((height, width * 2, 3), dtype=np.uint8)
        matching_visual[:, :width, :] = frame1_ORB
        matching_visual[:, width:, :] = frame2_ORB
        points1 = []
        points2 = []
        for i in range(Kp1_np.shape[0]):
            if (matches[i] != -1):
                pt1 = (int(Kp1_np[i][0]), int(Kp1_np[i][1]))
                pt2 = (int(Kp2_np[matches[i]][0] + width),
                       int(Kp2_np[matches[i]][1]))
                cv2.line(matching_visual, pt1, pt2, (0, 255, 0), 2)
                pt1, pt2 = list(pt1), list(pt2)
                pt2[0] = pt2[0] - width
                points1.append(pt1)
                points2.append(pt2)
        cv2.imshow('KeyPoint Matching', matching_visual)

        # frame = visual_comparison_of_reprojection(_Kp2, Kp2_np, height, width)
        # cv2.imshow('KeyPoint Matching', frame)
        # plt.imshow(frame)
        cv2.waitKey(1)
        plt.axis('off')
        # fig, ax = plt.subplots(figsize=(8, 4))
        # ax.plot(_time, lin_Acc[:, 0], 'r', label='linear acc x')
        # ax.plot(_time, lin_Acc[:, 1], 'g', label='linear acc y')
        # ax.plot(_time, lin_Acc[:, 2], 'b', label='linear acc z')
        # ax.set_title('Linear Acceleration')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Acceleration (m/s^2)')
        # ax.legend()
        # fig, ax = plt.subplots(figsize=(8, 4))
        # ax.plot(_time, lin_Vel_HP[:, 0], 'r', label='lin vel x')
        # ax.plot(_time, lin_Vel_HP[:, 1], 'g', label='lin vel y')
        # ax.plot(_time, lin_Vel_HP[:, 2], 'b', label='lin vel z')
        # ax.set_title('Linear Velocity')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Velocity (m/s)')
        # ax.legend()
        # fig, ax = plt.subplots(figsize=(8, 4))
        # ax.plot(_time, lin_Pos_HP[:, 0], 'r', label='lin pos hp x')
        # ax.plot(_time, lin_Pos_HP[:, 1], 'g', label='lin pos hp y')
        # ax.plot(_time, lin_Pos_HP[:, 2], 'b', label='lin pos hp z')
        # ax.set_title('High-pass Filtered Linear Position')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Position (m)')
        # ax.legend()
        # plt.show()
        # plt.pause(1)
        # plt.close()
        t1 = t2
        frame1_ORB = frame2_ORB
        frame1_depth = frame2_depth
        frame1_depth_inv = frame2_depth_inv
        idx1 = idx2
        Kp1_np = Kp2_np
        des1 = des2
        Kp1_3D = Kp2_3D
    current_frame += 1
