import numpy as np


def quaternProd(a, b):
    ab = np.zeros((a.shape[0], 4))
    ab[:, 0] = a[:, 0] * b[:, 0] - a[:, 1] * \
        b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3]
    ab[:, 1] = a[:, 0] * b[:, 1] + a[:, 1] * \
        b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2]
    ab[:, 2] = a[:, 0] * b[:, 2] - a[:, 1] * \
        b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1]
    ab[:, 3] = a[:, 0] * b[:, 3] + a[:, 1] * \
        b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]

    return ab


def quaternConj(q):
    qConj = np.column_stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]))
    return qConj


def quatern2euler(q):
    R = np.zeros((3, 3, q.shape[0]))

    R[0, 0, :] = 2 * q[:, 0]**2 - 1 + 2 * q[:, 1]**2
    R[1, 0, :] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    R[2, 0, :] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    R[2, 1, :] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R[2, 2, :] = 2 * q[:, 0]**2 - 1 + 2 * q[:, 3]**2

    phi = np.arctan2(R[2, 1, :], R[2, 2, :])
    theta = -np.arctan(R[2, 0, :] / np.sqrt(1 - R[2, 0, :]**2))
    psi = np.arctan2(R[1, 0, :], R[0, 0, :])

    euler = np.column_stack((phi.T, theta.T, psi.T))
    return euler


def quatern2rotMat(q):
    # Initialize rotation matrix array
    R = np.zeros((3, 3))

    R[0, 0] = 2 * q[0]**2 - 1 + 2 * q[1]**2
    R[0, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
    R[0, 2] = 2 * (q[1] * q[3] - q[0] * q[2])
    R[1, 0] = 2 * (q[1] * q[2] - q[0] * q[3])
    R[1, 1] = 2 * q[0]**2 - 1 + 2 * q[2]**2
    R[1, 2] = 2 * (q[2] * q[3] + q[0] * q[1])
    R[2, 0] = 2 * (q[1] * q[3] + q[0] * q[2])
    R[2, 1] = 2 * (q[2] * q[3] - q[0] * q[1])
    R[2, 2] = 2 * q[0]**2 - 1 + 2 * q[3]**2

    return R


def quaternRotate(v, q):
    row, col = v.shape
    v0XYZ = quaternProd(quaternProd(q, np.column_stack((np.zeros((row, 1)), v))), quaternConj(q))
    v = v0XYZ[:, 1:]
    return v
