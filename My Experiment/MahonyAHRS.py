import numpy as np
from quaternion_operations import quaternProd, quaternConj


class MahonyAHRS:
    def __init__(self, SamplePeriod=1/256, Quaternion=np.array([1, 0, 0, 0]), Kp=1, Ki=0):
        self.SamplePeriod = SamplePeriod
        self.Quaternion = Quaternion
        self.Kp = Kp
        self.Ki = Ki
        self.eInt = np.array([0, 0, 0], dtype=np.float64)

    def Update(self, Gyroscope, Accelerometer, Magnetometer):
        q = self.Quaternion

        # Normalise accelerometer measurement
        accel_norm = np.linalg.norm(Accelerometer)
        if accel_norm == 0:
            return
        Accelerometer /= accel_norm

        # Normalise magnetometer measurement
        mag_norm = np.linalg.norm(Magnetometer)
        if mag_norm == 0:
            return
        Magnetometer /= mag_norm

        # Reference direction of Earth's magnetic field
        h = quaternProd(q, quaternProd(
            np.array([0, *Magnetometer]), quaternConj(q)))
        b = np.array([0, np.linalg.norm([h[1], h[2]]), 0, h[3]])

        # Estimated direction of gravity and magnetic field
        v = np.array([
            2*(q[1]*q[3] - q[0]*q[2]),
            2*(q[0]*q[1] + q[2]*q[3]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        ])
        w = np.array([
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]),
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]),
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2)
        ])

        # Error is sum of cross product between estimated direction and measured direction of fields
        e = np.cross(Accelerometer, v) + np.cross(Magnetometer, w)

        if self.Ki > 0:
            self.eInt += e * self.SamplePeriod
        else:
            self.eInt = np.array([0, 0, 0])

        # Apply feedback terms
        Gyroscope = Gyroscope + self.Kp * e + self.Ki * self.eInt

        # Compute rate of change of quaternion
        qDot = 0.5 * quaternProd(q, np.array([0, *Gyroscope]))

        # Integrate to yield quaternion
        q += qDot * self.SamplePeriod
        self.Quaternion = q / np.linalg.norm(q)

    def UpdateIMU(self, Gyroscope, Accelerometer):
        q = self.Quaternion

        # Normalise accelerometer measurement
        accel_norm = np.linalg.norm(Accelerometer)
        if accel_norm == 0:
            return
        Accelerometer /= accel_norm
        # Estimated direction of gravity and magnetic flux
        v = np.array([
            2*(q[1]*q[3] - q[0]*q[2]),
            2*(q[0]*q[1] + q[2]*q[3]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        ])

        # Error is sum of cross product between estimated direction and measured direction of field
        e = np.cross(Accelerometer, v)

        if self.Ki > 0:
            self.eInt += e * self.SamplePeriod
        else:
            self.eInt = np.array([0, 0, 0], dtype=np.float64)

        # Apply feedback terms
        Gyroscope = Gyroscope + self.Kp * e + self.Ki * self.eInt

        # Compute rate of change of quaternion
        qDot = 0.5 * quaternProd(q.reshape(1, -1),
                                 np.array([[0, *Gyroscope]]))[0]

        # Integrate to yield quaternion
        q = q + qDot * self.SamplePeriod
        self.Quaternion = q / np.linalg.norm(q)
