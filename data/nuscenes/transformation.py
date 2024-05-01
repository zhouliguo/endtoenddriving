from pyquaternion import Quaternion
import numpy as np

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def get_global_pose(ept, epr, cst, csr, inverse=False):
    if inverse is False:
        global_from_ego = transform_matrix(ept, Quaternion(epr), inverse=False)
        ego_from_sensor = transform_matrix(cst, Quaternion(csr), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(cst, Quaternion(csr), inverse=True)
        ego_from_global = transform_matrix(ept, Quaternion(epr), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

filepath = 'data/nuscenes/image_t_r/mini_train/scene-0061.csv'
f = open(filepath, 'r')
record = f.readline().strip('\n').split(',')[1:]
record = np.array(record).astype(np.float32)

pose1 = get_global_pose(np.array([record[0],record[1],record[2]]), Quaternion([record[3],record[4],record[5],record[6]]), np.array([record[7],record[8],record[9]]), Quaternion([record[10],record[11],record[12],record[13]]))
pose2 = get_global_pose(np.array([record[0],record[1],record[2]]), Quaternion([record[3],record[4],record[5],record[6]]), np.array([record[7],record[8],record[9]]), Quaternion([record[10],record[11],record[12],record[13]]), True)

egopose_future = pose2.dot(pose1)

origin1 = np.array(pose1[:3, 3])
origin2 = np.array(pose2[:3, 3])
origin3 = np.array(egopose_future[:3, 3])
print(pose1)
print(pose2)
print(origin1)
print(origin2)
print(origin3)
