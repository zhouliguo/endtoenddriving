# 数据库(.db文件)包含以下表：
# camera, category, ego_pose, image, lidar, lidar_box, lidar_pc, log, scenario_tag, scene, track, traffic_light_status

import time
import numpy as np
import sqlite3
import cv2

# 归一化到[0,1]
def norm_0_1(a):
    a_min = np.min(a)
    a_max = np.max(a)

    return (a-a_min)/(a_max-a_min)

class veh_db():
    def __init__(self) -> None:
        pass

def main():
    # 连接数据库文件
    db_file = sqlite3.connect('data/nuplan/mini/2021.10.11.07.12.18_veh-50_00211_00304.db')  
    cur = db_file.cursor()

    '''
    cur.execute("select name from sqlite_master where type='table' order by name")
    print(cur.fetchall())

    cur.execute("PRAGMA table_info(ego_pose)")
    print(cur.fetchall())

    '''


    # 从 lidar_pc 表中同时取出 token 和 ego_pose_token，按时间排序
    cur.execute("select hex(token), hex(ego_pose_token) from lidar_pc order by timestamp")
    pc_tokens = cur.fetchall()

    # 从 lidar_box 表中取出 agents 位置的四个边界
    cur.execute("select max(x), min(x), max(y), min(y) from lidar_box")
    x_max,x_min,y_max,y_min = cur.fetchall()[0]

    print(x_max-x_min)
    print(y_max-y_min)

    #cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    # 将每一帧中的自车和周围 agents 取出 
    for pc_token in pc_tokens:
        map_image = np.zeros((5000, 2700, 3), np.uint8)+255

        # token 在数据库中的存储格式为 x'一串十六进制数值', 例如 x'9ed77323c37354ad'
        pc_token_str = 'x'+'\''+pc_token[0].lower()+'\''
        pc_ego_pose_token_str = 'x'+'\''+pc_token[1].lower()+'\''

        # 从 lidar_box 表中获取 agents 的位置和姿态信息
        cur.execute("select x, y, z, width, length, height from lidar_box where lidar_pc_token="+pc_token_str)
        lidar_boxs = cur.fetchall()

        # 从 ego_pose 表中获取自车的位置和姿态信息
        cur.execute("select x, y, z from ego_pose where token="+pc_ego_pose_token_str)
        egos = cur.fetchall()

        for lidar_box in lidar_boxs:
            x = int((lidar_box[0] - x_min) * 10)
            y = int((lidar_box[1] - y_min) * 10)
            cv2.circle(map_image, (x, y), 20, (0,255,0), -1)
        
        x = int((egos[0][0] - x_min) * 10)
        y = int((egos[0][1] - y_min) * 10)
        cv2.circle(map_image, (x, y), 20, (0,0,255), -1)

        cv2.imshow('1', map_image)
        cv2.waitKey(1)
    cv2.waitKey()

if __name__ == '__main__':
    main()