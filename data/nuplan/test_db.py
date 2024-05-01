# 数据库(.db文件)包含以下表：
# camera, category, ego_pose, image, lidar, lidar_box, lidar_pc, log, scenario_tag, scene, track, traffic_light_status

import time
import numpy as np
import sqlite3
import cv2
from test_maps import plot_map
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from state_representation import StateSE2
from orientation_box import OrientedBox
from Quaternion import Quaternion
# 归一化到[0,1]
def norm_0_1(a):
    a_min = np.min(a)
    a_max = np.max(a)

    return (a-a_min)/(a_max-a_min)

# 连接数据库文件
db_file = sqlite3.connect('data/nuplan/mini/train/2021.05.12.22.00.38_veh-35_01008_01518.db')  
cur = db_file.cursor()

'''
cur.execute("select name from sqlite_master where type='table' order by name")
print(cur.fetchall())

cur.execute("PRAGMA table_info(ego_pose)")
print(cur.fetchall())

cur.execute("select x from ego_pose")
xs = cur.fetchall()
print('x:', xs[:10])

cur.execute("select y from ego_pose")
ys = cur.fetchall()
print('y:', ys[:10])

cur.execute("select width from lidar_box")
ws = cur.fetchall()
print('width:', ws[:10])

cur.execute("select length from lidar_box")
ls = cur.fetchall()
print('length:', ls[:10])
'''

# 从 lidar_pc 表中同时取出 token 和 ego_pose_token，按时间排序
cur.execute("select hex(token), hex(ego_pose_token), timestamp from lidar_pc order by timestamp")
pc_tokens = cur.fetchall()

fig, ax = plt.subplots()
# 将每一帧中的自车和周围 agents 取出 
for pc_token in pc_tokens:
    #chose the timestamp where you want to begin to plot
    if(pc_token[2] > 1620858014050860):
    # token 在数据库中的存储格式为 x'一串十六进制数值', 例如 x'9ed77323c37354ad'
        pc_token_str = 'x'+'\''+pc_token[0].lower()+'\''
        pc_ego_pose_token_str = 'x'+'\''+pc_token[1].lower()+'\''

        # 从 lidar_box 表中获取 agents 的位置和姿态信息
        cur.execute("select x, y, z, yaw, width, length, height from lidar_box where lidar_pc_token="+pc_token_str)
        lidar_boxs = cur.fetchall()
        
        # 从 ego_pose 表中获取自车的位置和姿态信息
        cur.execute("select x, y, z, qw, qx, qy, qz from ego_pose where token="+pc_ego_pose_token_str)
        egos = cur.fetchall()
        q = Quaternion(egos[0][3], egos[0][4], egos[0][5], egos[0][6])
        ego_pose = StateSE2(egos[0][0], egos[0][1], q.yaw_pitch_roll[0])
        #Get four corner points of the ego
        ego_FL, ego_RL, ego_RR, ego_FR = OrientedBox(ego_pose, 4.049, 1.1485 * 2.0, 1.777).all_corners()
        ego_points = [(ego_FL.x,ego_FL.y),(ego_RL.x,ego_RL.y),(ego_RR.x,ego_RR.y),(ego_FR.x,ego_FR.y)]
        #plot the ego based on their four corner points
        ego_polygon = patches.Polygon(ego_points, facecolor='red')
        ax.add_patch(ego_polygon)
        for lidar_box in lidar_boxs:

            agent_pose = StateSE2(lidar_box[0], lidar_box[1], lidar_box[3])
            # Get four corner points of the agents
            FL, RL, RR, FR = OrientedBox(agent_pose,lidar_box[5],lidar_box[4],lidar_box[6]).all_corners()#Ponit2D datatype
            center_x = lidar_box[0]
            center_y = lidar_box[1]
            points = [(FL.x,FL.y),(RL.x,RL.y),(RR.x,RR.y),(FR.x,FR.y)]
            #plot the agents based on their four corner points
            polygon = patches.Polygon(points, facecolor='blue')
            ax.add_patch(polygon)
        x = egos[0][0]
        y = egos[0][1]
        # get the base line paths in lanes and intersections and the intersection background 
        candidate_blps_in_lanes,candidate_blps_in_intersections,candidate_intersections = plot_map(x,y)
        
        candidate_blps_in_lanes.plot(ax=ax, color='green')

        candidate_blps_in_intersections.plot(ax=ax, color='black')
        
        candidate_intersections.plot(ax=ax, color='grey',alpha=0.5)

        ax.set_aspect('equal')
        #set the limit for x,y axis
        ax.set_xlim(x-100, x+100)
        ax.set_ylim(y-100, y+100)
        # print the timestep for the current picture
        ax.set_title(f'Timestep: {pc_token[2]}')
        plt.draw()
        plt.pause(0.0001)
        ax.clear()
