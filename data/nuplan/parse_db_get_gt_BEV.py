# 数据库(.db文件)包含以下表：
# camera, category, ego_pose, image, lidar, lidar_box, lidar_pc, log, scenario_tag, scene, track, traffic_light_status
import math
import time
import numpy as np
import sqlite3
import cv2
# from parse_maps import plot_map
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
db_file = sqlite3.connect('/home/chang/nuplan/dataset/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db')  
cur = db_file.cursor()

# 从 lidar_pc 表中同时取出 token 和 ego_pose_token，按时间排序
cur.execute("select hex(token), hex(ego_pose_token), timestamp from lidar_pc order by timestamp")
pc_tokens = cur.fetchall()
fig, ax = plt.subplots(figsize=(19.2, 10.8))
count = 1
# fig, ax = plt.subplots()
# 将每一帧中的自车和周围 agents 取出 
for pc_token in pc_tokens[1::2]:
    if(pc_token[2] == 1620857889701076):
        pc_token_str = 'x'+'\''+pc_token[0].lower()+'\''
        pc_ego_pose_token_str = 'x'+'\''+pc_token[1].lower()+'\''
        cur.execute("select x, y, z, yaw, width, length, height from lidar_box where lidar_pc_token="+pc_token_str)
        lidar_boxs = cur.fetchall()
        cur.execute("select x, y, z, qw, qx, qy, qz from ego_pose where token="+pc_ego_pose_token_str)
        egos = cur.fetchall()
        q = Quaternion(egos[0][3], egos[0][4], egos[0][5], egos[0][6])
        ego_pose = StateSE2(egos[0][0], egos[0][1], q.yaw_pitch_roll[0])
        #Get four corner points of the ego
        ego_FL, ego_RL, ego_RR, ego_FR = OrientedBox(ego_pose, 4.049, 1.1485 * 2.0, 1.777).all_corners()
        # ego_points = [(ego_FL.x,ego_FL.y),(ego_RL.x,ego_RL.y),(ego_RR.x,ego_RR.y),(ego_FR.x,ego_FR.y)]
        # EGO车辆的原始坐标
        ego_x = egos[0][0]
        ego_y = egos[0][1]
        center_ego_x = ego_x
        center_ego_y = ego_y

        # 1. 计算偏移量
        offset_x = 0.5 - center_ego_x
        offset_y = 0.5 - center_ego_y

        # 2. 应用偏移量到ego车
        ego_FL.x += offset_x
        ego_FL.y += offset_y
        ego_RL.x += offset_x
        ego_RL.y += offset_y
        ego_RR.x += offset_x
        ego_RR.y += offset_y
        ego_FR.x += offset_x
        ego_FR.y += offset_y

        # 创建一个列表来收集所有的点
        all_points = [[ego_FL, ego_RL, ego_RR, ego_FR]]

        # with open("car_coordination/car.txt","a")as f:
        #     f.write(f"EGO_CAR_COORDINATION:\n\n{ego_points}\n\nOTHER_CAR_COORDINATION:\n\n")
        # ego_polygon = patches.Polygon(ego_points, facecolor='red')
        # ax.add_patch(ego_polygon)

        for lidar_box in lidar_boxs:
            agent_pose = StateSE2(lidar_box[0], lidar_box[1], lidar_box[3])
            FL, RL, RR, FR = OrientedBox(agent_pose, lidar_box[5], lidar_box[4], lidar_box[6]).all_corners()
        
            # 2. 应用偏移量到其他车
            FL.x += offset_x
            FL.y += offset_y
            RL.x += offset_x
            RL.y += offset_y
            RR.x += offset_x
            RR.y += offset_y
            FR.x += offset_x
            FR.y += offset_y

            # 添加到all_points列表中
            all_points.append([FL, RL, RR, FR])

        # 重新计算所有点的最大和最小x和y坐标
        min_x = min(point.x for car_points in all_points for point in car_points)
        max_x = max(point.x for car_points in all_points for point in car_points)
        min_y = min(point.y for car_points in all_points for point in car_points)
        max_y = max(point.y for car_points in all_points for point in car_points)
        
        # 计算归一化比例
        scale_x = 1.0 / (max_x - min_x)
        scale_y = 1.0 / (max_y - min_y)
        scale = max(scale_x, scale_y)
        # 选最大值用统一的缩放比例，使得矩形仍是矩形
        scale_x = scale
        scale_y = scale
        # 对所有点应用归一化
        for car_points in all_points:
            for point in car_points:
                point.x = (point.x - min_x) * scale_x
                point.y = 1 - ((point.y - min_y) * scale_y)
                # point.y = (point.y - min_y) * scale_y

        # 计算ego车(红色车)的中心点坐标
        ego_center_x = sum(p.x for p in all_points[0]) / 4
        ego_center_y = sum(p.y for p in all_points[0]) / 4

        # 计算平移量使红色车中心移到(0.5, 0.5)
        translate_x = 0.5 - ego_center_x
        translate_y = 0.5 - ego_center_y

        # 应用平移量到所有车辆
        for car_points in all_points:
            for point in car_points:
                point.x += translate_x
                point.y += translate_y

        for car_points in all_points:
            for point in car_points:
                point.x -= 0.5
                point.y -= 0.5

        # 计算FL和RL的连线与y轴之间的夹角
        theta = math.atan2(all_points[0][0].y - all_points[0][1].y, all_points[0][0].x - all_points[0][1].x)        

        # 旋转的角度是π/2 - theta
        rotate_angle = math.pi/2 - theta        

        # 旋转所有点
        for car_points in all_points:
            for point in car_points:
                new_x = math.cos(rotate_angle) * point.x - math.sin(rotate_angle) * point.y
                new_y = math.sin(rotate_angle) * point.x + math.cos(rotate_angle) * point.y
                point.x, point.y = new_x, new_y

        for car_points in all_points:
            for point in car_points:
                point.x += 0.5
                point.y += 0.5    

        # 删除超出范围的车辆
        all_points = [car_points for car_points in all_points if all(0 <= point.x <= 1 and 0 <= point.y <= 1 for point in car_points)]

        with open('car_coordination/car_coordinates.txt', 'w') as file:
            # 绘制每辆车的多边形
            for car_points in all_points:
                points = [(p.x, p.y) for p in car_points]
                color = 'red' if car_points == all_points[0] else 'blue'
                

                def calculate_distance(point1, point2):
                    return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5
                 # 计算矩形的长和宽
                length = calculate_distance(points[0], points[1])  # FL to RL
                width = calculate_distance(points[0], points[3])   # FL to FR

                # 计算矩形面积
                area = length * width

                if area > 0.0001:
                    polygon = patches.Polygon(points, facecolor=color)
                    ax.add_patch(polygon)
                # 计算中心点坐标
                    center_x = sum(p[0] for p in points) / 4
                    center_y = sum(p[1] for p in points) / 4
    
                    # 写入中心点坐标到文件中
                    file.write(f"{color} car center: ({center_x:.6f}, {center_y:.6f})\n")       
                    file.write(f"{color} car area: {area:.6f}\n")
    
                    # 写入四个顶点坐标
                    file.write(f"{color} car FL: ({points[0][0]:.6f}, {points[0][1]:.6f})\n")
                    file.write(f"{color} car RL: ({points[1][0]:.6f}, {points[1][1]:.6f})\n")
                    file.write(f"{color} car RR: ({points[2][0]:.6f}, {points[2][1]:.6f})\n")
                    file.write(f"{color} car FR: ({points[3][0]:.6f}, {points[3][1]:.6f})\n")
                    file.write("\n")  # 添加一个空行以分隔每辆车的数据
                
        # 设置图的参数
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()  # 反转Y轴
        ax.set_aspect('equal')
        plt.show()


########## 以下代码为测试代码

        # for lidar_box in lidar_boxs:

        #     agent_pose = StateSE2(lidar_box[0], lidar_box[1], lidar_box[3])
        #     # Get four corner points of the agents
        #     FL, RL, RR, FR = OrientedBox(agent_pose,lidar_box[5],lidar_box[4],lidar_box[6]).all_corners()#Ponit2D datatype
        #     center_x = lidar_box[0]
        #     center_y = lidar_box[1]

        #     center_agent_x = (FL.x + RL.x + RR.x + FR.x) / 4.0
        #     center_agent_y = (FL.y + RL.y + RR.y + FR.y) / 4.0
        #     epsilon = 0.000001   

        #     assert abs(center_agent_x - center_x) < epsilon, f"Expected center_ego_x to be {center_agent_x}, but got {center_x}"
        #     assert abs(center_agent_y - center_y) < epsilon, f"Expected center_ego_y to be {center_agent_y}, but got {center_y}"
        """
            同理,这里center_x 和 center_y 是一个agent box的中心点x和y坐标
        """
##########################################################################################
        #     points = [(FL.x - ego_x, FL.y - ego_y), (RL.x - ego_x, RL.y - ego_y), (RR.x - ego_x, RR.y - ego_y), (FR.x - ego_x, FR.y - ego_y)]
        #     print(points)
        #     with open("car_coordination/car.txt","a")as f:
        #         f.write(f"{points}\n\n")
        #     #plot the agents based on their four corner points
        #     polygon = patches.Polygon(points, facecolor='blue')
        #     ax.add_patch(polygon)

        # ax.set_aspect('equal')
        # ax.set_xlim(-60, 60)
        # ax.set_ylim(-60, 60)
        # ax.axis('off')
        # #ax.set_title(f'Timestep: {pc_token[2]}')
        # plt.draw()
        # # plt.savefig(f"/home/chang/Desktop/Guided Research/Dataset_BEV/BEV/{str(count).zfill(4)}_gt.jpg", dpi=100)  # 设置图像分辨率
        # plt.pause(150)
        # ax.clear()
        # count += 1 
        # # get the base line paths in lanes and intersections and the intersection background 

#######################################
        # fig = plt.figure(figsize=(19.2, 10.8))
        # ax.set_aspect('equal')
        # #set the limit for x,y axis
        # ax.set_xlim(x-100, x+100)
        # ax.set_ylim(y-100, y+100)
        # # print the timestep for the current picture
        # ax.axis('off')
        # ax.set_title(f'Timestep: {pc_token[2]}')
        # plt.draw()
        # plt.savefig(f"/home/chang/Desktop/Guided Research/Dataset_BEV/BEV/{str(count).zfill(4)}_gt.jpg",dpi=300)
        # plt.pause(0.0001)
        # ax.clear()
        # count += 1
#############################################

# ############################################################################
        
#         # 验证一下 上面ego_x 和 ego_y是不是ego的中心点坐标
#         center_ego_x = (ego_FL.x + ego_RL.x + ego_RR.x + ego_FR.x) / 4.0
#         center_ego_y = (ego_FL.y + ego_RL.y + ego_RR.y + ego_FR.y) / 4.0        

#         # 设置一个很小的误差值，例如0.0001
#         epsilon = 0.000001        

#         # 使用assert检查坐标
#         assert abs(center_ego_x - ego_x) < epsilon, f"Expected center_ego_x to be {center_ego_x}, but got {ego_x}"
#         assert abs(center_ego_y - ego_y) < epsilon, f"Expected center_ego_y to be {center_ego_y}, but got {ego_y}"
#         """
#         ego_x 是自车的中心点x坐标
#         ego_y 是自车的中心点y坐标
#         """
# #############################################################################