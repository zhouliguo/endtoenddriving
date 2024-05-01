import datetime
import time
import json
import numpy as np
import random
import glob
import os
import shutil

'''
timestamp = time.time()
print(timestamp)
dt = datetime.datetime.fromtimestamp(timestamp)
date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
print(date_str)

dt = datetime.datetime.fromtimestamp(1531883531912467/1000000)
date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
print(date_str)
dt = datetime.datetime.fromtimestamp(1531883531920339/1000000)
date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
print(date_str)
'''

val = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

mini_train = \
    ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']

mini_val = \
    ['scene-0103', 'scene-0916']
# 将图片和其对应的位置坐标提取出来，按scene分别存在csv文件中
def extract_image_position():
    f_scene = open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/scene.json', 'r')
    f_sample = open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/sample.json', 'r')
    f_sample_data = open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/sample_data.json', 'r')
    f_ego_pose = open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/ego_pose.json', 'r')

    scene = json.load(f_scene)
    sample = json.load(f_sample)
    sample_data = json.load(f_sample_data)
    ego_pose = json.load(f_ego_pose)

    f_scene.close()
    f_sample.close()
    f_sample_data.close()
    f_ego_pose.close()

    sample_data_new = []
    sd_sample_token = []
    for sd in sample_data:
        if sd['fileformat'] == 'jpg' and sd['is_key_frame'] == True:    # 把sample_data.json中的包含图片关键帧的记录取出来
            sample_data_new.append(sd)
            sd_sample_token.append(sd['sample_token'])
    sample_data_new = np.array(sample_data_new)
    sd_sample_token = np.array(sd_sample_token)

    ep_token = []
    for ep in ego_pose:
        ep_token.append(ep['token'])
    ep_token = np.array(ep_token)

    output = ['', '', '', '', '', '', ]
    for i, sc in enumerate(scene):
        f_scene = open('data/nuscenes/image_scenes/'+sc['name']+'.csv', 'w')
        for s in sample:
            if s["scene_token"] == sc['token']:
                index0 = np.where(sd_sample_token == s['token'])
                for index in index0[0]:
                    _, view, _ = sample_data_new[index]['filename'].split('/')
                    index1 = np.where(ep_token == sample_data_new[index]['ego_pose_token'])[0][0]
                    if view == 'CAM_FRONT':
                        output[0] = sample_data_new[index]['filename']+','+str(ego_pose[index1]['translation'][0])+','+str(ego_pose[index1]['translation'][1])+','+str(ego_pose[index1]['translation'][2])+','+str(ego_pose[index1]['rotation'][0])+','+str(ego_pose[index1]['rotation'][1])+','+str(ego_pose[index1]['rotation'][2])+','+str(ego_pose[index1]['rotation'][3])+'\n'
                    if view == 'CAM_FRONT_RIGHT':
                        output[1] = sample_data_new[index]['filename']+','+str(ego_pose[index1]['translation'][0])+','+str(ego_pose[index1]['translation'][1])+','+str(ego_pose[index1]['translation'][2])+','+str(ego_pose[index1]['rotation'][0])+','+str(ego_pose[index1]['rotation'][1])+','+str(ego_pose[index1]['rotation'][2])+','+str(ego_pose[index1]['rotation'][3])+'\n'
                    if view == 'CAM_BACK_RIGHT':
                        output[2] = sample_data_new[index]['filename']+','+str(ego_pose[index1]['translation'][0])+','+str(ego_pose[index1]['translation'][1])+','+str(ego_pose[index1]['translation'][2])+','+str(ego_pose[index1]['rotation'][0])+','+str(ego_pose[index1]['rotation'][1])+','+str(ego_pose[index1]['rotation'][2])+','+str(ego_pose[index1]['rotation'][3])+'\n'
                    if view == 'CAM_BACK':
                        output[3] = sample_data_new[index]['filename']+','+str(ego_pose[index1]['translation'][0])+','+str(ego_pose[index1]['translation'][1])+','+str(ego_pose[index1]['translation'][2])+','+str(ego_pose[index1]['rotation'][0])+','+str(ego_pose[index1]['rotation'][1])+','+str(ego_pose[index1]['rotation'][2])+','+str(ego_pose[index1]['rotation'][3])+'\n'
                    if view == 'CAM_BACK_LEFT':
                        output[4] = sample_data_new[index]['filename']+','+str(ego_pose[index1]['translation'][0])+','+str(ego_pose[index1]['translation'][1])+','+str(ego_pose[index1]['translation'][2])+','+str(ego_pose[index1]['rotation'][0])+','+str(ego_pose[index1]['rotation'][1])+','+str(ego_pose[index1]['rotation'][2])+','+str(ego_pose[index1]['rotation'][3])+'\n'
                    if view == 'CAM_FRONT_LEFT':
                        output[5] = sample_data_new[index]['filename']+','+str(ego_pose[index1]['translation'][0])+','+str(ego_pose[index1]['translation'][1])+','+str(ego_pose[index1]['translation'][2])+','+str(ego_pose[index1]['rotation'][0])+','+str(ego_pose[index1]['rotation'][1])+','+str(ego_pose[index1]['rotation'][2])+','+str(ego_pose[index1]['rotation'][3])+'\n'
                for o in output:
                    f_scene.write(o)
        f_scene.close()
        print(i)

    '''
    i = 0
    f_scene = open('data/nuscenes/scenes/'+f'{i:03d}'+'.csv', 'w')

    s_0 = sample[0]
    for s in sample:
        if s["scene_token"] != s_0["scene_token"]:  # 判断是否切换scene了
            s_0 = s
            i = i+1
            f_scene.close()
            f_scene = open('data/nuscenes/scenes/'+f'{i:03d}'+'.csv', 'w')

        index0 = np.where(sd_sample_token == s['token'])
        for index in index0[0]:
            sample_data_new[index]['filename']
            index1 = np.where(ep_token == sample_data_new[index]['ego_pose_token'])[0][0]
            f_scene.write(sample_data_new[index]['filename']+','+str(ego_pose[index1]['translation'][0])+','+str(ego_pose[index1]['translation'][1])+'\n')
    
        #for sd in sample_data_new:
        #    if sd['sample_token'] == s['token']:
        #        for ep in ego_pose:
        #            if sd['ego_pose_token'] == ep['token']:
        #                f_scene.write(sd['filename']+','+str(ep['translation'][0])+','+str(ep['translation'][1])+'\n')
        
        print(i)
    f_scene.close()
    '''

if __name__ == '__main__':
    extract_image_position()

    filepaths = glob.glob('data/nuscenes/image_scenes/*.csv')
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(filename)
        if filename[:-4] in mini_train:
            shutil.copy(filepath, 'data/nuscenes/image_scenes/mini_train/'+filename)
        if filename[:-4] in mini_val:
            shutil.copy(filepath, 'data/nuscenes/image_scenes/mini_val/'+filename)

        if filename[:-4] in val:
            shutil.move(filepath, 'data/nuscenes/image_scenes/val/'+filename)
        else:
            shutil.move(filepath, 'data/nuscenes/image_scenes/train/'+filename)

