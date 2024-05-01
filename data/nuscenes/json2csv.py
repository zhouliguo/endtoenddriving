import json
import numpy as np
import glob
import os
import shutil

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

def extract_image_position():
    # 读取 JSON 文件
    with open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/scene.json', 'r') as f:
        scene = json.load(f)

    with open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/sample.json', 'r') as f:
        sample = json.load(f)

    with open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/sample_data.json', 'r') as f:
        sample_data = json.load(f)

    with open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/ego_pose.json', 'r') as f:
        ego_pose = json.load(f)

    with open('D:/datasets/nuscenes/trainval/v1.0-trainval_meta/v1.0-trainval/calibrated_sensor.json', 'r') as f:
        calibrated_sensor = json.load(f)

    image_filename = []
    image_sampletoken = []

    lidar_filename = []
    lidar_sampletoken = []
    lidar_egoposetoken = []
    lidar_calibratedtoken = []

    ego_pose_token = []
    ego_pose_translation = []
    ego_pose_rotation = []

    calibrated_token = []
    calibrated_translation = []
    calibrated_rotation = []

    for sd in sample_data:
        if sd['fileformat'] == 'jpg' and sd['is_key_frame'] == True:    # 把sample_data.json中的包含图片关键帧的记录取出来
            image_filename.append(sd['filename'])
            image_sampletoken.append(sd['sample_token'])

        if ('LIDAR_TOP' in sd['filename']) and sd['is_key_frame'] == True:    # 把sample_data.json中的包含LiDAR关键帧的记录取出来
            lidar_filename.append(sd['filename'])
            lidar_sampletoken.append(sd['sample_token'])
            lidar_egoposetoken.append(sd['ego_pose_token'])
            lidar_calibratedtoken.append(sd['calibrated_sensor_token'])

    for ep in ego_pose:
        ego_pose_token.append(ep['token'])
        ego_pose_translation.append(ep['translation'])
        ego_pose_rotation.append(ep['rotation'])

    for cs in calibrated_sensor:
        calibrated_token.append(cs['token'])
        calibrated_translation.append(cs['translation'])
        calibrated_rotation.append(cs['rotation'])

    image_filename = np.array(image_filename)
    image_sampletoken = np.array(image_sampletoken)

    lidar_filename = np.array(lidar_filename)
    lidar_sampletoken = np.array(lidar_sampletoken)
    lidar_egoposetoken = np.array(lidar_egoposetoken)
    lidar_calibratedtoken = np.array(lidar_calibratedtoken)

    ego_pose_token = np.array(ego_pose_token)
    ego_pose_translation = np.array(ego_pose_translation)
    ego_pose_rotation = np.array(ego_pose_rotation)

    calibrated_token = np.array(calibrated_token)
    calibrated_translation = np.array(calibrated_translation)
    calibrated_rotation = np.array(calibrated_rotation)

    scene_token = []
    scene_name = []
    for sc in scene:
        scene_token.append(sc['token'])
        scene_name.append(sc['name'])

    scenes_with_samples = [] #所有场景，每个场景包含其所有样本
    sample_len = 0
    for sc_t, sc_n in zip(scene_token, scene_name):
        f_scene = open('data/nuscenes/'+sc_n+'.csv', 'w')

        samples_in_scene = [] #在同一个场景里的所有样本
        for sa in sample:
            if sa['scene_token'] == sc_t:
                samples_in_scene.append(sa)
        sis_prev = samples_in_scene[0]
        for sis in samples_in_scene[1:]:
            if sis_prev['next'] != sis['token']:
                print('Wrong')
            sis_prev = sis
        sample_len = sample_len+len(samples_in_scene)
        #scenes_with_samples.append(samples_in_scene)

        for i, sis in enumerate(samples_in_scene):
            index0 = np.where(image_sampletoken == sis['token'])
            index1 = np.where(lidar_sampletoken == sis['token'])

            filename = image_filename[index0[0]]

            lep_token = lidar_egoposetoken[index1[0]]
            lc_token = lidar_calibratedtoken[index1[0]]
            
            index2 = np.where(ego_pose_token == lep_token)
            index3 = np.where(calibrated_token == lc_token)

            et = ego_pose_translation[index2[0]][0]
            er = ego_pose_rotation[index2[0]][0]

            ct = calibrated_translation[index3[0]][0]
            cr = calibrated_rotation[index3[0]][0]

            output = ['', '', '', '', '', '', ]
            for fn in filename:
                _, view, _ = fn.split('/')

                if view == 'CAM_FRONT':
                    output[0] = fn#+','+str(et[0])+','+str(et[1])+','+str(et[2])+','+str(er[0])+','+str(er[1])+','+str(er[2])+','+str(er[3])+','+str(ct[0])+','+str(ct[1])+','+str(ct[2])+','+str(cr[0])+','+str(cr[1])+','+str(cr[2])+','+str(cr[3])+'\n'
                if view == 'CAM_FRONT_RIGHT':
                    output[1] = fn#+','+str(et[0])+','+str(et[1])+','+str(et[2])+','+str(er[0])+','+str(er[1])+','+str(er[2])+','+str(er[3])+','+str(ct[0])+','+str(ct[1])+','+str(ct[2])+','+str(cr[0])+','+str(cr[1])+','+str(cr[2])+','+str(cr[3])+'\n'
                if view == 'CAM_BACK_RIGHT':
                    output[2] = fn#+','+str(et[0])+','+str(et[1])+','+str(et[2])+','+str(er[0])+','+str(er[1])+','+str(er[2])+','+str(er[3])+','+str(ct[0])+','+str(ct[1])+','+str(ct[2])+','+str(cr[0])+','+str(cr[1])+','+str(cr[2])+','+str(cr[3])+'\n'
                if view == 'CAM_BACK':
                    output[3] = fn#+','+str(et[0])+','+str(et[1])+','+str(et[2])+','+str(er[0])+','+str(er[1])+','+str(er[2])+','+str(er[3])+','+str(ct[0])+','+str(ct[1])+','+str(ct[2])+','+str(cr[0])+','+str(cr[1])+','+str(cr[2])+','+str(cr[3])+'\n'
                if view == 'CAM_BACK_LEFT':
                    output[4] = fn#+','+str(et[0])+','+str(et[1])+','+str(et[2])+','+str(er[0])+','+str(er[1])+','+str(er[2])+','+str(er[3])+','+str(ct[0])+','+str(ct[1])+','+str(ct[2])+','+str(cr[0])+','+str(cr[1])+','+str(cr[2])+','+str(cr[3])+'\n'
                if view == 'CAM_FRONT_LEFT':
                    output[5] = fn+','+str(et[0])+','+str(et[1])+','+str(et[2])+','+str(er[0])+','+str(er[1])+','+str(er[2])+','+str(er[3])+','+str(ct[0])+','+str(ct[1])+','+str(ct[2])+','+str(cr[0])+','+str(cr[1])+','+str(cr[2])+','+str(cr[3])+'\n'
            for o in output[:-1]:
                f_scene.write(o)
                f_scene.write(',')
            f_scene.write(output[-1])
        f_scene.close()
        print(sc_n)

if __name__ == '__main__':
    extract_image_position()

    filepaths = glob.glob('data/nuscenes/*.csv')
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(filename)
        if filename[:-4] in mini_train:
            shutil.copy(filepath, 'data/nuscenes/mini_train/'+filename)
        if filename[:-4] in mini_val:
            shutil.copy(filepath, 'data/nuscenes/mini_val/'+filename)

        if filename[:-4] in val:
            shutil.move(filepath, 'data/nuscenes/val/'+filename)
        else:
            shutil.move(filepath, 'data/nuscenes/train/'+filename)