import pickle
import os

#mini_infos = os.listdir('D:/datasets/OpenScene/dataset/openscene-v1.1/meta_datas/mini')
#mini_infos = [os.path.join('D:/datasets/OpenScene/dataset/openscene-v1.1/meta_datas/mini', each) for each in mini_infos if each.endswith('.pkl')]

#train_paths = mini_infos[:int(len(mini_infos) * 0.85)]
#val_paths = mini_infos[int(len(mini_infos) * 0.85):]

train_paths = os.listdir('D:/datasets/OpenScene/dataset/openscene-v1.1/meta_datas/trainval')
train_paths = [os.path.join('D:/datasets/OpenScene/dataset/openscene-v1.1/meta_datas/trainval', each) for each in train_paths if each.endswith('.pkl')]

val_paths = os.listdir('D:/datasets/OpenScene/dataset/openscene-v1.1/meta_datas/test')
val_paths = [os.path.join('D:/datasets/OpenScene/dataset/openscene-v1.1/meta_datas/test', each) for each in val_paths if each.endswith('.pkl')]

for file in train_paths:
    with open(file, 'rb') as f:
        train_infos=pickle.load(f)
        #f_csv = open('data/openscene/train/'+os.path.basename(file)+'.csv', 'w')
        sc = 0
        for ti in train_infos:
            if ti['frame_idx'] == 0:
                f_csv = open('data/openscene/train/'+ti['scene_name']+'.csv', 'w')
                sc = sc+1
            e2gt = ti['ego2global_translation']
            e2gr = ti['ego2global_rotation']
            l2et = ti['lidar2ego_translation']
            l2er = ti['lidar2ego_rotation']
            f_csv.write(ti['cams']['CAM_F0']['data_path']+',')
            f_csv.write(ti['cams']['CAM_L0']['data_path']+',')
            f_csv.write(ti['cams']['CAM_R0']['data_path']+',')
            f_csv.write(ti['cams']['CAM_L1']['data_path']+',')
            f_csv.write(ti['cams']['CAM_R1']['data_path']+',')
            f_csv.write(ti['cams']['CAM_L2']['data_path']+',')
            f_csv.write(ti['cams']['CAM_R2']['data_path']+',')
            f_csv.write(ti['cams']['CAM_B0']['data_path']+',')
            f_csv.write(str(e2gt[0])+','+str(e2gt[1])+','+str(e2gt[2])+',')
            f_csv.write(str(e2gr[0])+','+str(e2gr[1])+','+str(e2gr[2])+','+str(e2gr[3])+',')
            f_csv.write(str(l2et[0])+','+str(l2et[1])+','+str(l2et[2])+',')
            f_csv.write(str(l2er[0])+','+str(l2er[1])+','+str(l2er[2])+','+str(l2er[3])+'\n')
        f_csv.close()

val_infos = []
for file in val_paths:
    with open(file, 'rb') as f:
        val_infos=pickle.load(f)
        #f_csv = open('data/openscene/val/'+os.path.basename(file)+'.csv', 'w')
        sc = 0
        for ti in val_infos:
            if ti['frame_idx'] == 0:
                f_csv = open('data/openscene/val/'+ti['scene_name']+'.csv', 'w')
                sc = sc+1
            e2gt = ti['ego2global_translation']
            e2gr = ti['ego2global_rotation']
            l2et = ti['lidar2ego_translation']
            l2er = ti['lidar2ego_rotation']
            f_csv.write(ti['cams']['CAM_F0']['data_path']+',')
            f_csv.write(ti['cams']['CAM_L0']['data_path']+',')
            f_csv.write(ti['cams']['CAM_R0']['data_path']+',')
            f_csv.write(ti['cams']['CAM_L1']['data_path']+',')
            f_csv.write(ti['cams']['CAM_R1']['data_path']+',')
            f_csv.write(ti['cams']['CAM_L2']['data_path']+',')
            f_csv.write(ti['cams']['CAM_R2']['data_path']+',')
            f_csv.write(ti['cams']['CAM_B0']['data_path']+',')
            f_csv.write(str(e2gt[0])+','+str(e2gt[1])+','+str(e2gt[2])+',')
            f_csv.write(str(e2gr[0])+','+str(e2gr[1])+','+str(e2gr[2])+','+str(e2gr[3])+',')
            f_csv.write(str(l2et[0])+','+str(l2et[1])+','+str(l2et[2])+',')
            f_csv.write(str(l2er[0])+','+str(l2er[1])+','+str(l2er[2])+','+str(l2er[3])+'\n')
        f_csv.close()
