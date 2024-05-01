import shutil
import argparse
import threading
from nuimages import NuImages
from utils import category_to_detection_name


def run_process(
        nu_root='/data/sets/nuimages',
        yl_root="/data/sets/yolo",
        data_version="v1.0-mini",
        data_type="val",
        processor=2
):

    global dataset_root, yolo_images, yolo_labels

    dataset_root = nu_root
    yolo_images = yl_root + "images/" + data_type + "/"
    yolo_labels = yl_root + "labels/" + data_type + "/"

    for i in range(processor):
        p = Processor(i, processor, dataset_root, data_version)
        p.start()

class Processor(threading.Thread):
    def __init__(self, k, total_processor, nu_root, data_version):
        super(Processor, self).__init__()
        self.idx = k
        self.ts = total_processor
        self.nuim = NuImages(dataroot=nu_root, version=data_version, verbose=False, lazy=True)
        self.deal_length = int(len(self.nuim.sample_data) / total_processor)
        self.obj_list = ['animal','human.pedestrian.adult','human.pedestrian.child', 'human.pedestrian.construction_worker',
                         'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer','human.pedestrian.stroller',
                        'human.pedestrian.wheelchair','movable_object.barrier', 'movable_object.debris',
                        'movable_object.pushable_pullable', 'movable_object.trafficcone','static_object.bicycle_rack',
                        'vehicle.bicycle','vehicle.bus.bendy', 'vehicle.bus.rigid','vehicle.car',
                         'vehicle.construction','vehicle.emergency.ambulance','vehicle.emergency.police','vehicle.motorcycle','vehicle.trailer','vehicle.truck']

    def xyxy2xywh(self, xmin, ymin, xmax, ymax):
        x = (xmax + xmin) / 2 / 1600
        y = (ymax + ymin) / 2 / 900
        w = (xmax - xmin) / 1600
        h = (ymax - ymin) / 900

        return x, y, w, h

    def merge_name(self,name):
        if name == 'animal':  # 动物
            return '0'
        elif name == 'human.pedestrian.child':  # 幼童
            return '1'
        elif name in ['human.pedestrian.adult', 'human.pedestrian.construction_worker',
                            'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer',
                            'human.pedestrian.stroller', 'human.pedestrian.wheelchair']:  # 行人
            return '2'
        elif name in ['movable_object.barrier', 'movable_object.debris',
                            'movable_object.pushable_pullable', 'static_object.bicycle_rack']:  # 障碍
            return '3'
        elif name == 'movable_object.trafficcone':  # 雪糕筒
            return '4'
        elif name == 'vehicle.bicycle':  # 自行车
            return '5'
        elif name in ['vehicle.bus.bendy', 'vehicle.bus.rigid']:  # bus
            return '6'
        elif name in ['vehicle.emergency.ambulance', 'vehicle.emergency.police',
                            'vehicle.car']:
            return '7'
        elif name == 'vehicle.motorcycle':  # 摩托车
            return '8'
        elif name == 'vehicle.construction':  # 工程车
            return '9'
        elif name == 'vehicle.trailer':  # 拖车
            return '10'
        elif name == 'vehicle.truck':  # 货车
            return '11'
        
    def merge_name1(self, name):
        if name == 'barrier':  # 障碍物
            return '0'
        elif name == 'bicycle':  # 自行车
            return '1'
        elif name == 'bus':  # 大客车、公交车
            return '2'
        elif name == 'car':  # 小汽车
            return '3'
        elif name == 'construction_vehicle':  # 工程车
            return '4'
        elif name == 'motorcycle':  # 摩托车
            return '5'
        elif name == 'pedestrian':  # 行人
            return '6'
        elif name == 'traffic_cone':    # 雪糕筒
            return '7'
        elif name == 'trailer':  # 拖车
            return '8'
        elif name == 'truck':  # 卡车
            return '9'

    def process(self):
        start = self.idx * self.deal_length
        end = (self.idx + 1) * self.deal_length
        if self.idx == self.ts - 1:
            end = len(self.nuim.sample_data)
        for item in self.nuim.sample_data[start:end]:
            if item['is_key_frame'] == True:
                stamp = str(item['timestamp'])
                shutil.copy(dataset_root+item['filename'], yolo_images+stamp+'.jpg')
                obj_ann, _ = self.nuim.list_anns(item['sample_token'])
                with open(yolo_labels+stamp+'.txt','w+') as f:
                    for obj in obj_ann:
                        txt_line = ''
                        single_obj = self.nuim.get("object_ann",obj)
                        category = self.nuim.get("category", single_obj['category_token'])
                        #if category['name'] not in self.obj_list:
                        #    continue
                        #txt_line += self.merge_name(category['name']) + " "
                        name = category_to_detection_name(category['name'])
                        if name is None:
                            continue
                        txt_line += self.merge_name1(name) + " "
                        [*cordinates] = self.xyxy2xywh(*single_obj['bbox'])
                        for idx,value in enumerate(cordinates):
                            if idx != len(cordinates)-1:
                                txt_line += str(value) + ' '
                            else:
                                txt_line += str(value) + '\n'
                        f.write(txt_line)

    def run(self):
        self.process()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nu-root', type=str, default= 'D:/datasets/nuimages/', help='Nuscenes dataset path')
    parser.add_argument('--yl-root', type=str, default="D:/datasets/nuimages/", help='yolo path')
    parser.add_argument('--data-version', type=str, default="v1.0-train", help='dataset version, v1.0-train/val/test/mini')
    parser.add_argument('--data-type', type=str, default="train", help='the target yolo dir name, eg train/valid')
    parser.add_argument('--processor', type=int, default=2, help='number of threads')
    opt = parser.parse_args()
    return opt


def main(opt):
    run_process(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)