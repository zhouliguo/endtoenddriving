import cv2
import glob

root = 'D:/datasets/nuscenes/trainval/'
filepaths = glob.glob('data/nuscenes/image_scenes/train/*.csv')

for filepath in filepaths:
    f = open(filepath, 'r')
    imagepaths = f.readlines()
    for imagepath in imagepaths[::6]:
        imagepath = imagepath.split(',')[0]
        image = cv2.imread(root+imagepath)
        cv2.imshow('image', image)
        cv2.waitKey(10)

