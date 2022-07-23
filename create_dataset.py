import pybullet_data as pd
import pybullet as p

import os
import numpy as np
import cv2
import torch
import argparse
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm
import shutil


def save_img(img, base_path_0, time, view_id):
    color_img_print = np.ascontiguousarray(img)
    rgb_path = os.path.join(base_path_0, "rgb_img")
    img_path_2 = os.path.join(rgb_path, str(time) +"_" +str(view_id)+ '.png')
    cv2.imwrite(img_path_2, color_img_print)


def create_box_bullet(scale, pos):
    l = scale[0]
    w = scale[1]
    h = scale[2]

    x = pos[0]
    y = pos[1]
    z = pos[2]
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[l / 2, w / 2, h / 2],
        rgbaColor=[128, 128, 128, 1])

    collison_box_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[l / 2, w / 2, h / 2]
    )

    wall_id = p.createMultiBody(
        baseMass=10000,
        baseCollisionShapeIndex=collison_box_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[x, y, z]
    )
    return wall_id

def create_wall():
    xmin = -0.85
    ymin = -0.2
    xmax = -0.25
    ymax = 0.2
    TopHeight = 0.5

    wall_id_1 = create_box_bullet([xmax - xmin, 0.1, TopHeight], [(xmax - xmin) / 2 + xmin, ymin - 0.05, TopHeight / 2])
    wall_id_2 = create_box_bullet([xmax - xmin, 0.1, TopHeight], [(xmax - xmin) / 2 + xmin, ymax + 0.05, TopHeight / 2])
    wall_id_3 = create_box_bullet([0.1, ymax - ymin, TopHeight], [xmin - 0.05, (ymax - ymin) / 2 + ymin, TopHeight / 2])
    wall_id_4 = create_box_bullet([0.1, ymax - ymin, TopHeight], [xmax + 0.05, (ymax - ymin) / 2 + ymin, TopHeight / 2])

    return wall_id_1, wall_id_2, wall_id_3, wall_id_4

def main():

    p.connect(p.DIRECT)  # p.DIRECT
    number_list = [5, 6, 7, 10, 11, 12, 15, 16, 18, 20]

    viewMat_list = [
        [1.0, 0.0, -0.0, 0.0, -0.0, 1.0, -0.00017452123574912548, 0.0, 0.0, 0.00017452123574912548, 1.0, 0.0, 0.5,
         -7.417152664856985e-05, -0.4266100227832794, 1.0],
        [1.0, 0.0, -0.0, 0.0, -0.0, 0.8660240173339844, -0.5000023245811462, 0.0, 0.0, 0.5000023245811462,
         0.8660240173339844, 0.0, 0.5, 2.925097942352295e-05, -0.426557332277298, 1.0],
        [0.017451846972107887, 0.8658873438835144, -0.49993473291397095, 0.0, -0.9998477697372437, 0.015113634057343006,
         -0.00872611254453659, 0.0, 0.0, 0.5000109076499939, 0.8660191893577576, 0.0, 0.012434440664947033,
         0.4329407513141632, -0.6765085458755493, 1.0],
        [-0.999847412109375, 0.015133202075958252, -0.008737090043723583, 0.0, -0.017474284395575523,
         -0.8658948540687561, 0.49992072582244873, 0.0, 0.0, 0.4999971091747284, 0.8660269975662231, 0.0,
         -0.4962104260921478, 0.007570326328277588, -0.4309096336364746, 1.0],
        [-0.01745261251926422, -0.8658953309059143, 0.4999208450317383, 0.0, 0.9998477697372437, -0.015114436857402325,
         0.008726253174245358, 0.0, 0.0, 0.49999701976776123, 0.8660271763801575, 0.0, -0.005017626099288464,
         -0.43294382095336914, -0.1765807718038559, 1.0]]

    projMat = [0.74999994, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.00002003, -1.0, 0.0, 0.0, -0.0200002, 0.0]

    name_list = []

    root_base_path = os.getcwd()
    name_list_path = os.path.join(root_base_path, "objects", "name_list.txt")
    base_path_img = os.path.join(root_base_path, "dataset")
    f = open(name_list_path)
    for line in f:
        name_list.append(line.strip())
    f.close()
    base_path = os.path.join(root_base_path, "objects", "urdf27")

    workspace_limits = np.asarray([[-0.7, -0.50], [-0.12, 0.12], [0.1, 0.5]])
    time_id = 0

    for test_time_all in range(2):

        p.setGravity(0.000000, 0.000000, -9.800000)
        planeId = p.loadURDF(os.path.join(pd.getDataPath(), "plane.urdf"))

        wall_id_1, wall_id_2, wall_id_3, wall_id_4 = create_wall()
        number_one = random.choice(number_list)
        object_name_list = random.sample(name_list, number_one)
        flags = p.URDF_USE_INERTIA_FROM_FILE
        # 根据标注的object_name_list生成场景
        obj_id_list = []
        object_mesh_id = []
        scale_list = []

        for object_idx in range(number_one):
            object_name = np.random.choice(object_name_list)
            obj_index = name_list.index(object_name)
            # 获取为真正意义上的label
            object_mesh_id.append(obj_index + 1)

            drop_x = (workspace_limits[0][1] - workspace_limits[0][0]) * np.random.random_sample() + \
                     workspace_limits[0][0]
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0]) * np.random.random_sample() + \
                     workspace_limits[1][0]


            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]

            obj_id = p.loadURDF(os.path.join(base_path, object_name), basePosition=[drop_x, drop_y, 0.22],
                baseOrientation=[drop_x, drop_y, 0.22, object_orientation[0]], flags=flags)


            for _ in range(480):
                p.stepSimulation()
            obj_id_list.append(obj_id)


        # 移除墙壁
        p.removeBody(wall_id_1)
        p.removeBody(wall_id_2)
        p.removeBody(wall_id_3)
        p.removeBody(wall_id_4)

        for _ in range(200):
            p.stepSimulation()

        object_mesh_id.insert(0, 0)


        for view_id in tqdm(range(5)):
            viewMat = viewMat_list[view_id]
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=1440, height=1024, viewMatrix=viewMat,
                projectionMatrix=projMat, flags=flags)


            color_img = np.asarray(rgbImg)
            color_img = color_img[:, :, :3]
            color_img = color_img.astype(np.uint8)


            color_img_print = color_img.copy()
            color_img_print = color_img_print[:, :, ::-1]  # RGB-->BGR


            save_img(color_img_print, base_path_img, time_id, view_id)

            class_id = np.array(object_mesh_id).reshape(-1, 1)

            segImg = segImg - 4
            segImg[segImg <= 0] = 0

            segImg_copy = segImg.copy()

            seg_label = class_id[segImg, :].reshape([segImg.shape[0], segImg.shape[1]])



            ints_image = Image.fromarray(np.uint8(segImg_copy))
            label_img = Image.fromarray(np.uint8(seg_label))

            ints_path = os.path.join(base_path_img, "ints_img")
            label_path = os.path.join(base_path_img, "label_img")

            img_path_1 = os.path.join(ints_path, str(time_id) + "_" + str(view_id) + '.png')
            img_path_2 = os.path.join(label_path, str(time_id) + "_" + str(view_id) + '.png')

            ints_image.save(img_path_1)
            label_img.save(img_path_2)

        time_id = time_id + 1

        p.resetSimulation()

if __name__ == '__main__':
    # Parse arguments
    main()

