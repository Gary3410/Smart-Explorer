import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import open3d as o3d
from sklearn.cluster import KMeans, estimate_bandwidth, DBSCAN
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from color_map import deepglobe_color_map as dcm
def deepglobe_color_map():
    colorize = np.zeros([17, 3], dtype=np.int64)
    colorize[0, :] = [184, 179, 168]
    colorize[1, :] = [255, 0, 0]
    colorize[2, :] = [255, 127, 0]
    colorize[3, :] = [255, 255, 0]
    colorize[4, :] = [0, 255, 0]
    colorize[5, :] = [0, 0, 255]
    colorize[6, :] = [38, 0, 51]
    colorize[7, :] = [148, 0, 211]
    colorize[8, :] = [128, 42, 42]
    colorize[9, :] = [188, 143, 143]
    colorize[10, :] = [34, 139, 34]
    colorize[11, :] = [255, 127, 80]
    colorize[12, :] = [8, 46, 84]
    colorize[13, :] = [176, 48, 96]
    colorize[14, :] = [51, 161, 201]
    colorize[15, :] = [218, 112, 214]
    colorize[16, :] = [244, 164, 96]

    return colorize

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid[np.newaxis,:]
    m = np.max(np.sqrt(np.sum(np.power(pc,2), axis=1)),axis=0)
    pc = pc / m[np.newaxis,np.newaxis]
    return pc

def numpy2Image(prediction):
    colorize = deepglobe_color_map()
    label = colorize[prediction-100, :].reshape([prediction.shape[0], prediction.shape[1], 3])
    return label

def numpy2Image_1(prediction):
    colorize = deepglobe_color_map()
    label = colorize[prediction, :].reshape([prediction.shape[0], prediction.shape[1], 3])
    return label
def show_prediction(prediction, prediction_path):
    prediction_img = numpy2Image(prediction)
    image = Image.fromarray(np.uint8(prediction_img))
    image.save(prediction_path)

def show_prediction_1(prediction, prediction_path):
    prediction_img = numpy2Image_1(prediction)
    image = Image.fromarray(np.uint8(prediction_img))
    image.save(prediction_path)

def show_3D(data, id):
    #data[:, :3] = pc_normalize(data[:, :3])
    data = data[data[:, 3]>0]
    print(data.shape)
    colormap = []
    lab = np.asarray([[184, 179, 168],
                    [255, 0, 0],
                    [255, 127, 0],
                    [255, 255, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [38, 0, 51],
                    [148, 0, 211]])/255.0
    colormap = [[] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        colormap[i] = lab[int(data[i, 3])]
    #plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colormap, s=20, marker='.')  # , cmap='plasma')
    plt.show()
    plt.close()

def show_3D_GT(surface_pts, color_pts, id):
    lab = np.asarray([[184, 179, 168],
                      [255, 0, 0],
                      [255, 127, 0],
                      [255, 255, 0],
                      [0, 255, 0],
                      [0, 0, 255],
                      [38, 0, 51],
                      [148, 0, 211]]) / 255.0
    lab_all = np.asarray([[184, 179, 168],
                      [255, 0, 0],
                      [255, 127, 0],
                      [255, 255, 0],
                      [0, 255, 0],
                      [0, 0, 255],
                      [38, 0, 51],
                      [148, 0, 211]])
    ax = plt.subplot(111, projection='3d')

    ax.view_init(elev=30, azim=-60)

    print("sur", surface_pts.shape)
    print("col", color_pts.shape)

    ax.scatter(surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2], c=color_pts, s=20, marker='.')  # , cmap='plasma')
    plt.show()
    plt.close()
def show_3D_single(data, id, cls_id, test_time=0):
    # data[:, :3] = pc_normalize(data[:, :3])
    data = data[data[:, 3] == cls_id]

    colorize = dcm()
    colorize = colorize / 255.0
    colormap = [[] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        colormap[i] = colorize[int(data[i, 3])]
    # plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')


    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colormap, s=20, marker='.')  # , cmap='plasma')
    plt.show()
    plt.close()
def point_filter(data_all, nc):
    filter_data = []
    for cls in range(1, nc):
        data = data_all[data_all[:, 3]==cls]
        if data.shape[0] == 0:
            continue
        data_xyz = data[:, :3]
        #ms = KMeans(n_clusters=4, random_state=0).fit(data_xyz)
        bandwidth = estimate_bandwidth(data_xyz, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data_xyz)

        labels = ms.labels_
        #cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        number_list = []
        for i in range(n_clusters_):
            my_members = labels == i
            my_data = data[my_members]
            number_list.append(my_data.shape[0])
        print("each_number", number_list)
        max_value = min(number_list)
        max_idx = number_list.index(max_value)
        filter_data_one = data[labels != max_idx]
        filter_data.append(filter_data_one)
    return np.concatenate(filter_data, axis=0)

def point_filter_DBSCAN(data_all, nc):
    filter_data = []
    for cls in range(1, nc):
        data = data_all[data_all[:, 3]==cls]
        if data.shape[0] == 0:
            continue
        data_xyz = data[:, :3]
        data_xyz_normal = StandardScaler().fit_transform(data_xyz)
        print("start filter")
        db = DBSCAN(eps=0.3, min_samples=10).fit(data_xyz_normal)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        #unique_labels = set(labels)
        print("end filter")
        filter_data_one = data[labels != -1]
        filter_data.append(filter_data_one)
    return np.concatenate(filter_data, axis=0)

def points2depth(depth_img, points, camera_intrinsics):

    x1 = 0
    y1 = 0
    x2 = 640
    y2 = 480
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]
    depth_img = np.full((im_h, im_w), np.inf)
    pix_x = points[:, 0]
    pix_y = points[:, 1]
    pix_z = points[:, 2]
    pix_x = np.reshape(pix_x, (im_h, im_w))
    pix_y = np.reshape(pix_y, (im_h, im_w))
    pix_z = np.reshape(pix_z, (im_h, im_w))
    cam_pts_x = np.divide(pix_x*camera_intrinsics[0][0], pix_z)
    cam_pts_y = np.divide(pix_y*camera_intrinsics[1][1], pix_z)
    cam_pts_x = cam_pts_x + camera_intrinsics[0][2]
    cam_pts_y = cam_pts_y + camera_intrinsics[1][2]
    cam_pts_x.shape = (1, im_h*im_w)
    cam_pts_y.shape = (1, im_h*im_w)
    cam_pts_x = np.absolute(cam_pts_x)
    cam_pts_y = np.absolute(cam_pts_y)
    cam_xy = np.concatenate((cam_pts_x, cam_pts_y), axis=0)
    cam_xy = np.around(cam_xy)
    index = np.where((cam_xy[0, :] > x1) & (cam_xy[0, :] < x2))
    cam_xy = cam_xy[:, index]
    cam_xy = cam_xy.reshape([2, -1])
    index = np.where((cam_xy[1, :] > y1) & (cam_xy[1, :] < y2))
    cam_xy = cam_xy[:, index]
    cam_xy = cam_xy.reshape([2, -1])
    print("cam_xy", cam_xy.shape)
    for i in range(cam_xy.shape[1]):
        depth_img[int(cam_xy[1, i])][int(cam_xy[0, i])] = pix_z[int(cam_xy[1, i])][int(cam_xy[0, i])]
    print(depth_img.shape)
    return depth_img

def get_different(surface_pts, depth_img, cam_post):
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]
    add_v = np.ones([surface_pts.shape[0], 1])
    pts = np.concatenate((surface_pts, add_v), axis=1)
    cam_pose_inv = np.linalg.inv(cam_post)
    pts = np.transpose(np.dot(cam_pose_inv, np.transpose(pts)))
    pix_z = pts[:, 2]
    depth_img_one = np.reshape(pix_z, (im_h, im_w))
    return depth_img_one

def get_point2depth(surface_pts, depth_img, cam_post, camera_intrinsics):
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]
    x1 = 0
    y1 = 0
    x2 = 640
    y2 = 480
    depth_img = np.full((im_h, im_w), np.nan)
    depth_img_1 = np.full((im_h, im_w), 1000)
    depth_img_2 = np.full((im_h, im_w), 100)
    depth_img_3 = np.full((im_h, im_w), 10)
    add_v = np.ones([surface_pts.shape[0], 1])
    pts = np.concatenate((surface_pts, add_v), axis=1)
    cam_pose_inv = np.linalg.inv(cam_post)

    pts = np.transpose(np.dot(cam_pose_inv, np.transpose(pts)))
    # pts_depth [x, y, z]
    pts_depth = np.transpose(np.dot(camera_intrinsics, np.transpose(pts[:, :3])))
    cam_pts_x = np.transpose(pts_depth[:, 0]/pts_depth[:, 2])
    cam_pts_y = np.transpose(pts_depth[:, 1]/pts_depth[:, 2])
    cam_pts_x.shape = (1, im_h * im_w)
    cam_pts_y.shape = (1, im_h * im_w)
    cam_pts_z = np.reshape(pts_depth[:, 2], (1, im_h*im_w))

    cam_xy = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=0)


    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
    grid_z0 = griddata(cam_xy[:2, :].T, cam_xy[2, :].T, (pix_x, pix_y), method='nearest') #linear\nearest
    return grid_z0

def insert_linear_pos(data, resize=None, x_scale=1, y_scale=1):
    m_, n_ = data.shape
    n_new = n_
    m_new = m_
    n_scale, m_scale = n_ / n_new, m_ / m_new
    m_indxs = np.repeat(np.arange(m_new), n_new).reshape(m_new, n_new)
    n_indxs = np.array(list(range(n_new)) * m_new).reshape(m_new, n_new)

    m_indxs_c = (m_indxs + 0.5) * m_scale - 0.5
    n_indxs_c = (n_indxs + 0.5) * n_scale - 0.5

    m_indxs_c[np.where(m_indxs_c < 0)] = 0.0
    n_indxs_c[np.where(n_indxs_c < 0)] = 0.0


    m_indxs_c_down = m_indxs_c.astype(int)
    n_indxs_c_down = n_indxs_c.astype(int)
    m_indxs_c_up = m_indxs_c_down + 1
    n_indxs_c_up = n_indxs_c_down + 1

    m_max = m_ - 1
    n_max = n_ - 1
    m_indxs_c_up[np.where(m_indxs_c_up > m_max)] = m_max
    n_indxs_c_up[np.where(n_indxs_c_up > n_max)] = n_max


    pos_0_0 = data[m_indxs_c_down, n_indxs_c_down].astype(int)
    pos_0_1 = data[m_indxs_c_up, n_indxs_c_down].astype(int)
    pos_1_1 = data[m_indxs_c_up, n_indxs_c_up].astype(int)
    pos_1_0 = data[m_indxs_c_down, n_indxs_c_up].astype(int)

    m, n = np.modf(m_indxs_c)[0], np.modf(n_indxs_c)[0]
    return pos_0_0, pos_0_1, pos_1_1, pos_1_0, m, n

def linear_insert_1color(img_dt, resize, fx=None, fy=None):
    pos_0_0, pos_0_1, pos_1_1, pos_1_0, m, n = insert_linear_pos(img_dt=img_dt, resize=resize, x_scale=fx, y_scale=fy)
    a = (pos_1_0 - pos_0_0)
    b = (pos_0_1 - pos_0_0)
    c = pos_1_1 + pos_0_0 - pos_1_0 - pos_0_1
    return np.round(a * n + b * m + c * n * m + pos_0_0).astype(int)

def linear_insert(img_dt, resize, fx=None, fy=None):

    if len(img_dt.shape) == 3:
        out_img0 = linear_insert_1color(img_dt[:,:,0], resize=resize, fx=fx, fy=fy)
        out_img1 = linear_insert_1color(img_dt[:,:,1], resize=resize, fx=fx, fy=fy)
        out_img2 = linear_insert_1color(img_dt[:,:,2], resize=resize, fx=fx, fy=fy)
        out_img_all = np.c_[out_img0[:,:,np.newaxis], out_img1[:,:,np.newaxis], out_img2[:,:,np.newaxis]]
    else:
        out_img_all = linear_insert_1color(img_dt, resize=resize, fx=fx, fy=fy)
    return out_img_all.astype(np.uint8)

