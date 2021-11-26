#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:29:44 2021

@author: mattwear
"""

### Pose Analysis Functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
import cv2
import re
import json
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
              [6, 8], [8, 9]]

def plot3D(ax, points, edges, marker_size = 100):
    ax.grid(False)
    oo = 1e10
    xmax,ymax,zmax = -oo,-oo,-oo
    xmin,ymin,zmin = oo, oo, oo
    #edges = mpii_edges
    c='b'
    marker = 'o'
    points = points.reshape(-1, 3)
    x, y, z = np.zeros((3, points.shape[0]))
    for j in range(points.shape[0]):
        x[j] = points[j, 0].copy()
        y[j] = points[j, 2].copy()
        z[j] = -points[j, 1].copy()
        xmax = max(x[j], xmax)
        ymax = max(y[j], ymax)
        zmax = max(z[j], zmax)
        xmin = min(x[j], xmin)
        ymin = min(y[j], ymin)
        zmin = min(z[j], zmin)
    ax.scatter(x, y, z, s = marker_size, c = c, marker = marker)
    for e in edges:
        ax.plot(x[e], y[e], z[e], c = c)
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
        
#2D plot of the 3D body pose in the x-y plane (ignoring z-axis)
def plot2D(ax, pose_3d, mpii_edges):
    for e in range(len(mpii_edges)):
        ax.plot(pose_3d[mpii_edges[e]][:, 0], -1*pose_3d[mpii_edges[e]][:, 1])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
def plot2D3DPose(array_id, save_df, poses_2d, poses_3d, img_path, mpii_edges):
    img_file = save_df['file'][array_id]
    image = importImage(img_path + img_file)
    pose_2d = pose_to_matrix(poses_2d[array_id])
    pose_3d = pose_to_matrix(poses_3d[array_id][:-1])
    print('Array ID: ' + str(array_id))
    print("File Name: " + img_file)

    fig = plt.figure(figsize=(15, 5))
    #fig.patch.set_visible(False)
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(image)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    ax.set_title('(a) Input Image', y=-0.14)

    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(image)
    for e in range(len(mpii_edges)):
        ax.plot(pose_2d[mpii_edges[e]][:, 0], pose_2d[mpii_edges[e]][:, 1], c='b', lw=3, marker='o')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    ax.set_title('(b) 2D Pose Estimation', y=-0.14)

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    plot3D(ax, pose_3d, mpii_edges, marker_size=30)
    ax.set_title('(c) 3D Pose Estimation (CVI)', y=-0.23)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_zticks([])
    
    ax = fig.add_subplot(1, 4, 4)
    plot2D(ax, pose_3d, mpii_edges)
    ax.set_title('(d) 3D Pose Estimation (CVI) 2D Projection', y=-0.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    #plt.savefig('viz/poseEstimationExample.png', dpi=500)
    plt.show()
        
def pose_to_matrix(pose):
    if len(pose) == 48:
        pose_matrix = pose.reshape(16, 3)
    else:
        pose_matrix = pose.reshape(16, 2)
    return pose_matrix

def importImage(img):
    #Import image
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    return image

def rotatePose(pose_3d, theta):
    #Rotate body pose by theta degrees around the y axis
    #Input: pose_3d - 16x3 array representing the coordinates of the body pose
    #Returns: 16x3 array of rotated body pose coordinates
    radian = math.radians(theta)

    rotation_matrix = np.array([[np.cos(radian), 0, np.sin(radian)],
                                [0, 1, 0],
                                [-np.sin(radian), 0, np.cos(radian)]])

    rotated_pose = np.zeros((len(pose_3d), 3))
    for i in range(len(pose_3d)):
        rotated_pose[i] = rotation_matrix @ pose_3d[i]
    return rotated_pose

def hipWidth(pose_3d):
    #Input: pose_3d - 16x3 np array representing a single 3D body pose
    #Returns euclidean distance in x-y space of the two hip joints
    #Indices of both hip locations are 2 and 3.
    return np.linalg.norm(pose_3d[3][:2]-pose_3d[2][:2])

def cameraInvariantPose(pose_3d):
    # Function to get the optimal rotated pose
    best_pose = pose_3d
    max_hip_width = hipWidth(pose_3d)
    theta_ranges = list(range(10, 100, 10)) + list(range(270, 360, 10))

    for theta in theta_ranges:
        rotated_pose = rotatePose(pose_3d, theta=theta)
        hip_width = hipWidth(rotated_pose)
        if hip_width > max_hip_width:
            best_pose = rotated_pose
            max_hip_width = hip_width
    return best_pose

def flipBehindPoses(cvi_arr):
    #Rotates the poses that are photographed from behind 180 degrees
    sets_3d_cvi = np.zeros(cvi_arr.shape)
    for i in range(len(sets_3d_cvi)):
        #Rotate the photos from behind by 180 degrees
        pose = pose_to_matrix(cvi_arr[i])
        if pose[10][0] > pose[15][0]: #if RHx > LHx
            #print("This is a photo from behind")
            #Rotate this pose 180 degrees
            new_pose = rotatePose(pose, 180).flatten()
        else:
            #print("This is a photo from in front")
            new_pose = pose.flatten()
        sets_3d_cvi[i] = new_pose
    return sets_3d_cvi

def cameraInvariantDataset(raw_poses):
    #Converts the raw body point dataset to a cleaned camera-invariant one
    cleaned_pose_arr = raw_poses.copy()
    for i in range(len(raw_poses)):
        pose_3d = pose_to_matrix(raw_poses[i])
        best_pose = cameraInvariantPose(pose_3d)
        cleaned_pose_arr[i] = best_pose.flatten()
    return cleaned_pose_arr

def getFreezeFrame(shots, shot_id):
    onevone = shots.copy()
    ### Plot a shooting Situation
    #onevone['shot_freeze_frame'][shot_id][0]['position']['name']
    shooter_x = onevone['location'][shot_id][0]
    shooter_y = onevone['location'][shot_id][1]

    num_players = len(onevone['shot_freeze_frame'][shot_id])
    is_gk = np.zeros(num_players)
    is_teammate = np.zeros(num_players)
    freeze_frame_x = np.zeros(num_players)
    freeze_frame_y = np.zeros(num_players)
    for i in range(num_players):
        freeze_frame_x[i] = onevone['shot_freeze_frame'][shot_id][i]['location'][0]
        freeze_frame_y[i] = onevone['shot_freeze_frame'][shot_id][i]['location'][1]
        is_gk[i] = onevone['shot_freeze_frame'][shot_id][i]['position']['name'] == 'Goalkeeper'
        is_teammate[i] = onevone['shot_freeze_frame'][shot_id][i]['teammate']

    attacking_team_x = freeze_frame_x[is_teammate.astype(bool)]
    attacking_team_y = freeze_frame_y[is_teammate.astype(bool)]
    defending_team_x = freeze_frame_x[~ is_teammate.astype(bool)]
    defending_team_y = freeze_frame_y[~ is_teammate.astype(bool)]
    gk_x = freeze_frame_x[is_gk.astype(bool)]
    gk_y = freeze_frame_y[is_gk.astype(bool)]
    return shooter_x,shooter_y,attacking_team_x,attacking_team_y,defending_team_x,defending_team_y,gk_x,gk_y,is_gk

def distance_to_goal(shooter_x, shooter_y, goal_x = 120, goal_y=40):
    return np.linalg.norm(np.array([shooter_x,shooter_y])-np.array([goal_x,goal_y]))

def goal_angle(shooter_x, shooter_y, goal_x = 120, goal_y=40):
    return math.degrees(math.atan(np.abs(goal_y-shooter_y) / np.abs(goal_x - shooter_x)))

def getPhotoID(df):
    #Extract photo_id
    photo_id = []
    for i in range(len(df)):
        photo_id.append(int(re.findall(r"(\d+).", df['file'][i])[0]))
    df['photo_id'] = photo_id
    return df

def silhouetteInertia(poses):
    #Silhouette scores and inertia to find optimal number of clusters, k
    silhouette = []
    inertia = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(poses)
        silhouette.append(silhouette_score(poses, clusters))
        inertia.append(kmeans.inertia_)
    return (silhouette, inertia)

def plotSilIner(sil, iner, save, k_min=2, k_max=10):
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    ax[0].plot(range(k_min,k_max+1), sil)
    ax[0].set_xlabel('Number of Clusters, k')
    ax[0].set_ylabel('Silhouette Score')
    ax[1].plot(range(k_min,k_max+1), iner)
    ax[1].set_xlabel('Number of Clusters, k')
    ax[1].set_ylabel('Inertia')
    plt.tight_layout()
    plt.savefig('viz/' + save + '.png', dpi=500)
    plt.show()

def getKMeans(poses, k):
    kmeans = KMeans(n_clusters=k)
    clusters_kmeans = kmeans.fit_predict(poses)
    return kmeans, clusters_kmeans

def getGMM(poses, k):
    gmm = GaussianMixture(n_components=k, n_init=10)
    clusters_gmm = gmm.fit_predict(poses)
    return clusters_gmm

def getHier(poses, k):
    return AgglomerativeClustering(n_clusters = k, affinity='euclidean', linkage='ward').fit_predict(poses)

def plotManifold(pose_arr, kmeans_labels, gmm_labels, hier_labels, k, save):
    tsne_raw = TSNE(n_components=2).fit_transform(pose_arr)
    pca_raw = PCA(n_components=2).fit_transform(pose_arr)
    lle_raw = LocallyLinearEmbedding(n_components=2, n_neighbors=5).fit_transform(pose_arr)
    colors_kmeans = cm.nipy_spectral(kmeans_labels.astype(float) / k)
    colors_gmm = cm.rainbow(gmm_labels.astype(float) / k)
    colors_hier = cm.coolwarm(hier_labels.astype(float) / k)
    rows = ['t-SNE', 'PCA', 'LLE']
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes[0, 0].scatter(tsne_raw[:,0], tsne_raw[:,1], c=colors_kmeans)
    axes[0, 0].set_title('K-Means', size=16)
    axes[1, 0].scatter(pca_raw[:,0], pca_raw[:,1], c=colors_kmeans)
    axes[2, 0].scatter(lle_raw[:,0], lle_raw[:,1], c=colors_kmeans)
    
    axes[0, 1].scatter(tsne_raw[:,0], tsne_raw[:,1], c=colors_gmm)
    axes[0, 1].set_title('Gaussian Mixture Model', size=16)
    axes[1, 1].scatter(pca_raw[:,0], pca_raw[:,1], c=colors_gmm)
    axes[2, 1].scatter(lle_raw[:,0], lle_raw[:,1], c=colors_gmm)
    
    axes[0, 2].scatter(tsne_raw[:,0], tsne_raw[:,1], c=colors_hier)
    axes[0, 2].set_title('Hierarchical Clustering', size=16)
    axes[1, 2].scatter(pca_raw[:,0], pca_raw[:,1], c=colors_hier)
    axes[2, 2].scatter(lle_raw[:,0], lle_raw[:,1], c=colors_hier)
    
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 4, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=16, ha='right', va='center')
    
    plt.savefig('viz/' + save + '.png', dpi=500)
    plt.show()

def clusterExamples(k, n_examples, path, model_clusters, pose_df, pose_arr, mpii_edges, save):
    ax_array = np.linspace(1, k * 2 * n_examples - (k * 2 - 1), n_examples).astype(int)
    fig = plt.figure(figsize=(15, 15))
    for a in ax_array:
        addition = 0
        for cluster in range(k):
            arr_id = np.random.choice(np.where(model_clusters == cluster)[0])
            photo_id = ImageID(pose_df, arr_id)
            ax = fig.add_subplot(n_examples, k*2, a + addition)
            ax.imshow(importImage(path + photo_id))
            ax.set_xticks([])
            ax.set_yticks([])
            addition += 1
            ax = fig.add_subplot(n_examples, k*2, a+addition)
            plot2D(ax, pose_to_matrix(pose_arr[arr_id][:-1]), mpii_edges)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            if a == 1:
                ax.set_title('Cluster ' + str(cluster), position=(-0.1, 1), size=16)
            addition += 1

    plt.tight_layout()
    plt.savefig('viz/' + save + '.png')
    plt.show()


def ImageID(df, array_id):
    #Get photo id's of poses
    return df.loc[array_id, 'file']

def bodyAngle(pose_3d):
    midpoint = (pose_3d[0][:2] + pose_3d[5][:2])/2
    midpoint[1] *= -1
    torso = pose_3d[7][:2] * np.array([1, -1])
    body_angle_vec = torso-midpoint
    return math.atan2(body_angle_vec[0], body_angle_vec[1])*180/math.pi

def handHeight(pose_3d):
    hand_height = np.abs(np.min(-pose_3d[:, 1]) - np.min(-pose_3d[[10, 15]][:, 1]))
    return hand_height

def bodyHeight(pose_3d):
    #Body Height
    height = np.abs(np.max(-pose_3d[:, 1]) - np.min(-pose_3d[:, 1]))
    return height

def handWidth(pose_3d):
    return np.linalg.norm(pose_3d[10] - pose_3d[15])

def hipHeight(pose_3d):
    return np.abs(-np.min(-pose_3d[:, 1]))

def minLowerLegDist(pose_3d):
    return np.min([np.linalg.norm(pose_3d[1][:2] - pose_3d[0][:2]), np.linalg.norm(pose_3d[4][:2] - pose_3d[5][:2])])

def feetWidth(pose_3d):
    return np.linalg.norm(pose_3d[0] - pose_3d[5])

def minArmAngle(pose_3d):
    left_arm = pose_3d[15] - pose_3d[8]
    right_arm = pose_3d[10] - pose_3d[8]
    left_angle = math.atan2(np.abs(left_arm[1]), np.abs(left_arm[0]))*180/math.pi
    right_angle = math.atan2(np.abs(right_arm[1]), np.abs(right_arm[0]))*180/math.pi
    return np.min([left_angle, right_angle])

def minLowerLegAngle(pose_3d):
    left_arm = pose_3d[5] - pose_3d[4]
    right_arm = pose_3d[1] - pose_3d[0]
    left_angle = math.atan2(np.abs(left_arm[1]), np.abs(left_arm[0]))*180/math.pi
    right_angle = math.atan2(np.abs(right_arm[1]), np.abs(right_arm[0]))*180/math.pi
    return np.min([left_angle, right_angle])

def PosesFeatureSpace(clean_poses):
    #Input: clean_poses - dataset off all of the camera-invariant poses
    #Returns: dataset of poses in feature space
    pose_features = np.zeros((len(clean_poses), 9))
    for i in range(len(clean_poses)):
        pose_3d = pose_to_matrix(clean_poses[i])
        feature_array = np.array([bodyHeight(pose_3d), handHeight(pose_3d),
                                  bodyAngle(pose_3d), handWidth(pose_3d),
                                  hipHeight(pose_3d), minLowerLegDist(pose_3d),
                                  feetWidth(pose_3d), minArmAngle(pose_3d),
                                  minLowerLegAngle(pose_3d)])
        pose_features[i] = feature_array
    return pose_features

def importSBjson(file_name, path='data/events/'):
    with open(path+file_name) as data_file:
        #print (mypath+'events/'+file)
        data = json.load(data_file)
    
    #get the nested structure into a dataframe 
    #store the dataframe in a dictionary with the match id as key (remove '.json' from string)
    df = pd.json_normalize(data, sep = "_").assign(match_id = file_name[:-5])
    return df

def removePoorPredictions(set_3d_cvi_df):
    #List of the array_ids in which to remove because they are bad prediction of true pose
    to_remove_sets = np.array([1,6,7,11,14,25,27,28,31,32,37,40,42,43,44,51,52,53,55,58,
                               63,65,72,81,83,85,87,94,96,108,109,110,113,114,116,117,119,
                               123,131,133,135,136,137,140,141,143,144,147,150,151,154,156,
                               157,159,160,161,163,167,170,176,189,193,195,196,198,200,
                               202,203,206,207,210,211,213,216,217,218,220,227,228,235,
                               237,238,242,243,244,245,250,251,252,255,261,262,267,268,
                               270,271,274,275,276,282,287,291,296,297,298,304,305,311,312,
                               316,320,323,324,326,327,328,333,334,335,341,350,351,352,370,
                               372,374,379,387,388,389,390,395,397,401,406,411,413,414,418,
                               419,423,433,436,439,443,446,451,452,453,456,462,465,470,472,
                               474,475,480,489,490,494,502,507,509,515,517,522,528,532,533,
                               537,553,555,558,566,567,570,572,575,579,580,585,])
    #Remove selected poses
    set_3d_cvi_clean_df = set_3d_cvi_df.drop(to_remove_sets).reset_index(drop=True)
    keep_cols = np.array(list(range(48)) + ['gk_engage'])
    sets_3d_cvi_clean = set_3d_cvi_clean_df.loc[:,keep_cols].values
    return sets_3d_cvi_clean























