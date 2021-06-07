"""***python调用包的引用部分***"""
# 科学计算基础库，用于存储和处理大型矩阵
import numpy as np
# 基于 NumPy 的数据操作库，提供了高效地操作大型数据集所需的工具和方法
import pandas as pd
# 绘图库，支持多种可视化功能
import matplotlib as mpl
import matplotlib.pyplot as plt
# 基于Pandas和Matplotlib的地理数据处理、绘图库，以及GeoPandas的依赖项
import geopandas as gpd
# 用于查找符合特定规则的文件路径名
import glob
# 处理日期和时间的标准库
import time,datetime
# 生成随机数的模块
import random as rd
# 数学库
import math
from math import tan, atan, acos, sin, cos, asin, sqrt, radians
# 平面特征进行集合理论分析和操作,几何对象的基本类型是点、曲线和曲面
from shapely.geometry import LineString,Point
from shapely.ops import nearest_points
# 经纬坐标模块
import pyproj
from pyproj import Transformer
# 基于Python语言的机器学习工具
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# 高级Python地理空间绘图库
import geoplot as gplt
# 动态时间规整（DTW）算法的python实现
from fastdtw import fastdtw
# tslearn库中dtw的计算模块
from tslearn.metrics import dtw
# 可以处理插值、积分、优化、图像处理、常微分方程数值解的求解、信号处理等问题的软件包
import scipy
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform, euclidean
# 用于创建、操作和研究复杂网络的结构、动态和功能
import networkx as nx
from networkx.algorithms import community
# HTTP库
import requests

"""***代码函数定义部分***"""
# 对某辆出租车轨迹进行分段，提取上下车点
def get_sub_trajectory(df):
    loads=[]
    no_loads=[]
    pick_up=[]
    drop_off=[]

    # 记录每一段轨迹的开始
    idx1 = -1
    # 记录每一段轨迹的结束
    idx2 = -1
    # 记录原始的出租车状态：空车/重车
    old_status = ''

    for index, row in df.iterrows():
        status = row['空车/重车']
        # 初始化，当对索引号为0的时
        if index==0:
            idx1 = index
            old_status = status

        # 判断状态是否转变，当发生了改变（status != old_status）时
        if status != old_status:
            # 将两次分界点之间的状态作为一段轨迹，取df的子集df[轨迹开始index1:轨迹结束idx2 + 1]，此处+1是*****
            sub_df = df[idx1:idx2 + 1]
            # 当状态为'重车'时，说明此时出租车是载客的状态，记录在loads轨迹集中
            if old_status=='重车':
                loads.append(sub_df)
                # 对于下车点记录经纬度(row['经度'],row['纬度'])，记录在drop_off点集中
                drop_off.append((row['经度'],row['纬度']))
            else:
                # 当状态不为‘重车’，即为‘空车’时，说明此时出租车是空车的状态，记录在no_loads数据集中
                no_loads.append(sub_df)
                # 对于上车点记录经纬度(row['经度'],row['纬度'])，记录在pick_up点集中
                pick_up.append((row['经度'],row['纬度']))

            # 当前时刻即为下一个时段的开始
            idx1 = index
            idx2 = index
            # 状态更新为下一个时段的状态
            old_status = status
        else:
            # 当状态没有发生改变时，轨迹结束index往后推一个
            idx2 = index

    # 对最后一个时段的数据进行处理并归档
    sub_df = df[idx1:idx2 + 1]
    if old_status=='重车':
        loads.append(sub_df)
    else:
        no_loads.append(sub_df)

    return loads, no_loads, pick_up, drop_off

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“4上下车点的可视化+聚类结果”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

# 读取经过排序处理后的得到的轨迹数据csv
data_track = pd.read_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv')

# 对出租车轨迹进行分段，提取上下车点
x, xx, pick_up, drop_off = get_sub_trajectory(data_track)

# 对上下车点构造dataframe便于操作
pick_df = pd.DataFrame(pick_up)
pick_df.columns = ['经度','纬度']
pick_df['类型'] = '上车点'
drop_df = pd.DataFrame(drop_off)
drop_df.columns = ['经度','纬度']
drop_df['类型'] = '下车点'

# 将上车点和下车点的dataframe合并，并重置index，防止索引重复
points_df = pick_df.append(drop_df).reset_index(drop=True)
# 基于上下车点dataframe构造GeoDataFrame数据集
gdf = gpd.GeoDataFrame(
    points_df, geometry=gpd.points_from_xy(points_df['经度'], points_df['纬度']),crs=4326)
# 使matplotlib支持中文字体
plt.rcParams['font.family']=['SimHei']
# 绘图，分别指定渲染字段、颜色表、显示图例、点大小、图片大小
ax = gdf.plot(column='类型',cmap='coolwarm',legend=True,markersize=2,figsize=(10,10))
# 设置标题
plt.title("201811 所有出租车的上下车点经纬度显示")
# 叠加武汉市路网
wuhan_road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
wuhan_road.plot(ax=ax, linewidth=0.5, alpha=0.5, color='grey')
# 设置子图的显示经纬度范围
ax.set_ylim([30,31.5])
ax.set_xlim([113.5,115])

# 保存图片在路径下：data/processed/output_picture/
plt.savefig('data/processed/output_picture/所有出租车的上下车点经纬度显示201811')

# 提取上下车点的地理坐标集合，为聚类做准备
location_points_df = np.array(points_df.loc[:, ['经度', '纬度']])
# 分割字图并设置整个图幅的大小为2000*1000
fig, axes = plt.subplots(1,2,figsize=(20,10))

# 使用k均值聚类kmeans,指定10个类
# sklearn.cluster.KMeans()包含fit(x[,y])方法，用来执行k均值聚类
# 推荐使用fit_predict(x[,y])计算簇的中心并且预测每个样本对应的簇类别，相当于先调用fit(X)再调用predict(X)
kmeans_value = KMeans(n_clusters=10).fit(location_points_df)
# 将得到的K均值聚类结果作为一个新的字段加入上下车点集中
points_df['K'] = kmeans_value.labels_
# 基于加入聚类后的上下车点dataframe构造GeoDataFrame数据集并绘图显示
gdf_kmeans = gpd.GeoDataFrame(
    points_df, geometry=gpd.points_from_xy(points_df['经度'], points_df['纬度']),crs=4326)
plt.rcParams['font.family']=['SimHei']
# 生成第一幅子图
ax1 = gdf_kmeans.plot(ax=axes[0],column='K',cmap='coolwarm',markersize=2)

# DBSCAN聚类
# Density-Based Spatial Clustering of Applications with Noise（具有噪声的基于密度的聚类方法）
# 是一种基于密度的空间聚类算法，将具有足够密度的区域划分为簇，并在具有噪声的空间数据库中发现任意形状的簇，它将簇定义为密度相连的点的最大集合。
# 传统的DBSCAN密度聚类算法，需要：邻域阈值(Eps)和点数阈值(minPts)2个参数来对数据集进行聚类
DBSCAN_value = DBSCAN(eps=0.01).fit(location_points_df)
# 将得到的DBSCAN聚类结果作为一个新的字段加入上下车点集中
points_df['DB'] = DBSCAN_value.labels_

# 核心点
core = points_df[points_df['DB']>-1]
# 边界点
border = points_df[points_df['DB']==-1]
# 核心点构造GeoDataFrame并进行可视化
gdf_dbscan = gpd.GeoDataFrame(
    core, geometry=gpd.points_from_xy(core['经度'], core['纬度']),crs=4326)
plt.rcParams['font.family']=['SimHei']

# 生成第二幅子图
ax2 = gdf_dbscan.plot(ax=axes[1],column='DB', cmap='coolwarm', markersize=3)
# 边界点构造GeoDataFrame并进行可视化
gdf_dbscan0 = gpd.GeoDataFrame(
    border, geometry=gpd.points_from_xy(border['经度'], border['纬度']),crs=4326)
# 核心点的基础上叠加边界点
gdf_dbscan0.plot(ax=axes[1], column='DB', color='black', markersize=1)

# 设置主标题和子标题
plt.suptitle('201811 K均值和DBSCAN聚类后的上下车点经纬度显示')
ax1.set_title('K均值聚类后的上下车点地图可视化')
ax2.set_title('DBSCAN聚类后的上下车点地图可视化')

# 叠加武汉市路网
wuhan_road.plot(ax=ax1, linewidth=0.5, alpha=0.5, color='grey')
wuhan_road.plot(ax=ax2, linewidth=0.5, alpha=0.5, color='grey')
# 设置子图范围
ax1.set_ylim([30,31.5])
ax1.set_xlim([113.5,115])
ax2.set_ylim([30,31.5])
ax2.set_xlim([113.5,115])

plt.savefig('data/processed/output_picture/K均值聚类和DBSCAN聚类后的上下车点经纬度显示201811')

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("\n程序“4上下车点的可视化+聚类结果”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“4上下车点的可视化+聚类结果”的主体代码执行的时间为：%d 秒" % maincode_seconds)
