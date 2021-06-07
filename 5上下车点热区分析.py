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

# 按照时间范围取一定的轨迹子集
def get_sub_df_bytimerange(df,timerange):
    print('按时间范围 ',timerange[0],' 到 ',timerange[1],'取轨迹')
    sub_df = pd.DataFrame()
    # 按时间范围筛选轨迹
    sub_df = df[(df['定位时间']>=timerange[0]) & (df['定位时间']<=timerange[1])]
    # 重设索引，去除重复
    sub_df = sub_df.reset_index(drop=True)
    return sub_df

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“5上下车点热区分析”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

# 读取经过排序处理后的得到的轨迹数据csv
data_track = pd.read_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv')
# 按照时间取子集：分别为早上7点到早上9点（早高峰，上班），晚上17点到晚上19点（晚高峰，下班）
data_track_sub_7_9 = get_sub_df_bytimerange(data_track,['2018-11-05 07:00:00','2018-11-05 09:00:00'])
data_track_sub_7_9 = data_track_sub_7_9.append(get_sub_df_bytimerange(data_track,['2018-11-10 07:00:00','2018-11-10 09:00:00']))
data_track_sub_17_19 = get_sub_df_bytimerange(data_track,['2018-11-05 17:00:00','2018-11-05 19:00:00'])
data_track_sub_17_19 = data_track_sub_17_19.append(get_sub_df_bytimerange(data_track,['2018-11-10 17:00:00','2018-11-10 19:00:00']))

# 数据准备，需要准备三个时段的出租车数据：全天、早高峰时期、晚高峰时期
# 对全天的所有的出租车
pick = pd.DataFrame()
drop = pd.DataFrame()
# 以出租车为单位，循环遍历每一个出租车的轨迹
for tid in set(data_track['出租车ID']):
    track_a_TAXI = data_track[data_track['出租车ID'] == tid].reset_index(drop=True)
    x,xx,pick_up,drop_off = get_sub_trajectory(track_a_TAXI)
    # 对上下车点分别构造dataframe便于操作，分别赋予上下车点的类型
    if len(pick_up)>0:
        pick_df=pd.DataFrame(pick_up)
        pick_df.columns = ['经度','纬度']
        pick_df['类型'] = '上车点'
        pick=pick.append(pick_df)
    if len(drop_off)>0:
        drop_df=pd.DataFrame(drop_off)
        drop_df.columns = ['经度','纬度']
        drop_df['类型'] = '下车点'
        drop=drop.append(drop_df)
pick = pick.reset_index(drop=True)
drop = drop.reset_index(drop=True)

# 构造上车点的GeoDataFrame
gdf_pick = gpd.GeoDataFrame(
    pick, geometry=gpd.points_from_xy(pick['经度'], pick['纬度']),crs=4326)
# 构造下车点的GeoDataFrame
gdf_drop = gpd.GeoDataFrame(
    drop, geometry=gpd.points_from_xy(drop['经度'], drop['纬度']),crs=4326)

# 对早高峰时期（7-9）的所有的出租车
pick_sub_7_9 = pd.DataFrame()
drop_sub_7_9 = pd.DataFrame()
# 以出租车为单位，循环遍历每一个出租车的轨迹
for tid in set(data_track_sub_7_9['出租车ID']):
    track_a_TAXI = data_track_sub_7_9[data_track_sub_7_9['出租车ID'] == tid].reset_index(drop=True)
    x,xx,pick_up,drop_off = get_sub_trajectory(track_a_TAXI)
    # 对上下车点分别构造dataframe便于操作，分别赋予上下车点的类型
    if len(pick_up)>0:
        pick_df = pd.DataFrame(pick_up)
        pick_df.columns = ['经度','纬度']
        pick_df['类型'] = '上车点'
        pick_sub_7_9 = pick_sub_7_9.append(pick_df)
    if len(drop_off)>0:
        drop_df = pd.DataFrame(drop_off)
        drop_df.columns = ['经度','纬度']
        drop_df['类型'] = '下车点'
        drop_sub_7_9 = drop_sub_7_9.append(drop_df)
pick_sub_7_9 = pick_sub_7_9.reset_index(drop=True)
drop_sub_7_9 = drop_sub_7_9.reset_index(drop=True)

# 构造上车点的GeoDataFrame
gdf_pick_sub_7_9 = gpd.GeoDataFrame(
    pick_sub_7_9, geometry=gpd.points_from_xy(pick_sub_7_9['经度'], pick_sub_7_9['纬度']),crs=4326)
# 构造下车点的GeoDataFrame
gdf_drop_sub_7_9 = gpd.GeoDataFrame(
    drop_sub_7_9, geometry=gpd.points_from_xy(drop_sub_7_9['经度'], drop_sub_7_9['纬度']),crs=4326)

# 对晚高峰时期（17-19）的所有的出租车
pick_sub_17_19 = pd.DataFrame()
drop_sub_17_19 = pd.DataFrame()
# 以出租车为单位，循环遍历每一个出租车的轨迹
for tid in set(data_track_sub_17_19['出租车ID']):
    track_a_TAXI = data_track_sub_17_19[data_track_sub_17_19['出租车ID'] == tid].reset_index(drop=True)
    x,xx,pick_up,drop_off = get_sub_trajectory(track_a_TAXI)
    # 对上下车点分别构造dataframe便于操作，分别赋予上下车点的类型
    if len(pick_up)>0:
        pick_df = pd.DataFrame(pick_up)
        pick_df.columns = ['经度','纬度']
        pick_df['类型'] = '上车点'
        pick_sub_17_19 = pick_sub_17_19.append(pick_df)
    if len(drop_off)>0:
        drop_df = pd.DataFrame(drop_off)
        drop_df.columns = ['经度','纬度']
        drop_df['类型'] = '下车点'
        drop_sub_17_19 = drop_sub_17_19.append(drop_df)
pick_sub_17_19 = pick_sub_17_19.reset_index(drop=True)
drop_sub_17_19 = drop_sub_17_19.reset_index(drop=True)

# 构造上车点的GeoDataFrame
gdf_pick_sub_17_19 = gpd.GeoDataFrame(
    pick_sub_17_19, geometry=gpd.points_from_xy(pick_sub_17_19['经度'], pick_sub_17_19['纬度']),crs=4326)
# 构造下车点的GeoDataFrame
gdf_drop_sub_17_19 = gpd.GeoDataFrame(
    drop_sub_17_19, geometry=gpd.points_from_xy(drop_sub_17_19['经度'], drop_sub_17_19['纬度']),crs=4326)

# 绘图显示
# 全天的数据绘图
plt.rcParams['font.family']=['SimHei']
# 分割字图并设置整个图幅的大小为2000*2000
fig, axes = plt.subplots(2,2,figsize=(20,20))
# 上车点地图可视化
ax1=gdf_pick.plot(ax=axes[0,0],column='类型',color='red',legend=True,markersize=2)
ax1.set_ylim([30.4,30.8])
ax1.set_xlim([114.0,114.6])

# 下车点地图可视化
ax2=gdf_drop.plot(ax=axes[0,1],column='类型',color='blue',legend=True,markersize=2)
ax2.set_ylim([30.4,30.8])
ax2.set_xlim([114.0,114.6])

# 上车点热区可视化
ax3=gplt.kdeplot(ax=axes[1,0],df=gdf_pick,cmap='Reds',shade=True)
ax3.set_ylim([30.4,30.8])
ax3.set_xlim([114.0,114.6])

# 下车点热区可视化
ax4=gplt.kdeplot(ax=axes[1,1],df=gdf_drop,cmap='Blues',shade=True)
ax4.set_ylim([30.4,30.8])
ax4.set_xlim([114.0,114.6])

# 武汉市道路数据，第一个为全部的武汉市道路数据，第二个为武汉市三环以内的数据，在这里主要用到的是武汉市全部的道路数据
wuhan_All_road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
wuhan_Part_road = gpd.GeoDataFrame.from_file('data/road/WuhanPartroad/WHroad.shp')
wuhan_All_road.plot(ax=ax1,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax2,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax3,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax4,linewidth = 0.5, alpha = 0.5, color = 'black')

# 设置主标题和子标题
plt.suptitle('201811 所有出租车上下车点热区可视化')
ax1.set_title('上车点地图可视化')
ax2.set_title('下车点地图可视化')
ax3.set_title('上车点热区可视化')
ax4.set_title('下车点热区可视化')

# 调整子图之间的上下边距hspace和左右边距wspace
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.savefig('data/processed/output_picture/所有出租车全天上下车点热区可视化201811')

# 高峰时段的数据绘图
plt.rcParams['font.family']=['SimHei']
# 分割字图并设置整个图幅的大小为4000*2000
fig, axes = plt.subplots(2,4,figsize=(40,20))
# 早高峰上车点地图可视化
ax1 = gdf_pick_sub_7_9.plot(ax=axes[0,0],column='类型',color='red',legend=True,markersize=2)
ax1.set_ylim([30.4,30.8])
ax1.set_xlim([114.0,114.6])

# 早高峰下车点地图可视化
ax2 = gdf_drop_sub_7_9.plot(ax=axes[0,1],column='类型',color='blue',legend=True,markersize=2)
ax2.set_ylim([30.4,30.8])
ax2.set_xlim([114.0,114.6])

# 早高峰上车点热区可视化
ax3 = gplt.kdeplot(ax=axes[0,2],df=gdf_pick_sub_7_9,cmap='Reds',shade=True)
ax3.set_ylim([30.4,30.8])
ax3.set_xlim([114.0,114.6])

# 早高峰下车点热区可视化
ax4 = gplt.kdeplot(ax=axes[0,3],df=gdf_drop_sub_7_9,cmap='Blues',shade=True)
ax4.set_ylim([30.4,30.8])
ax4.set_xlim([114.0,114.6])

# 晚高峰上车点地图可视化
ax5 = gdf_pick_sub_17_19.plot(ax=axes[1,0],column='类型',color='red',legend=True,markersize=2)
ax5.set_ylim([30.4,30.8])
ax5.set_xlim([114.0,114.6])

# 晚高峰下车点地图可视化
ax6 = gdf_drop_sub_17_19.plot(ax=axes[1,1],column='类型',color='blue',legend=True,markersize=2)
ax6.set_ylim([30.4,30.8])
ax6.set_xlim([114.0,114.6])

# 晚高峰上车点热区可视化
ax7 = gplt.kdeplot(ax=axes[1,2],df=gdf_pick_sub_17_19,cmap='Reds',shade=True)
ax7.set_ylim([30.4,30.8])
ax7.set_xlim([114.0,114.6])

# 晚高峰下车点热区可视化
ax8 = gplt.kdeplot(ax=axes[1,3],df=gdf_drop_sub_17_19,cmap='Blues',shade=True)
ax8.set_ylim([30.4,30.8])
ax8.set_xlim([114.0,114.6])

# 武汉市道路数据，第一个为全部的武汉市道路数据，第二个为武汉市三环以内的数据，在这里主要用到的是武汉市全部的道路数据
wuhan_All_road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
wuhan_Part_road = gpd.GeoDataFrame.from_file('data/road/WuhanPartroad/WHroad.shp')
wuhan_All_road.plot(ax=ax1,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax2,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax3,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax4,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax5,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax6,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax7,linewidth = 0.5, alpha = 0.5, color = 'black')
wuhan_All_road.plot(ax=ax8,linewidth = 0.5, alpha = 0.5, color = 'black')

# 设置主标题和子标题
plt.suptitle('201811 所有出租车上下车点热区可视化')
ax1.set_title('早高峰上车点地图可视化')
ax2.set_title('早高峰下车点地图可视化')
ax3.set_title('早高峰上车点热区可视化')
ax4.set_title('早高峰下车点热区可视化')
ax5.set_title('晚高峰上车点地图可视化')
ax6.set_title('晚高峰下车点地图可视化')
ax7.set_title('晚高峰上车点热区可视化')
ax8.set_title('晚高峰下车点热区可视化')

# 调整子图之间的上下边距hspace和左右边距wspace
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.savefig('data/processed/output_picture/所有出租车高峰时段上下车点热区可视化201811')

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("\n程序“5上下车点热区分析”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“5上下车点热区分析”的主体代码执行的时间为：%d 秒" % maincode_seconds)