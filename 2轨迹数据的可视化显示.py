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

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“2轨迹数据的可视化显示”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

# 读取经过排序处理后的得到的轨迹数据csv
data_track = pd.read_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv')

# 选取某一辆出租车作为示例进行处理，修改taxiID_choose可以选择要选取的出租车ID
# 在这里使用的是随机数从轨迹数据集中任意选择一辆出租车进行分析，也可以直接手动设置taxiID_choose的值
taxiID_list = list(set(data_track['出租车ID'].tolist()))
print("当前出租车轨迹数据集涵盖出租车共 %d 辆" % len(taxiID_list))
taxiID_choose = taxiID_list[rd.sample(range(1, len(taxiID_list)-1), 1)[0]]
# taxiID_choose = 1015
print("选择出租车ID为 %d 的出租车轨迹进行可视化" % taxiID_choose)
data_track_of_oneTaxi = data_track[data_track['出租车ID'] == taxiID_choose].reset_index(drop=True)
print("出租车ID为 %d 的出租车一共有%d个数据点" % (taxiID_choose,len(data_track_of_oneTaxi)))
# 计算选中的出租车的数据点大小在所有出租车中的位次，位次越高（例如第1位）说明数据点越多，数据越大，处理时间越久
data_count_bytaxiID = []
for oneTaxiID in taxiID_list:
    pointscount = len(data_track[data_track['出租车ID'] == oneTaxiID])
    data_count_bytaxiID.append(pointscount)
data_count_bytaxiID.sort()
print("选择的出租车的轨迹点数量大小在所有出租车中排在第%d位\n" % (len(data_count_bytaxiID)-data_count_bytaxiID.index(len(data_track_of_oneTaxi))))


# 构造GeoDataFrame对象
# geopandas.GeoDataFrame(*args, geometry=None, crs=None, **kwargs)
GeoDf_of_oneTaxi = gpd.GeoDataFrame(
    data_track_of_oneTaxi, geometry=gpd.points_from_xy(data_track_of_oneTaxi['经度'].astype(float), data_track_of_oneTaxi['纬度'].astype(float)),crs=4326)

# 使matplotlib支持中文字体
plt.rcParams['font.family']=['SimHei']
# 绘图，分别指定渲染字段、颜色表、显示图例、点大小、图片大小
ax = GeoDf_of_oneTaxi.plot(column='空车/重车',cmap='coolwarm',legend=True,markersize=1,figsize=(15,15))
# 指定范围
# 如果想要根据经纬度的bbox来决定的话，释放下面两条代码即可
# ax.set_ylim([min(GeoDf_of_oneTaxi['纬度']),max(GeoDf_of_oneTaxi['纬度'])])
# ax.set_xlim([min(GeoDf_of_oneTaxi['经度']),max(GeoDf_of_oneTaxi['经度'])])
ax.set_ylim([30.4,30.8])
ax.set_xlim([114.0,114.6])
# 叠加武汉市路网
road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
road.plot(ax=ax, linewidth=0.5, alpha=0.5, color='grey')
plt.title("编号{taxiID}出租车“空车-重车”轨迹地图可视化".format(taxiID=taxiID_choose))
# 保存图像并在窗口中显示
plt.savefig('data/processed/output_picture/出租车“空车-重车”轨迹地图可视化/编号{taxiID}出租车“空车-重车”轨迹地图可视化'.format(taxiID=taxiID_choose))

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("程序“2轨迹数据的可视化显示”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“2轨迹数据的可视化显示”的主体代码执行的时间为：%d 秒" % maincode_seconds)

'''
在这里使用的是对所有轨迹点直接渲染，而没有优先进行轨迹分段后再对轨迹点进行渲染。
虽然这么做的操作更简单，但对于可视化而言，先进行轨迹分段再渲染轨迹点可以将离散的轨迹点按一个个时段进行划分，
便于实现仅针对某辆出租车或者某个时段的数据点进行显示
'''