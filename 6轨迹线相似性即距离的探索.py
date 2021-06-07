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

# 动态时间规整（DTW）：在两个时间序列之间计算dtw距离的代码实现
# “可以把序列某个时刻的点跟另一时刻多个连续时刻的点相对应”的做法称为时间规整
# DTW算法的步骤为：①计算两个序列各个点之间的距离矩阵；②寻找一条从矩阵左上角到右下角的路径，使得路径上的元素和最小
def dtw_distance(ts_a, ts_b, d=lambda x, y: abs(x - y), mww=10000):
    # Computes dtw distance between two time series 在两个时间序列之间计算dtw距离
    #
    # Args:
    #     ts_a: time series a 时间序列a
    #     ts_b: time series b 时间序列b
    #     d: distance function，Lambda函数又称匿名函数，作用是对输入的x和y执行冒号后面的表达式
    #     mww: max warping window, int, optional (default = infinity)
    #
    # Returns:
    #     dtw distance

    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    # 生成一个全为1的数组
    cost = np.ones((M, N))

    # 初始化第一行和第一列
    cost[0, 0] = d(ts_a[0], ts_b[0])
    # 比较系列a的第i个和系列b的第一个，赋值到第一列中，即列代表系列a
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])
    # 比较系列b的第j个和系列a的第一个，赋值到第一行中，即行代表系列b
    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]

# 获取数据集中所有长度相同（具有一样的轨迹点数目）的两条轨迹的TID号
def get_two_HaveSamePoints_tracks(track_dataframe):
    t1 = []
    t2 = []
    # 按照TID统计轨迹点数，得到每一条轨迹包含的数据点数量
    track_counts = track_dataframe['TID'].value_counts()
    # 使用track_counts.items()遍历整个track_counts
    for k1, v1 in track_counts.items():
        # k为轨迹的TID，v为轨迹包含的数据点数量
        for k2, v2 in track_counts.items():
            # 这里的意思是当TID不一样，但包含的数量点数量一样时，选到两条即可
            if k1 != k2 and v1 == v2:
                t1.append(k1)
                t2.append(k2)
                break
    drop_index = []
    for i in range(len(t1)):
        for j in range(i + 1, len(t1)):
            if t1[i] == t2[j] and t1[j] == t2[i]:
                drop_index.append(j)

    track1_HaveSamePoints = [t1[i] for i in range(0, len(t1), 1) if i not in drop_index]
    track2_HaveSamePoints = [t2[i] for i in range(0, len(t2), 1) if i not in drop_index]
    return track1_HaveSamePoints,track2_HaveSamePoints

# 对于序列s，实现随机丢弃其中的size个元素
def randomly_drop(s, drop_size):
    # 从输入参数drop_size获取丢弃元素的个数size
    size = drop_size

    if drop_size >= len(s):
        size = 50
    drops = []
    # 随机产生50个需要删除的元素的索引，
    # rd.randint()返回一个随机整型数，范围：(最小值<=x<最大值)
    # for i in range(50): drops.append(rd.randint(1,len(s)-1))这种方法生成的随机索引数会有重复
    # 因此使用random.sample(range(最小值,最大值),随机数数量)的方法生成无重复的随机数列表
    drops = rd.sample(range(1, len(s)-1), size)

    # 将drops中的元素去重之后重新生成list
    # 这里set()会导致前面使用rd.randint()生成的随机数列表有数据丢失，但random.sample()的方法不会
    drops = list(set(drops))
    # 输出drops的内容和元素个数检查是否正确
    # print(drops)
    # print(len(drops))
    dt2 = s.drop(s.index[drops])  # 去掉随机的50个点后的轨迹
    return dt2

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“6轨迹线相似性即距离的探索”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

# 读取经过排序处理后的得到的轨迹数据csv
data_track = pd.read_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv')
# 为loads轨迹列表中每个轨迹的DataFrame添加一列TID字段，同一条轨迹的TID是一致的
# 载客轨迹是返回参数的第一个，即get_sub_trajectory()[0]
loads_track = get_sub_trajectory(data_track)[0]
# 构造一个用来保存添加TID后的轨迹的容器reserved_tracks
reserved_tracks = []
# 设置初始的轨迹TID为1
track_id = 1
for i in range(0, len(loads_track)):
    # 路径长度需要大于20个记录，才添加TID并保存在reserved_tracks中
    if len(loads_track[i]) > 20:
        loads_track[i]['TID'] = track_id
        track_id += 1
        # 保存在reserved_tracks中
        reserved_tracks.append(loads_track[i])

# 在窗口中输出得到的reserved_tracks结果查看
# print(loads_track[0])

# 将reserved_tracks拼接为一个DataFrame
track_df = pd.concat(reserved_tracks)
# 保存轨迹记录在路径data/processed/output_data/下，命名为”含TID载客轨迹数据+时间“
track_df.to_csv('data/processed/output_data/含TID载客轨迹数据201811.csv',index=False, encoding="utf_8_sig")
# 获取数据集中所有长度相同（具有一样的轨迹点数目）的两条轨迹的TID号
track1_HaveSamePoints,track2_HaveSamePoints = get_two_HaveSamePoints_tracks(track_df)
print("此轨迹数据集中一共有%d对长度相同（具有一样的轨迹点数目）的两条轨迹" % len(track1_HaveSamePoints))

# 由于数据太多，只选择数据集中的部分数据进行分析即可，在这里先生成一个保留下来的索引列表，再根据这个列表保留1000对数据进行分析
Savelist_index = [rd.randint(0,len(track1_HaveSamePoints))for i in range(1000)]
track1_HaveSamePoints_Savepart = [track1_HaveSamePoints[i] for i in Savelist_index]
track2_HaveSamePoints_Savepart = [track2_HaveSamePoints[i] for i in Savelist_index]
track1_HaveSamePoints = track1_HaveSamePoints_Savepart
track2_HaveSamePoints = track2_HaveSamePoints_Savepart
print("经过处理后，使用此轨迹数据集中的%d对长度相同（具有一样的轨迹点数目）的两条轨迹进行分析" % len(track1_HaveSamePoints))
print("被选中的轨迹索引值列表为：")
print(Savelist_index)

# 记录开始计算轨迹对之间的欧式距离的时间，这里由于算的比较快，故使用毫秒计数
time_test_Euclid_start_microsecond1 = datetime.datetime.now().microsecond
time_test_Euclid_start_microsecond2 = time.mktime(datetime.datetime.now().timetuple())

for i in range(len(track1_HaveSamePoints)):
    # 根据得到的两条轨迹的TID号，将两条轨迹提出来分别命名为：distance_test_track1和distance_test_track2
    distance_test_track1 = track_df[track_df['TID'] == track1_HaveSamePoints[i]]
    distance_test_track2 = track_df[track_df['TID'] == track2_HaveSamePoints[i]]

    # 分别将两条用于测试轨迹距离的轨迹的经纬度提取出来
    test_track1_lat = distance_test_track1['纬度'].values
    test_track2_lat = distance_test_track2['纬度'].values
    test_track1_lon = distance_test_track1['经度'].values
    test_track2_lon = distance_test_track2['经度'].values

    # 使用最基础的欧几里得距离方法，测度它们之间的轨迹距离，实际上就转化为每个轨迹中对应节
    # 点间的欧氏距离均值
    # np.array之间可以之间相加减，np.square获取矩阵元素的平方
    # 经纬度的差的平方和的开根号，再求和取平均得到
    Euclid_tracks_distance = np.sum(np.sqrt(np.square(np.array(test_track1_lat) - np.array(test_track2_lat)) +
                                            np.square(np.array(test_track1_lon) - np.array(test_track2_lon)))) / len(distance_test_track1)
    # print("第%d对包含相同数量的数据点之间的欧式距离为：%f" % (i,Euclid_tracks_distance))

# 记录结束计算轨迹对之间的欧式距离的时间
time_test_Euclid_end_microsecond1 = datetime.datetime.now().microsecond
time_test_Euclid_end_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
# 结束时间减去开始时间再除以处理的轨迹对数，得到每一次计算欧氏距离花费的时间
cost_time_Euclid = float((time_test_Euclid_end_microsecond2 - time_test_Euclid_start_microsecond2) * 1000 +(time_test_Euclid_end_microsecond1 - time_test_Euclid_start_microsecond1) / 1000)/len(track1_HaveSamePoints)
print('每一次计算欧氏距离用时：%fms' % cost_time_Euclid)

# 记录开始使用fastdtw计算轨迹对之间的dtw距离的时间，这里由于算的比较快，故使用毫秒计数
time_test_fastdtw_start_microsecond1 = datetime.datetime.now().microsecond
time_test_fastdtw_start_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
for i in range(len(track1_HaveSamePoints)):
    # 根据得到的两条轨迹的TID号，将两条轨迹提出来分别命名为：distance_test_track1和distance_test_track2
    distance_test_track1 = track_df[track_df['TID'] == track1_HaveSamePoints[i]]
    distance_test_track2 = track_df[track_df['TID'] == track2_HaveSamePoints[i]]

    # 将用于测试轨迹距离的第一条轨迹的经纬度提取出来
    test_track1_lat = distance_test_track1['纬度'].values
    test_track1_lon = distance_test_track1['经度'].values

    # 为了验证DTW方法是否适用，我们对上面获得的两条包含相同数据点数量的轨迹进行处理，随机丢弃其中一条轨迹的5个元素
    # 这里随机丢弃distance_test_track2中的5个元素，命名为distance_test_track2_drop
    distance_test_track2_drop = randomly_drop(distance_test_track2, 5)

    # print('轨迹1的长度：' + str(len(distance_test_track1)) + '\n轨迹2的长度：' + str(len(distance_test_track2_drop)))
    test_track2_drop_lat = distance_test_track2_drop['纬度'].values
    test_track2_drop_lon = distance_test_track2_drop['经度'].values

    # 使用fastdtw的模块计算在经度、纬度方向上计算两条轨迹的偏差，dist选用欧几里得
    dist_fast_lat, path1 = fastdtw(test_track1_lat, test_track2_drop_lat, dist=euclidean)
    dist_fast_lon, path2 = fastdtw(test_track1_lon, test_track2_drop_lon, dist=euclidean)
    # “**”在python中是乘方运算符，也称幂运算符，“**2”表示平方，下式代表经过fastdtw的模块计算的经纬方向偏差的平方和的开根值
    # print("使用fastdtw方法第%d次计算两条轨迹的轨迹距离为：%f" % (i+1,math.sqrt(dist_fast_lat ** 2 + dist_fast_lon ** 2)))

# 记录结束使用fastdtw计算轨迹对之间的dtw距离的时间
time_test_fastdtw_end_microsecond1 = datetime.datetime.now().microsecond
time_test_fastdtw_end_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
# 结束时间减去开始时间再除以处理的轨迹对数，得到使用fastdtw方法单次计算两条轨迹的dtw距离的时间
cost_time_fastdtw = float((time_test_fastdtw_end_microsecond2 - time_test_fastdtw_start_microsecond2) * 1000 +(time_test_fastdtw_end_microsecond1 - time_test_fastdtw_start_microsecond1) / 1000)/len(track1_HaveSamePoints)
print('使用fastdtw方法单次计算两条轨迹的dtw距离的平均耗时为：%fms' % cost_time_fastdtw)

# 记录开始使用自定义的dtw_distance函数计算轨迹对之间的dtw距离，这里由于算的比较快，故使用毫秒计数
time_test_dtw_distance_start_microsecond1 = datetime.datetime.now().microsecond
time_test_dtw_distance_start_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
for i in range(len(track1_HaveSamePoints)):
    # 根据得到的两条轨迹的TID号，将两条轨迹提出来分别命名为：distance_test_track1和distance_test_track2
    distance_test_track1 = track_df[track_df['TID'] == track1_HaveSamePoints[i]]
    distance_test_track2 = track_df[track_df['TID'] == track2_HaveSamePoints[i]]

    # 将用于测试轨迹距离的第一条轨迹的经纬度提取出来
    test_track1_lat = distance_test_track1['纬度'].values
    test_track1_lon = distance_test_track1['经度'].values

    # 为了验证DTW方法是否适用，我们对上面获得的两条包含相同数据点数量的轨迹进行处理，随机丢弃其中一条轨迹的5个元素
    # 这里随机丢弃distance_test_track2中的5个元素，命名为distance_test_track2_drop
    distance_test_track2_drop = randomly_drop(distance_test_track2, 5)

    # print('轨迹1的长度：' + str(len(distance_test_track1)) + '\n轨迹2的长度：' + str(len(distance_test_track2_drop)))
    test_track2_drop_lat = distance_test_track2_drop['纬度'].values
    test_track2_drop_lon = distance_test_track2_drop['经度'].values

    # 使用dtw_distance的方法计算在经度、纬度方向上计算两条轨迹的偏差
    dtw_distance_lat = dtw_distance(test_track1_lat, test_track2_drop_lat)
    dtw_distance_lon = dtw_distance(test_track1_lon, test_track2_drop_lon)
    # “**”在python中是乘方运算符，也称幂运算符，“**2”表示平方，下式代表经过dtw_distance的方法计算的经纬方向偏差的平方和的开根值
    # print("使用dtw_distance方法第%d次计算两条轨迹的轨迹距离为：%f" % (i + 1, math.sqrt(dtw_distance_lat ** 2 + dtw_distance_lon ** 2)))

# 记录结束使用自定义的dtw_distance函数计算轨迹对之间的dtw距离的时间
time_test_dtw_distance_end_microsecond1 = datetime.datetime.now().microsecond
time_test_dtw_distance_end_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
# 结束时间减去开始时间再除以处理的轨迹对数，得到使用dtw_distance函数单次计算两条轨迹的dtw距离的时间
cost_time_dtw_distance = float((time_test_dtw_distance_end_microsecond2 - time_test_dtw_distance_start_microsecond2) * 1000 +(time_test_dtw_distance_end_microsecond1 - time_test_dtw_distance_start_microsecond1) / 1000)/len(track1_HaveSamePoints)
print('使用dtw_distance函数单次计算两条轨迹的dtw距离的平均耗时为：%fms' % cost_time_dtw_distance)

# 记录开始使用tslearn中的dtw模块计算轨迹对之间的dtw距离，这里由于算的比较快，故使用毫秒计数
time_test_tslearndtw_start_microsecond1 = datetime.datetime.now().microsecond
time_test_tslearndtw_start_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
for i in range(len(track1_HaveSamePoints)):
    # 根据得到的两条轨迹的TID号，将两条轨迹提出来分别命名为：distance_test_track1和distance_test_track2
    distance_test_track1 = track_df[track_df['TID'] == track1_HaveSamePoints[i]]
    distance_test_track2 = track_df[track_df['TID'] == track2_HaveSamePoints[i]]

    # 将用于测试轨迹距离的第一条轨迹的经纬度提取出来
    test_track1_lat = distance_test_track1['纬度'].values
    test_track1_lon = distance_test_track1['经度'].values

    # 为了验证DTW方法是否适用，我们对上面获得的两条包含相同数据点数量的轨迹进行处理，随机丢弃其中一条轨迹的5个元素
    # 这里随机丢弃distance_test_track2中的5个元素，命名为distance_test_track2_drop
    distance_test_track2_drop = randomly_drop(distance_test_track2, 5)

    # print('轨迹1的长度：' + str(len(distance_test_track1)) + '\n轨迹2的长度：' + str(len(distance_test_track2_drop)))
    test_track2_drop_lat = distance_test_track2_drop['纬度'].values
    test_track2_drop_lon = distance_test_track2_drop['经度'].values

    # 使用tslearn中的dtw模块计算在经度、纬度方向上计算两条轨迹的偏差
    dtw_distance_lat = dtw(test_track1_lat, test_track2_drop_lat)
    dtw_distance_lon = dtw(test_track1_lon, test_track2_drop_lon)
    # “**”在python中是乘方运算符，也称幂运算符，“**2”表示平方，下式代表经过dtw_distance的方法计算的经纬方向偏差的平方和的开根值
    # print("使用dtw_distance方法第%d次计算两条轨迹的轨迹距离为：%f" % (i + 1, math.sqrt(dtw_distance_lat ** 2 + dtw_distance_lon ** 2)))

# 记录结束使用tslearn中的dtw模块计算轨迹对之间的dtw距离的时间
time_test_tslearndtw_end_microsecond1 = datetime.datetime.now().microsecond
time_test_tslearndtw_end_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
# 结束时间减去开始时间再除以处理的轨迹对数，得到使用tslearn中的dtw模块单次计算两条轨迹的dtw距离的时间
cost_time_tslearndtw = float((time_test_tslearndtw_end_microsecond2 - time_test_tslearndtw_start_microsecond2) * 1000 +(time_test_tslearndtw_end_microsecond1 - time_test_tslearndtw_start_microsecond1) / 1000)/len(track1_HaveSamePoints)
print('使用tslearn中的dtw模块单次计算两条轨迹的dtw距离的平均耗时为：%fms' % cost_time_tslearndtw)

# 验证DTW方法是否适用到此结束

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("\n程序“6轨迹线相似性即距离的探索”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“6轨迹线相似性即距离的探索”的主体代码执行的时间为：%d 秒" % maincode_seconds)