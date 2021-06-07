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

# 从轨迹中生成经纬度列表"coords"，并生成新的dataframe元素，字段名为：{"id","taxi","track_id","start","coords"}，返回值为dataframe的列表
def get_track_coords(track_df, TIDs):
    # 参数：track_df轨迹的dataframe，TIDs表示TID的列表
    tracks = []
    new_id = 1
    for ids in TIDs:
        # if ids % 1000 == 0:
        #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # 根据TID获取轨迹中各个数据点的经纬度
        track_lons = track_df[track_df['TID'] == ids]['经度'].values.tolist()
        track_lats = track_df[track_df['TID'] == ids]['纬度'].values.tolist()
        # 重新生成一个列表tracks并作为返回值，新的dataframe的字段名为：{"id","taxi","track_id","start","coords"}
        # 字段名分别的含义为："id"：选取的轨迹的重新编号；"taxi"：出租车ID，"track_id"：轨迹的TID号；
        # "start"：轨迹的第一个数据点的定位时间；"coords"：轨迹各数据点的经纬度列表[[经度列表],[维度列表]]
        tracks.append({'id':new_id, 'taxi': track_df[track_df['TID'] == ids]['出租车ID'].values.tolist()[0], 'track_id': ids,
                       'start':track_df[track_df['TID'] == ids]['定位时间'].values.tolist()[0], 'coords': [track_lons, track_lats]})
        new_id += 1
    return tracks

# 使用fastdtw方法获取表示两两轨迹之间的距离矩阵
def np_dtw_matrix(dataset):
    # 参数：选取的部分出租车轨迹列表{"id","taxi","track_id","start","coords"}
    # N表示出租车轨迹的总数
    N = len(dataset)
    # 初始化距离矩阵res，np.zeros((N,N),dtype='float32')生成一个零值矩阵
    res = np.zeros((N,N),dtype='float32')
    print("开始计算两两轨迹之间的距离矩阵，当前时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i in range(N):
        # 较多时间将会被用于动态得求出两条一般的轨迹的间距，这里每隔20个数据输出一次当前时间，至少让用户知道他还在执行代码
        if i % 20 == 0:
            print("仍然在计算两两轨迹之间的距离矩阵，不要着急，当前时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for j in range(i+1, N):
            # fastdtw方法计算dtw距离，['coords'][0]为经度，['coords'][1]为纬度
            dis_lon, _ = fastdtw(dataset[i]['coords'][0], dataset[j]['coords'][0], dist=euclidean)
            dis_lat, _ = fastdtw(dataset[i]['coords'][1], dataset[j]['coords'][1], dist=euclidean)
            res[i,j] = math.sqrt(dis_lat**2 + dis_lon**2)
            # 矩阵为对称矩阵，因此可以通过求解矩阵的一半来加快部分运行的效率，另外一半对称获得即可
            res[j,i] = res[i,j]
    return res

# 通过百度地图API将经纬度转换成乡镇级的地理名称
def coordinatesToPosition(lng,lat):
    '''
    函数输入为： lng: 经度, lat: 纬度

    函数输出为：
    address:     解析后的地理位置名称
    province:    省份名称
    city:        城市名
    district:    县级行政区划名
    town:        乡镇级行政区划
    adcode:      县级行政区划编码
    town_code:   镇级行政区划编码
    business:    坐标所在商圈信息，如 "人民大学,中关村,苏州街"。最多返回3个
    regionsName: 点所处的区域名
    '''

    AK = 'mpiROVbiOwCzul140PHChZikaaTltbwv'
    # coordtype=wgs84ll 表示输入的坐标为WGS84坐标
    # extensions_town=true 表示行政区划返回乡镇级数据
    url ='http://api.map.baidu.com/reverse_geocoding/v3/?output=json&coordtype=wgs84ll&ak=%s&location=%s,%s&extensions_town=true&extensions_poi=1'%(AK,lat,lng)
    Result = requests.get(url)
    # status_code为get请求的状态码，常用200表示成功接收请求并已完成整个处理过程
    if Result.status_code == 200:
        resultValue = Result.json()
        # 'status'为服务状态码，返回为0表示服务请求正常召回
        if resultValue['status'] == 0:
            resultValue = resultValue['result']
            resultValue = {
                'address': resultValue['formatted_address'],
                'province': resultValue['addressComponent']['province'],
                'city': resultValue['addressComponent']['city'],
                'district': resultValue['addressComponent']['district'],
                'town': resultValue['addressComponent']['town'],
                'adcode': resultValue['addressComponent']['adcode'],
                'town_code': resultValue['addressComponent']['town_code'],
                'business': resultValue['business'],
                'regionsName': resultValue['sematic_description'],
            }
        else:
            resultValue = None
        return resultValue
    else:
        print('无法获取(%s,%s)的地理信息！' % (lat,lng))

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“8出租车目的地预测”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

# 数据准备
track_data = pd.read_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv')

# c为控制循环次数的变量
c = 0
# 广埠屯坐标(114.36141,30.52376)
# clng = 114.36141
# clat = 30.52376
# 洪山广场坐标(114.3315714763881,30.547395825334213)
clng = 114.3315714763881
clat = 30.547395825334213


# print("选用%d条轨迹进行出租车目的地预测" % n)
print("轨迹起点选在洪山广场，洪山广场的经纬度坐标为：(%f,%f)" % (clng,clat))
# 新建一个dataframe，命名为track_part_data_StartAt_GPT，用来存储部分被选用的起点位于广埠屯的出租车轨迹数据
track_part_data_StartAt_GPT = pd.DataFrame()

# 对所有的出租车
for tid in set(track_data['出租车ID']):
    # 截取['出租车ID']==tid的出租车轨迹，重新设置索引
    df1 = track_data[track_data['出租车ID'] == tid].reset_index(drop=True)
    # 对['出租车ID']==tid的出租车轨迹进行分段，提取上下车点
    loads, no_loads, pick_up, drop_off = get_sub_trajectory(df1)
    # 对载客的轨迹进行分析
    for trac in loads:
        trac = trac.reset_index(drop=True)
        lat = float(trac.iloc[0]['纬度'])
        lng = float(trac.iloc[0]['经度'])
        # 起始点在广谱屯（经度差和纬度差均小于0.002且轨迹长度大于10个数据点）
        if abs(lat-clat) < 0.002 and abs(lng-clng) < 0.002 and len(trac) > 10:
            trac['TID'] = c
            track_part_data_StartAt_GPT = track_part_data_StartAt_GPT.append(trac)
            c += 1
            print("第%d条轨迹的起始点经纬度坐标为：(%f,%f)" % (c,lat,lng))

track_part_data_StartAt_GPT = track_part_data_StartAt_GPT.reset_index(drop=True)

plt.rcParams['font.family'] = ['SimHei']
# 分割字图并设置整个图幅的大小为2000*2000
fig, axes = plt.subplots(2,2,figsize=(20,20))

# 可视化查看这些轨迹，生成第一个子图
gdf1 = gpd.GeoDataFrame(
    track_part_data_StartAt_GPT, geometry=gpd.points_from_xy(track_part_data_StartAt_GPT['经度'], track_part_data_StartAt_GPT['纬度']),crs=4326)
ax1 = gdf1.plot(ax=axes[0,0],column='TID',markersize=10,figsize=(10,10),cmap='Set1')
station = gpd.GeoDataFrame(geometry=gpd.points_from_xy([clng],[clat]), crs=4326)
station.plot(ax=ax1)
ax1.set_title('部分起始点在洪山广场附近的出租车的轨迹可视化')

tIDs = track_part_data_StartAt_GPT['TID'].unique().tolist()

# 分出一个测试轨迹
# tIDs[:-1]表示从0到-1，tIDs[-1:]表示从-1到tIDs的最后一个
# track_coords是对track_part_data_StartAt_GPT中所有的轨迹（除去测试轨迹）提取经纬度
track_coords = get_track_coords(track_part_data_StartAt_GPT, tIDs[:-1])
# test_track_coords是对track_part_data_StartAt_GPT中的最后一个轨迹提取经纬度，将被用于计算测试轨迹到各个类的距离，以此估计测试轨迹的目的地
test_track_coords = get_track_coords(track_part_data_StartAt_GPT, tIDs[-1:])
print("用于训练预测模型的轨迹集共有%d条轨迹" % len(track_coords))

# 对track_part_data_StartAt_GPT中所有的轨迹求两两轨迹之间的距离矩阵dtw_matrix
dtw_matrix = np_dtw_matrix(track_coords)
print("两两轨迹之间的距离矩阵为：")
print(dtw_matrix)

# 使用前面“轨迹聚类”中的轨迹聚类方法，形成轨迹的平面聚类
Z1 = sch.linkage(dtw_matrix, method='average')
# print(Z1)
cluster1 = sch.fcluster(Z1, t=1, criterion='inconsistent')

print("轨迹聚类的分类结果为：")
print(cluster1)
print("轨迹聚类共分%d类" % len(set(cluster1.tolist())))

# 对track_coords添加一个字段['cluster']并将聚类的结果赋值
for i in range(0, len(track_coords)):
    track_coords[i]['cluster'] = cluster1.tolist()[i]
# 由于track_coords是track_part_data_StartAt_GPT派生出来的，可以根据track_part_data_StartAt_GPT.TID和track_coords的'track_id'对应
# 将聚类结果添加到track_part_data_StartAt_GPT中
for label in cluster1.tolist():
    # 对于每一个label生成一个'track_id'的列表
    tids = [track_coords[i]['track_id'] for i in range(0, len(track_coords)) if track_coords[i]['cluster'] == label]
    # track_part_data_StartAt_GPT.TID等于ids时，代表着这个轨迹属于这个类，将label赋值给'cluster'字段
    for ids in tids:
        track_part_data_StartAt_GPT.loc[(track_part_data_StartAt_GPT.TID == ids), 'cluster'] = label

# 去除掉聚类的异常结果
track_part_data_StartAt_GPT = track_part_data_StartAt_GPT[track_part_data_StartAt_GPT['cluster'] <= len(set(cluster1.tolist()))]
# 生成第2个子图
gdf1 = gpd.GeoDataFrame(
    track_part_data_StartAt_GPT, geometry=gpd.points_from_xy(track_part_data_StartAt_GPT['经度'], track_part_data_StartAt_GPT['纬度']), crs=4326)
ax2 = gdf1.plot(ax=axes[0,1],column='cluster', markersize=10, figsize=(10, 10), cmap='Set1')
ax2.set_title('部分起始点在洪山广场附近的出租车的轨迹聚类结果可视化')

# track_part_data_StartAt_GPT中的轨迹数
N = len(track_coords)
# 将聚类结果生成列表并计算列表元素个数
clist = cluster1.tolist()
N2 = len(set(clist))
# 初始化距离矩阵为零矩阵(1行，N2列）
res = np.zeros((1, N2), dtype='float32')
# 初始化计数矩阵为零矩阵(1行，N2列）
res_count = np.zeros((1, N2), dtype='float32')

for i in range(N):
    # 计算测试轨迹test_track_coords与轨迹集track_coords中的每一个轨迹在经纬度方向上的dtw距离
    dis_lon, _ = fastdtw(track_coords[i]['coords'][0], test_track_coords[0]['coords'][0], dist=euclidean)
    dis_lat, _ = fastdtw(track_coords[i]['coords'][1], test_track_coords[0]['coords'][1], dist=euclidean)

    # 计算到每个聚类的dtw距离累计和
    # 这里我觉得源码存在问题，按照源码的意思没有对应到这条轨迹对应的聚类上
    # 因此我将clist[i] - 1更换成了track_coords[i]['cluster'] - 1
    res[0, track_coords[i]['cluster'] - 1] += math.sqrt(dis_lat ** 2 + dis_lon ** 2)
    # 统计每一个聚类与track_coords中的轨迹的最邻近次数
    res_count[0, track_coords[i]['cluster'] - 1] += 1

# 对每一个聚类取平均，得到每一个聚类的平均dtw距离
res = res / res_count
# 取平均dtw距离最小的聚类的索引位置值+1作为聚类值
c = res.tolist()[0].index(min(res.tolist()[0])) + 1
print("测试轨迹的目的地和聚类%d的最相近\n" % c)
# 此处应该是取平均dtw距离最小的聚类作为出租车目的地的预测值，即出租车最有可能沿着这些轨迹前进，对源码进行了修改
forecast_track = track_part_data_StartAt_GPT[track_part_data_StartAt_GPT['cluster'] == c].reset_index()
# 目的地地点在forecast_track中的索引号列表，用于提取目的地
destination_index = []

for i in range(len(forecast_track)):
    if(forecast_track.loc[i,'TID'] != forecast_track.loc[(i+1),'TID']):
        destination_index.append(i)
    if(i == (len(forecast_track))-2):
        break
destination_index.append(len(forecast_track)-1)
destination = forecast_track.loc[destination_index]
print("测试轨迹可能的目的地为：")
for index, row in destination.iterrows():
    positionTemp = coordinatesToPosition(row['经度'],row['纬度'])
    print(positionTemp)

# 生成第3个子图
gdf2 = gpd.GeoDataFrame(
    forecast_track, geometry=gpd.points_from_xy(forecast_track['经度'], forecast_track['纬度']), crs=4326)
ax3 = gdf2.plot(ax=axes[1,0],column='cluster', markersize=10, figsize=(10, 10), cmap='Set1')
ax3.set_title('出租车测试轨迹最有可能的轨迹预测结果可视化')

# 生成第4个子图
gdf3 = gpd.GeoDataFrame(
    destination, geometry=gpd.points_from_xy(destination['经度'], destination['纬度']), crs=4326)
ax4 = gdf3.plot(ax=axes[1,1],column='cluster', markersize=20, figsize=(10, 10), cmap='Set1')
ax4.set_title('出租车测试轨迹的目的地预测结果可视化')

# 叠加武汉市路网
wuhan_road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
wuhan_road.plot(ax=ax1, linewidth=0.5, alpha=0.5, color='grey')
wuhan_road.plot(ax=ax2, linewidth=0.5, alpha=0.5, color='grey')
wuhan_road.plot(ax=ax3, linewidth=0.5, alpha=0.5, color='grey')
wuhan_road.plot(ax=ax4, linewidth=0.5, alpha=0.5, color='grey')

# 设置子图范围
ax1.set_ylim([30.4,30.7])
ax1.set_xlim([114.15,114.45])
ax2.set_ylim([30.4,30.7])
ax2.set_xlim([114.15,114.45])
ax3.set_ylim([30.4,30.7])
ax3.set_xlim([114.15,114.45])
ax4.set_ylim([30.4,30.7])
ax4.set_xlim([114.15,114.45])

# 保存在路径data/processed/output_picture/下
plt.suptitle('201811 起始点在广埠屯附近的出租车轨迹目的地预测')
plt.savefig("data/processed/output_pictur"
            "e/起始点在洪山广场附近的出租车轨迹目的地预测201811")

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("\n程序“8出租车目的地预测”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“8出租车目的地预测”的主体代码执行的时间为：%d 秒" % maincode_seconds)