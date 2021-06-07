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
# 选取一定数量的出租车一天中的所有轨迹，降低数据运算开销
def select_cars(car_ids, size):
    # 参数：car_ids表示所有出租车ID的唯一值，size表示选取的数量，返回值为选取的出租车的ID列表
    # 生成需要选取的出租车id的索引号随机列表
    selected_indexs = rd.sample(range(1, len(car_ids)-1), size)
    # 去重之后重新生成list
    selected_indexs = list(set(selected_indexs))
    selected_car_ids = [car_ids[i] for i in selected_indexs]
    return selected_car_ids

# 根据抽取的出租车ID精简DataFrame
def get_selected_track_df(tdf, selected_ids):
    selected_dfs = []
    for Taxi_id in selected_ids:
        selected_dfs.append(tdf[tdf['出租车ID'] == Taxi_id])
    # 重新生成筛选后的轨迹DataFrame
    selected_tracks = pd.concat(selected_dfs)
    return selected_tracks

# 由于原有的轨迹数据量太大（近60,000条），计算时间过长，选取其中1,000条轨迹
def random_select_tracks(tracks, size):
    len_tracks = len(tracks)
    selected = []
    for i in range(int(1.05 * size)): selected.append(rd.randint(1,len(tracks)-1))  # 随机产生选取的轨迹
    selected = list(set(selected))  # 确定抽样的轨迹ID
    selected_tracks = [tra for tra in tracks if tra['id'] in selected]  # 随机抽取轨迹
    return selected_tracks

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

# 统计层次聚类中出现最频繁的轨迹类型，将其通过标记字段与其他类型轨迹区分开
def get_frequent_track_sch(labels, track_dicts, tdf):
    # 统计频繁轨迹所属的类，max()函数的key决定要按什么东西去比较大小，count()方法用于统计某个元素在列表中出现的次数
    max_label = max(labels, key=labels.count)
    # 统计频繁轨迹ID（TID），当轨迹所属的类为前面得到的频繁轨迹所属的类时，返回这一轨迹的TID号
    frequent_TID = [track_dicts[i]['track_id'] for i in range(0, len(track_dicts)) if track_dicts[i]['cluster'] == max_label]
    # 新增加一个字段'frequency'，并初始化为0，表示目前都为非频繁轨迹
    tdf['frequencySch'] = 0
    new_df = tdf
    # 频繁轨迹的frequency字段值更改为1
    for ids in frequent_TID:
        new_df.loc[(new_df.TID == ids), 'frequencySch'] = 1
    return new_df

# 统计kmeans聚类中出现最频繁的轨迹类型，将其通过标记字段与其他类型轨迹区分开
def get_frequent_track_kmeans(labels, track_dicts, tdf):
    # 统计频繁轨迹所属的类，max()函数的key决定要按什么东西去比较大小，count()方法用于统计某个元素在列表中出现的次数
    max_label = max(labels, key=labels.count)
    # 统计频繁轨迹ID（TID），当轨迹所属的类为前面得到的频繁轨迹所属的类时，返回这一轨迹的TID号
    frequent_TID = [track_dicts[i]['track_id'] for i in range(0, len(track_dicts)) if track_dicts[i]['kMeansCluster'] == max_label]
    # 新增加一个字段'frequency'，并初始化为0，表示目前都为非频繁轨迹
    tdf['frequencyKmeans'] = 0
    new_df = tdf
    # 频繁轨迹的frequency字段值更改为1
    for ids in frequent_TID:
        new_df.loc[(new_df.TID == ids), 'frequencyKmeans'] = 1
    return new_df

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
print("程序“7轨迹聚类”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

track_df = pd.read_csv('data/processed/output_data/含TID载客轨迹数据201811.csv')
# 定义一个用来存储选用的出租车ID的列表reserved_cars
reserved_cars = []
# 获取“含TID载客轨迹数据”文件中的所有出租车ID的唯一值
car_IDs = track_df['出租车ID'].unique().tolist()

# 选取一定数量的出租车的'出租车ID'列表,taxiCount为选择的出租车数量
taxiCount = 20
selected_car_IDs = select_cars(car_IDs, taxiCount)
# 根据抽取的出租车ID精简DataFrame
selected_tracks_by_car_id = get_selected_track_df(track_df, selected_car_IDs)

# 20辆出租车的数据集selected_tracks_by_car_id中所有轨迹的TID的唯一值列表
TIDs = selected_tracks_by_car_id['TID'].unique().tolist()
# 新的dataframe的字段名为：{"id","taxi","track_id","start","coords"}
track_Selected = get_track_coords(selected_tracks_by_car_id, TIDs)

# 输出taxiCount辆出租车包含的轨迹数量总和
print("选取的%d辆出租车包含的轨迹数量总和为：%d" % (taxiCount,len(track_Selected)))

# 计算出租车两两轨迹之间的距离矩阵
dtw_matrix = np_dtw_matrix(track_Selected)
# 保存为2位小数的浮点数，用逗号分隔，保存到data/processed/output_data/路径下的txt文件中
np.savetxt("data/processed/output_data/出租车两两轨迹之间的距离矩阵.txt", dtw_matrix, fmt="%.10f", delimiter=",")
# dtw_matrix = np.loadtxt('data/processed/output_data/出租车两两轨迹之间的距离矩阵.txt',delimiter=",")
print("出租车两两轨迹之间的距离矩阵为：")
print(dtw_matrix)

# 使用scipy.cluster.hierarchy.linkage进行层次聚类
# method是指计算类间距离的方法,比较常用的有3种:
# single:最近邻,把类与类间距离最近的作为类间距
# average:平均距离,UPGMA算法（非加权组平均）法
# ward:沃德方差最小化算法
Z1 = sch.linkage(dtw_matrix,method='average')

# 从给定链接矩阵定义的层次聚类中形成平面聚类
# scipy.cluster.hierarchy.fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)
# Z是上面生成的层级聚类的结果
# t是形成凝聚簇时要应用的阈值
# criterion是形成扁平簇的准则：inconsistent、distance、maxclust、monocrit、maxclust_monocrit
# inconsistent：如果一个群集节点及其所有后代的不一致值小于或等于t，则其所有叶后代都属于同一平面群集。如果没有非单例群集满足此条件，则将每个节点分配给其自己的群集
# maxclust：找到一个最小阈值r，以使同一平坦簇中的任意两个原始观测值之间的共形距离不超过r且不超过t个平坦簇
cluster1 = sch.fcluster(Z1, t=1, criterion='inconsistent')
Z2 = sch.linkage(dtw_matrix, method='ward')
cluster2 = sch.fcluster(Z2, t=25, criterion='maxclust')

# 获取k均值聚类结果，聚类数选择7
kmeans_value = KMeans(n_clusters=7).fit(dtw_matrix)

# 将分类的标签赋予对应的轨迹
for i in range(0, len(track_Selected)):
    # 此处以未限定最大分类数的聚类结果数的情况作为示例
    track_Selected[i]['cluster'] = cluster1.tolist()[i]
    track_Selected[i]['kMeansCluster'] = kmeans_value.labels_[i]

# 在实验轨迹数据中标记出聚类所得的频繁轨迹，分别使用层次聚类和kmeans聚类两种聚类方式
frequent_tracks_df_sch = get_frequent_track_sch(cluster1.tolist(), track_Selected, selected_tracks_by_car_id)
frequent_tracks_df_kmeans = get_frequent_track_kmeans((kmeans_value.labels_).tolist(), track_Selected, selected_tracks_by_car_id)

# 将频繁轨迹单独提取出来，即'frequencySch'或'frequencyKmeans'等于1的轨迹
frequent_tracks_sch_only = frequent_tracks_df_sch[frequent_tracks_df_sch['frequencySch'] == 1]
frequent_tracks_kmeans_only = frequent_tracks_df_kmeans[frequent_tracks_df_kmeans['frequencyKmeans'] == 1]

# 对层次聚类得到的频繁轨迹平均地理坐标做逆向地址解析
lng_average_frequentTracks_sch = 0.0
lat_average_frequentTracks_sch = 0.0
for index, row in frequent_tracks_sch_only.iterrows():
    lng_average_frequentTracks_sch += row['经度']
    lat_average_frequentTracks_sch += row['纬度']
lng_average_frequentTracks_sch /= len(frequent_tracks_sch_only)
lat_average_frequentTracks_sch /= len(frequent_tracks_sch_only)
position_frequentTracks_sch = coordinatesToPosition(lng_average_frequentTracks_sch,lat_average_frequentTracks_sch)
print('经过层次聚类得到的频繁轨迹数据点的平均地理坐标为：%f,%f' % (lng_average_frequentTracks_sch,lat_average_frequentTracks_sch))
print('经过逆向地址解析得到此点的地理位置信息为：')
print(position_frequentTracks_sch)
print('\n')

# 对k均值聚类得到的频繁轨迹平均地理坐标做逆向地址解析
lng_average_frequentTracks_kmeans = 0.0
lat_average_frequentTracks_kmeans = 0.0
for index, row in frequent_tracks_kmeans_only.iterrows():
    lng_average_frequentTracks_kmeans += row['经度']
    lat_average_frequentTracks_kmeans += row['纬度']
lng_average_frequentTracks_kmeans /= len(frequent_tracks_kmeans_only)
lat_average_frequentTracks_kmeans /= len(frequent_tracks_kmeans_only)
position_frequentTracks_kmeans = coordinatesToPosition(lng_average_frequentTracks_kmeans,lat_average_frequentTracks_kmeans)
print('经过k均值聚类得到的频繁轨迹数据点的平均地理坐标为：%f,%f' % (lng_average_frequentTracks_kmeans,lat_average_frequentTracks_kmeans))
print('经过逆向地址解析得到此点的地理位置信息为：')
print(position_frequentTracks_kmeans)

plt.rcParams['font.family']=['SimHei']
# 分割字图并设置整个图幅的大小为2000*1000
fig, axes = plt.subplots(1,2,figsize=(20,10))
# 使用geopandas可视化出租车轨迹
# 基于层次聚类的结果绘图
frequent_gdf_sch = gpd.GeoDataFrame(frequent_tracks_df_sch, geometry=gpd.points_from_xy(frequent_tracks_df_sch['经度'], frequent_tracks_df_sch['纬度']))
ax1 = frequent_gdf_sch.plot(ax=axes[0],column='frequencySch',cmap='coolwarm',legend=True,markersize=1,figsize=(10,10))
# 基于kmeans聚类的结果绘图
frequent_gdf_kmeans = gpd.GeoDataFrame(frequent_tracks_df_kmeans, geometry=gpd.points_from_xy(frequent_tracks_df_kmeans['经度'], frequent_tracks_df_kmeans['纬度']))
ax2 = frequent_gdf_kmeans.plot(ax=axes[1],column='frequencyKmeans',cmap='coolwarm',legend=True,markersize=1,figsize=(10,10))

# 叠加武汉市路网
wuhan_road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
wuhan_road.plot(ax=ax1, linewidth=0.5, alpha=0.5, color='grey')
wuhan_road.plot(ax=ax2, linewidth=0.5, alpha=0.5, color='grey')

plt.suptitle('201811 层次聚类和K均值聚类后的出租车的频繁轨迹可视化')
ax1.set_title('层次聚类后的出租车的频繁轨迹可视化')
ax2.set_title('kmeans聚类后的出租车的频繁轨迹可视化')
plt.savefig("data/processed/output_picture/出租车的频繁轨迹可视化201811")

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("\n程序“7轨迹聚类”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“7轨迹聚类”的主体代码执行的时间为：%d 秒" % maincode_seconds)