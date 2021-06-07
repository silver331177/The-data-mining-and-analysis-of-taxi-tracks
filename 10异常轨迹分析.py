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
def look_into_df(df):
    # 输出数据的具体信息（数据量、出租车数、时空覆盖范围）查看
    print('选用的轨迹数据集共有 ',len(df),' 行数据')
    print('其中有不同出租车共 ',len(df['出租车ID'].unique()),' 辆')
    print('时间覆盖范围从 ',min(df['定位时间']),' 到 ',max(df['定位时间']))
    print('空间覆盖范围从 ',min(df['经度']),',',min(df['纬度']),' 到 ',max(df['经度']),',',max(df['纬度']))

def get_sub_df(df,posrange,timerange):
    print('按空间范围 ',posrange[0],' 到 ',posrange[1],'和时间范围 ',timerange[0],' 到 ',timerange[1],'取轨迹')
    sub_df = pd.DataFrame()
    # 首先先按空间范围筛选轨迹
    sub_df = df[(df['经度']>=posrange[0][0]) & (df['纬度']>=posrange[0][1]) & (df['经度']<=posrange[1][0]) & (df['纬度']<=posrange[1][1])]
    # 在第一次筛选的基础上按时间范围筛选轨迹
    sub_df = sub_df[(sub_df['定位时间']>=timerange[0]) & (sub_df['定位时间']<=timerange[1])]
    # 输出当前数据子集的具体信息
    look_into_df(sub_df)
    # 重设索引，去除重复
    sub_df = sub_df.reset_index(drop=True)
    return sub_df

def visualize(df):
    # 将数据叠加路网可视化查看，并将图片保存在data/processed/output_picture/下
    gdf= gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['经度'], df['纬度']),crs=4326)
    # 使matplotlib支持中文字体
    plt.rcParams['font.family']=['SimHei']
    ax=gdf.plot(markersize=1,figsize=(10,10))
    # 指定范围
    ax.set_ylim([min(df['纬度']),max(df['纬度'])])
    ax.set_xlim([min(df['经度']),max(df['经度'])])
    # 叠加武汉市路网
    road = gpd.GeoDataFrame.from_file('data/road/WuhanPartroad/WHroad.shp')
    road.plot(ax=ax,linewidth=0.5,alpha=0.5,color='grey')
    plt.savefig('data/processed/output_picture/提取一段时间和范围的出租车轨迹20181110')

# 轨迹分段函数
def get_sub_trac(df):
    tracks = []
    # 辅助记录
    idx1 = 0
    # loc()函数实现按行选取，这里的意思是选取索引值为idx1的那一行的'出租车ID'列
    tid1 = df.loc[idx1,'出租车ID']
    # 这里的意思是选取索引值为idx1的那一行的'定位时间'列，并转化成标准时间格式数据方便计算
    time1 = datetime.datetime.strptime(df.loc[idx1,'定位时间'], "%Y-%m-%d %H:%M:%S")
    # 使用迭代器循环读取df中的其他数据
    for index, row in df.iterrows():
        # 索引值
        idx2 = index
        # 出租车的TID号
        tid2 = row['出租车ID']
        # 定位时间
        time2 = datetime.datetime.strptime(row['定位时间'], "%Y-%m-%d %H:%M:%S")
        # 与idx1表示的车辆不同或者是时间间隔超过120秒
        if tid2 != tid1 or (time2-time1).seconds > 120:
            # 取出这一小段轨迹（idx1:idx2）
            sub_df = df[idx1:idx2]
            # 如果这段轨迹里面的数据点超过10个认为有效，将这一小段轨迹添加到tracks列表中
            if idx2-idx1 >= 10:
                tracks.append(sub_df)
            # 更新idx1的索引值、定位时间和出租车TID号为tid2的对应值
            idx1 = idx2
            time1 = time2
            tid1 = tid2
        else:
            time1 = time2
    # 对最后剩下的一小段轨迹进行处理
    if idx2-idx1 >= 9:
        sub_df = df[idx1:idx2+1]
    tracks.append(sub_df)
    return tracks

def get_track_coords(tdf, tIDs):
    # 存放轨迹中各点的经纬度坐标序列
    track_coords = []
    new_id = 1
    for ids in tIDs:
        # 每处理100条数据做一次中途反馈
        if ids % 100 == 0:
            print("数据还在跑，不要着急，当前时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        track_lons = tdf[tdf['TID'] == ids]['经度'].values.tolist()
        track_lats = tdf[tdf['TID'] == ids]['纬度'].values.tolist()
        # 重新为选取的轨迹编号，track_coords包括以下几个属性：
        # 'id'：轨迹的id号；'taxi'：出租车的id号；'track_id'：；'start'：轨迹的开始的时间点；'end'：轨迹的结束的时间点；'coords'：经纬度列表
        track_coords.append({
            'id': new_id,
            'taxi': tdf[tdf['TID'] == ids]['出租车ID'].values.tolist()[0],
            'track_id': ids,
            'start': tdf[tdf['TID'] == ids]['定位时间'].values.tolist()[0],
            'end': tdf[tdf['TID'] == ids]['定位时间'].values.tolist()[-1],
            'coords': [track_lons, track_lats]})
        new_id += 1
    return track_coords

# 计算dtw距离矩阵
def np_dtw_matrix(dataset):
    # 数据集中的轨迹总数
    N = len(dataset)
    # 初始化距离矩阵为零矩阵
    res = np.zeros((N,N),dtype='float32')
    for i in range(N):
        # 每处理50条数据做一次中途反馈
        if i % 50 == 0:
            print("还在计算距离矩阵，不要着急，当前时间为： %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))  # 较多时间用于动态得求出两条一般的轨迹的间距
        for j in range(i+1, N):
            # 分别计算在经纬度方向上的dtw距离
            dis_lon, _ = fastdtw(dataset[i]['coords'][0], dataset[j]['coords'][0], dist=euclidean)
            dis_lat, _ = fastdtw(dataset[i]['coords'][1], dataset[j]['coords'][1], dist=euclidean)
            res[i,j] = math.sqrt(dis_lat**2 + dis_lon**2)
            # 利用矩阵的对称性，通过求解矩阵的一半来加快部分运行的效率
            res[j,i] = res[i,j]
    return res

# 计算SSW和SSB
# SSW计算方式为，首先计算1和每个类的类内轨迹距离最小值的差，然后取SSW为所有类的最大值
# SSB计算方式为，首先计算不同类之间轨迹距离的最大值，然后用1与之相减
# SSW越小则类内距离越小，SSB越大则类间距离越大
def ssw_ssb(dtw_matr,cluster):
    m = list(set(cluster))
    ssw = 99999999.0
    sswt = 0.0
    ssb = 0.0
    # 循环处理每一个聚类，计算聚类内的统计指标ssw
    for i in m:
        # idx1的含义为：聚类号为i的轨迹的序号
        # 例如轨迹的聚类情况为[1 2 1 1 3 1]，idx1(i=1)=[0, 2, 3, 5]
        idx1 = [c for c in range(len(cluster)) if cluster[c] == i]
        # 如果这个聚类中只有一条轨迹时，指标sswt增加1
        if len(idx1) == 1:
            sswt += 1
        # 这个聚类中不只有一条轨迹，这个时候需要计算每个类的类内轨迹距离最小值
        else:
            for p in idx1:
                for q in idx1:
                    # 计算这个聚类中的两条不同轨迹的轨迹距离的最小值，后面再用1减去ssw，再加上前面聚类中只有一条轨迹时的sswt，即(1-ssw+sswt)才是真正的ssw
                    if p!=q:
                        if dtw_matr[p,q] < ssw:
                            ssw = dtw_matr[p,q]
        # 处理不同聚类之间的轨迹距离，计算聚类间的统计指标ssb
        for j in m:
            idx2 = [c for c in range(len(cluster)) if cluster[c] == j]
            for p in idx1:
                for q in idx2:
                    # 计算不同类之间轨迹距离的最大值，后面再用1与之相减得到ssb，即(1-ssb)才为真正的ssb
                    if dtw_matr[p,q]>ssb:
                        ssb=dtw_matr[p,q]
    # 这里指导代码的缩进有问题，导致了最后计算的ssw是负数，经过调整，往上减少一级缩进即可
    return 1-ssw+sswt,1-ssb

# 计算WB值，WB值的计算公式为：M*SSW(M)/SSB(M)
def WBindex(ssw,ssb,m):
    m = float(m)
    return m*ssw/ssb

# 计算CH值，wb值的计算公式为：(SSB(M)/(M-1))/(SSW(M)/(N-M))，其中M是聚类数，N是轨迹数
def CHindex(ssw,ssb,m,n):
    m = float(m)
    n = float(n)
    return (ssb/(m-1))/(ssw/(n-m))

# 计算Xu值，Xu值的计算公式为: log(sqrt(SSW(M)/N^2))+logM，其中M是聚类数，N是轨迹数
def Xuindex(ssw,m,n):
    ssw = float(ssw)
    m = float(m)
    n = float(n)
    # 对数计算的底取10
    return np.log10(sqrt(ssw/(n*n)))+np.log10(m)

def get_Tracks_TimeLenth(tracks):
    tracks_TimeLenth_df = pd.DataFrame(columns=['track_id', 'taxi', 'cluster', 'start', 'end', 'track_seconds', 'track_distance'])

    tracks_seconds = 0.0
    tracks_distance = 0.0
    for i in range(len(tracks)):
        # 轨迹开始的时间
        time_track_start = datetime.datetime.strptime(tracks[i]['start'], "%Y-%m-%d %H:%M:%S")
        # 轨迹结束的时间
        time_track_end = datetime.datetime.strptime(tracks[i]['end'], "%Y-%m-%d %H:%M:%S")
        # 轨迹结束时间减去开始时间得到轨迹行驶的时间
        track_seconds = (time_track_end - time_track_start).seconds

        # 统计所有轨迹的时间和
        tracks_seconds += track_seconds

        # 计算当前轨迹的行驶距离
        track_distance = 0.0
        for j in range(len(tracks[i]['coords'][0])-1):
            # 当前点的经度
            lon_early = tracks[i]['coords'][0][j]
            # 当前点的纬度
            lat_early = tracks[i]['coords'][1][j]
            # 后一个点的经度
            lon_late = tracks[i]['coords'][0][j+1]
            # 后一个点的纬度
            lat_late = tracks[i]['coords'][1][j+1]

            # 地球长半轴:6378千米
            earth_R = 6378137
            # 由于采样间隔时间较短，这里将两个数据点的直线距离视为出租车在这个采样间隔中行驶的轨迹
            # 由经纬度计算两点间的实际距离
            lat_early = lat_early * math.pi / 180.0
            lat_late = lat_late * math.pi / 180.0
            lat_diff = lat_early - lat_late
            lon_diff = (lon_early - lon_late) * math.pi / 180.0
            track_distance_P2P = 2 * earth_R * np.arcsin(sqrt(sin(lat_diff/2.0)*sin(lat_diff/2.0)+cos(lat_early)*cos(lat_late)*sin(lon_diff/2.0)*sin(lon_diff/2.0)))
            track_distance += track_distance_P2P

        # 将计算得到的当前轨迹的行驶距离累加得到整条轨迹的行驶距离
        tracks_distance += track_distance

        # 将每条轨迹的信息作为新的一行添加到tracks_TimeLenth_df中作为输出
        new_row = pd.DataFrame({
            'track_id': tracks[i]['track_id'],
            'taxi': tracks[i]['taxi'],
            'cluster': tracks[i]['cluster'],
            'start': tracks[i]['start'],
            'end': tracks[i]['end'],
            'track_seconds': track_seconds,
            'track_distance': track_distance
        }, index=[0])
        tracks_TimeLenth_df = tracks_TimeLenth_df.append(new_row,ignore_index=True)
    average_tracks_seconds = tracks_seconds / len(tracks)
    average_tracks_distance = tracks_distance / len(tracks)
    print('特定两点之间轨迹集的时间距离统计结果文件已经保存在 %s 路径下' % ('data/processed/output_data/'))
    tracks_TimeLenth_df.to_csv('data/processed/output_data/特定两点之间轨迹集的时间距离统计结果.csv')
    return tracks_TimeLenth_df,average_tracks_seconds,average_tracks_distance

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“10异常轨迹分析”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

df = pd.read_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv')
# 取涵盖武昌站和武汉站区域的2018-11-05早上7点到2018-11-10晚上21点的数据
sub_df = get_sub_df(df,[(114.25,30.5),(114.5,30.68)],['2018-11-01 00:00:00','2018-11-30 23:59:59'])
# 可视化查看
visualize(sub_df)
# 对经过时空范围限制的数据进行轨迹分段
tracks0 = get_sub_trac(sub_df)
track_WuhanToWuchang = pd.DataFrame()
tid = 1
# 遍历分段轨迹集中的每一个轨迹
for trac in tracks0:
    trac = trac.reset_index(drop=True)
    # 选取从武汉站附近出发，到达武昌站附近的轨迹，经纬度差在0.01以内的均认为在误差可接受范围内
    if abs(trac.loc[0,'经度']-114.43)<0.02 and abs(trac.loc[0,'纬度']-30.61)<0.02 and abs(trac.loc[len(trac)-1,'经度']-114.32)<0.02 and abs(trac.loc[len(trac)-1,'纬度']-30.53)<0.02:
        # 重新设置轨迹的编号，从1开始
        trac['TID'] = tid
        tid += 1
        track_WuhanToWuchang = track_WuhanToWuchang.append(trac)

# 重置索引
track_WuhanToWuchang = track_WuhanToWuchang.reset_index(drop=True)
# 获取轨迹编号的唯一值列表
tIDs = track_WuhanToWuchang['TID'].unique().tolist()
print('满足要求的从武汉站出发，到达武昌站的轨迹一共有 %d 条' % len(tIDs))
# 返回添加轨迹中各点的经纬度坐标序列之后的轨迹集tracks
tracks = get_track_coords(track_WuhanToWuchang,tIDs)

# 叠加路网显示从武汉站出发到达武昌站的所有出租车轨迹
gdf = gpd.GeoDataFrame(
        track_WuhanToWuchang, geometry=gpd.points_from_xy(track_WuhanToWuchang['经度'], track_WuhanToWuchang['纬度']),crs=4326)
# 使matplotlib支持中文字体
plt.rcParams['font.family'] = ['SimHei']
ax = gdf.plot(markersize=1,figsize=(10,10),column='TID')
# 指定范围
ax.set_ylim([(min(track_WuhanToWuchang['纬度'])-0.05),(max(track_WuhanToWuchang['纬度'])+0.05)])
ax.set_xlim([(min(track_WuhanToWuchang['经度'])-0.05),(max(track_WuhanToWuchang['经度'])+0.05)])
# 叠加武汉市路网
road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
road.plot(ax=ax,linewidth=0.5,alpha=0.5,color='grey')
plt.savefig('data/processed/output_picture/从武汉站出发到达武昌站的出租车轨迹201811')

# 计算tracks的dtw距离矩阵
dtw_matrix = np_dtw_matrix(tracks)

# 定义一个存储WB值、CH值和Xu值的列表
WBs = []
CHs = []
Xus = []
# 轨迹数，即计算CH值和Xu值中会使用到的N
n = len(tracks)

# possible为可能的聚类数，排除掉只分了1类，同时注意论文中说明的“超过100个集群是没有意义的”
# 因此写一个判断语句，若轨迹数大于100，则将100作为最大的聚类数
if len(tracks) > 100:
    maxPossible = 100
else:
    maxPossible = len(tracks)
possible = range(2,maxPossible)

# 基于WB、CH和Xu值探索最佳聚类效果
for i in possible:
    # 使用scipy.cluster.hierarchy.linkage进行层次聚类
    Z = sch.linkage(dtw_matrix,method='ward')
    # 从给定链接矩阵定义的层次聚类中形成平面聚类
    # 注意，此处criterion为’maxclust’时,t代表了最大的聚类的个数，例如t设置4则最大聚类数量为4类，当聚类满足4类的时候，迭代停止
    cluster = sch.fcluster(Z, t=i, criterion='maxclust')
    # 输出聚类结果
    print(cluster)
    # 计算SSW和SSB
    ssw, ssb = ssw_ssb(dtw_matrix, cluster.tolist())
    WB = WBindex(ssw,ssb,i)
    CH = CHindex(ssw,ssb,i,n)
    Xu = Xuindex(ssw,i,n)
    WBs.append(WB)
    CHs.append(CH)
    Xus.append(Xu)

# 指标值越小，聚类效果越好，输出最佳聚类的序号
print('使用WB值作为指标，最佳聚类数为%d' % possible[WBs.index(min(WBs))])
print('使用Xu值作为指标，最佳聚类数为%d' % possible[Xus.index(min(Xus))])
# 使用CH值作为指标应该是指标值越大、聚类效果越好
print('使用CH值作为指标，最佳聚类数为%d' % possible[CHs.index(max(CHs))])

# 若使用WB值作为评价指标，则possible[WBs.index(min(WBs))]为最终的聚类数
# 使用scipy.cluster.hierarchy.linkage进行层次聚类
Z = sch.linkage(dtw_matrix,method='ward')
# 从给定链接矩阵定义的层次聚类中形成平面聚类
# 注意，此处criterion为’maxclust’时,t代表了最大的聚类的个数，由于三个指标差别有点大，因此取三个指标的平均值作为t，当聚类满足的时候，迭代停止
t = (possible[WBs.index(min(WBs))]+possible[Xus.index(min(Xus))]+possible[CHs.index(max(CHs))])/3
print('最终使用的聚类数为：%d 个' % t)
cluster = sch.fcluster(Z, t=t, criterion='maxclust')

# 定义一个用于记录各个聚类中的轨迹条数的列表，cluster_item_count[0]对应的是聚类号为1的聚类
cluster_item_count = [0] * len(set(cluster.tolist()))
# 将分类的标签赋予对应的轨迹
for i in range(len(tracks)):
    tracks[i]['cluster'] = cluster.tolist()[i]
    # 这里注意到聚类号是从1开始的，而列表的下标是从0开始的，因此cluster_item_count[0]对应的是聚类号为1的聚类中的轨迹条数
    cluster_item_count[cluster.tolist()[i]-1] += 1

print('\n每个聚类中的轨迹数分别为：')
print(cluster_item_count)

anomalousTracks_index = []
for i in range(len(cluster_item_count)):
    if cluster_item_count[i] == 1:
        anomalousTracks_index.append(i+1)

print('\n异常轨迹的聚类号分别为：')
print(anomalousTracks_index)

# 叠加路网分别显示从武汉站出发到达武昌站的正常和异常的出租车轨迹
normalTracks_df = pd.DataFrame()
anomalousTracks_df = pd.DataFrame()
for i in range(len(tracks)):
    if tracks[i]['cluster'] in anomalousTracks_index:
        anomalousTracks_df = anomalousTracks_df.append(pd.DataFrame([tracks[i]])).reset_index(drop=True)
    else:
        normalTracks_df = normalTracks_df.append(pd.DataFrame([tracks[i]])).reset_index(drop=True)
# 使matplotlib支持中文字体
plt.rcParams['font.family'] = ['SimHei']
# 分割字图并设置整个图幅的大小为2000*1000
fig, axes = plt.subplots(1,2,figsize=(20,10))

# 武汉市路网
road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
ax1 = road.plot(ax=axes[0],linewidth=0.5,alpha=0.5,color='grey')
ax1.set_title("正常轨迹的可视化结果显示")
ax2 = road.plot(ax=axes[1],linewidth=0.5,alpha=0.5,color='grey')
ax2.set_title("异常轨迹的可视化结果显示")

for i in range(len(normalTracks_df)):
    normalTracks_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(normalTracks_df['coords'][i][0], normalTracks_df['coords'][i][1]),crs=4326)
    normalTracks_gdf.plot(ax=ax1, markersize=1, figsize=(10, 10), column='id', color='green')

for i in range(len(anomalousTracks_df)):
    anomalousTracks_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(anomalousTracks_df['coords'][i][0], anomalousTracks_df['coords'][i][1]),crs=4326)
    anomalousTracks_gdf.plot(ax=ax2, markersize=1, figsize=(10, 10), column='id', color='red')

# 指定范围
ax1.set_ylim([(min(track_WuhanToWuchang['纬度'])-0.05),(max(track_WuhanToWuchang['纬度'])+0.05)])
ax1.set_xlim([(min(track_WuhanToWuchang['经度'])-0.05),(max(track_WuhanToWuchang['经度'])+0.05)])
ax2.set_ylim([(min(track_WuhanToWuchang['纬度'])-0.05),(max(track_WuhanToWuchang['纬度'])+0.05)])
ax2.set_xlim([(min(track_WuhanToWuchang['经度'])-0.05),(max(track_WuhanToWuchang['经度'])+0.05)])
plt.savefig('data/processed/output_picture/从武汉站出发到达武昌站的正常和异常的出租车轨迹201811')

tracks_TimeLenth_df, average_tracks_seconds, average_tracks_distance = get_Tracks_TimeLenth(tracks)
print('\n经过统计，每条轨迹的经过的时间和距离统计结果如下所示')
print(tracks_TimeLenth_df)
print('所有轨迹的平均时间为：%f' % average_tracks_seconds)
print('所有轨迹的平均距离为：%f' % average_tracks_distance)

# 基于时间和路程两个指标对异常轨迹产生的原因进行的分析
for index,row in tracks_TimeLenth_df.iterrows():
    for i in range(len(anomalousTracks_index)):
        if row['cluster'] == anomalousTracks_index[i]:
            if row['track_distance'] <= average_tracks_distance and row['track_seconds'] <= average_tracks_seconds:
                print('属于聚类 %d 的异常轨迹被认定为第一类异常轨迹，即司机发现了一条距离短且耗时少的路径，可以作为推荐路线给其他司机使用' % (row['cluster']))
            if row['track_distance'] <= average_tracks_distance and row['track_seconds'] > average_tracks_seconds:
                print('属于聚类 %d 的异常轨迹被认定为第二类异常轨迹，即司机发现了一条距离短但耗时比较长的路径，适合于不赶时间，希望节省油耗的司机和乘客' % (row['cluster']))
            if row['track_distance'] > average_tracks_distance and row['track_seconds'] <= average_tracks_seconds:
                print('属于聚类 %d 的异常轨迹被认定为第三类异常轨迹，即司机发现了一条距离远但耗时比较短的路径，适合于赶时间的司机和乘客' % (row['cluster']))
            if row['track_distance'] > average_tracks_distance and row['track_seconds'] > average_tracks_seconds:
                print('属于聚类 %d 的异常轨迹被认定为第四类异常轨迹，即司机可能是由于主观上为了增加收入或者由于道路封闭不得不绕远路' % (row['cluster']))
                # 这里还可以进一步判断是否是由于高峰期绕路
                if row['start'] >= '2018-11-10 07:00:00' and row['end'] <= '2018-11-10 09:00:00':
                    print('且属于聚类 %d 的异常轨迹处于早高峰时段' % (row['cluster']))
                elif row['start'] >= '2018-11-10 17:00:00' and row['end'] <= '2018-11-10 19:00:00':
                    print('且属于聚类 %d 的异常轨迹处于晚高峰时段' % (row['cluster']))
                else:
                    print('且属于聚类 %d 的异常轨迹没有位于任何高峰时段' % (row['cluster']))

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("\n程序“10异常轨迹分析”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“10异常轨迹分析”的主体代码执行的时间为：%d 秒" % maincode_seconds)