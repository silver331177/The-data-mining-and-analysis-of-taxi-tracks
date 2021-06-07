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
# Transformer.from_crs(from_crs, to_crs)用来进行坐标转换
# EPSG:4326   WGS84大地测量坐标系
# EPSG:32649  WGS84 UTM区域49N
wgs84 = pyproj.CRS.from_epsg(4326)
utm49N = pyproj.CRS.from_epsg(32649)
# transformer1表示从WGS84转化到投影坐标系，transformer2表示从投影坐标系转化到WGS84
transformer_Wgs84toUTM = Transformer.from_crs(wgs84, utm49N)
transformer_UTMtoWgs84 = Transformer.from_crs(utm49N, wgs84)

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
            # 将两次分界点之间的状态作为一段轨迹，取df的子集df[轨迹开始index1:轨迹结束idx2 + 1]
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

    # 由于最后一个时段没有转折点了，因此对最后一个时段的数据进行处理并归档
    sub_df = df[idx1:idx2 + 1]
    if old_status=='重车':
        loads.append(sub_df)
    else:
        no_loads.append(sub_df)

    return loads, no_loads, pick_up, drop_off

# 对一段时间的轨迹进行处理，执行线性插值
def interpolation(trac):
    trac_inter = pd.DataFrame()
    # 对太少的轨迹进行剔除
    if len(trac)<10:
        return trac_inter
    # 使用.loc[]的方法取轨迹开始点的'定位时间'列，并使用strptime()函数将字符串转换为日期时间
    time_before = datetime.datetime.strptime(trac.loc[0,'定位时间'], "%Y-%m-%d %H:%M:%S")

    '''
    除了线性插值，还可以使用：
    ① 抛物线插值，用恒定加速度进行估算；
    ② 多项式插值，加速度可以发生变化
    '''

    # 开始使用遍历器iterrows遍历这一段轨迹
    for index, row in trac.iterrows():
        # 向Dataframe添加新的行，且为了index不出现重现，设置不使用index标签，相当于重置索引，即ignore_index=True
        trac_inter = trac_inter.append(row,ignore_index=True)
        time_now = datetime.datetime.strptime(row['定位时间'], "%Y-%m-%d %H:%M:%S")

        # 获取时间差(t1-time_start)的秒数
        time_delta = (time_now-time_before).seconds
        # 180s内无数据，则干脆抛弃这条轨迹，即返回一个空的Dataframe即可
        if time_delta > 180:
            return pd.DataFrame()
        # 30s内无数据，则插值
        if time_delta > 30:
            # 每间隔15s进行一次插值
            new_points = pd.DataFrame()
            # n为插值的个数
            n = int(time_delta/15)
            for i in range(n):
                # datetime.timedelta(seconds=15)的意思为获取一个15秒的增量
                time_before = (time_before+datetime.timedelta(seconds=15))

                # 得到插值点的时间
                row['定位时间'] = time_before.strftime("%Y-%m-%d %H:%M:%S")
                # 从WGS84转化到投影坐标系：utm_x, utm_y
                utm_x, utm_y = transformer_Wgs84toUTM.transform(row['纬度'],row['经度'])

                # '速度'中存放的速度单位应为：km/h，因此转化为m/s，根据公式：'速度'*1000/60/60得到m/s再*15s得到变化的距离delta_distance
                delta_distance = float(row['速度'])*1000*15/60/60
                # 这块地方源码有一些问题，应该使用上方向角
                # '方向'字段表示的车辆的瞬时方向：0为整正北，180为正南，根据方向角计算各个方向的增量
                delta_y = delta_distance * math.cos(float(row['方向']))
                delta_x = delta_distance * math.sin(float(row['方向']))

                # 从投影坐标系转化到WGS84：lat,lng
                lat,lng = transformer_UTMtoWgs84.transform(utm_x+delta_x,utm_y+delta_y)
                # 得到插值点的经纬度
                row['纬度'] = lat
                row['经度'] = lng

                # 将新生成row插入到trac_inter中，出租车ID、速度、方向和状态都不变，插值点和前一个点的区别只有时间和经纬度
                trac_inter = trac_inter.append(row,ignore_index=True)
        time_before = time_now
    return trac_inter

# 使用均值滤波器对（插值后）的轨迹进行平滑
def mean_filter(trac,n):
    # 窗口大小应该是奇数，如果是偶数使它增加1，奇数则不变
    n = int(n/2)*2+1
    # 判断窗口大小，如果这一个时段的时间点还没有窗口大直接丢掉这个时段，返回一个空的DataFrame
    if len(trac)<n:
        return pd.DataFrame()

    '''
    这里使用的是均值滤波器，同理还可以构建：
    ①中值滤波，对于第i个轨迹点，使用以i为中心的n个点的坐标的中值估算i点的真值
    ②卡尔曼滤波，对于第i个轨迹点，使用前n个点的位置和速度估算i点的位置和速度
    '''

    trac_filter = trac
    # 使用窗口在轨迹中进行遍历
    for i in range(int(n/2),len(trac)-int(n/2)):
        # 取出窗口中左边int(n/2)个数据点，右边int(n/2)+1个点，对于窗口中间的点，进行均值滤波
        sub_trac = trac[i-int(n/2):i+int(n/2)+1]
        lat=0.0
        lng=0.0
        for index, row in sub_trac.iterrows():
            lat += float(row['纬度'])
            lng += float(row['经度'])
        # 取平均值
        lat/=n
        lng/=n
        # 将结果重新赋值给第i个数据点（即此窗口的中心点）
        trac_filter.loc[i,'纬度']=lat
        trac_filter.loc[i,'经度']=lng
    return trac_filter

# 不附加方向属性的，仅基于距离的路网匹配策略
# ①对每个轨迹片段，计算缓冲区；②将缓冲区与路网计算交集；
# ③对轨迹片段上的每个点，与上述交集的最近点即为匹配点
def mapping_without_direction(trac,radius,road):
    mapped = trac
    # 点转线
    if len(trac) == 0:
        return pd.DataFrame()

    # 这里的源代码在运行的时候报了个错，不能够识别str格式的数据，只能识别数值型
    # 因此对于series使用.astype(float)转化成浮点型数据，对于后面出现的str直接使用float()即可
    xylist = [xy for xy in zip(trac['经度'].astype(float),trac['纬度'].astype(float))]
    # 根据输入的点构造线
    line = LineString(xylist)

    # 构建路径的GeoDataFrame对象
    GeoDf = gpd.GeoDataFrame(geometry=[line],crs=4326)
    # 需要注意的是缓冲区的单位与矢量数据自带单位保持一致，例如以米为单位，因此需要将矢量数据转换为合适的投影坐标系之后，再进行缓冲区分析
    # WGS84大地坐标系（经纬度）转 UTM zone 49N:
    GeoDf = GeoDf.to_crs(32649)

    # 生成缓冲区，GeoSeries.buffer(distance,resolution=16)返回几何图形的一个GeoSeries，表示每个几何对象在给定的距离内的所有点
    gs_buffer = GeoDf.buffer(radius)
    # 将缓冲区从UTM zone 49N 转化到WGS84大地坐标系
    gs_buffer = gs_buffer.to_crs(4326)
    # 构建缓冲区的GeoDataFrame对象
    GeoDf_buffer = gpd.GeoDataFrame(geometry=gs_buffer)

    # 缓冲区的GeoDataFrame对象与路网求交
    # 使用叠加分析overlay()，对两个GeoDataFrame中全部的矢量对象两两之间进行基于集合关系的叠加分析
    # how：字符型，用于声明空间叠加的类型，有'intersection'，'union'、'symmetric_difference'、'difference'等
    # 默认保留与df1矢量类型相同的记录，在这里即保留与road矢量类型相同的记录
    # 这里使用'intersection'，即相交
    subroad = gpd.overlay(road, GeoDf_buffer, how='intersection').to_crs(32649)
    # 遍历轨迹点，计算距离
    # unary_union，返回的shapely对象会自动对存在重叠的矢量对象进行融合
    if subroad.geometry.unary_union.is_empty:
        return mapped

    for index, row in trac.iterrows():
        # 对每一个数据点生成一个几何图形的序列GeoSeries
        pt = gpd.GeoSeries([Point(float(row['经度']),float(row['纬度']))],crs=4326).to_crs(32649)
        npts = nearest_points(pt[0],subroad.geometry.unary_union)
        # 这里我们需要的是在路网上的点，而前面知道nearest_points的返回与输入顺序相同，因此取npts中的第二项，即npts[1]
        npt = npts[1]
        # 投影变换，从UTM投影坐标系转化到WGS84坐标系
        npt = transformer_UTMtoWgs84.transform(npt.x,npt.y)
        mapped.loc[index,'经度'] = npt[1]
        mapped.loc[index,'纬度'] = npt[0]
    return mapped

# 附加方向属性的路网匹配策略：
# ①对每个轨迹片段，计算缓冲区；②将缓冲区与路网计算交集；
# ③基于轨迹点的方向和路网的方向初步进行匹配，获取满足方向要求的路网子集；
# ④对轨迹片段上的每个点，与上述交集的最近点即为匹配点
def mapping_with_direction(trac,radius,road):
    mapped = trac
    # 点转线
    if len(trac) == 0:
        return pd.DataFrame()

    # 这里的源代码在运行的时候报了个错，不能够识别str格式的数据，只能识别数值型
    # 因此对于series使用.astype(float)转化成浮点型数据，对于后面出现的str直接使用float()即可
    xylist = [xy for xy in zip(trac['经度'].astype(float),trac['纬度'].astype(float))]
    # 根据输入的点构造线
    line = LineString(xylist)

    # 构建路径的GeoDataFrame对象
    GeoDf = gpd.GeoDataFrame(geometry=[line],crs=4326)
    # 需要注意的是缓冲区的单位与矢量数据自带单位保持一致，例如以米为单位，因此需要将矢量数据转换为合适的投影坐标系之后，再进行缓冲区分析
    # WGS84大地坐标系（经纬度）转 UTM zone 49N:
    GeoDf = GeoDf.to_crs(32649)

    # 生成缓冲区，GeoSeries.buffer(distance,resolution=16)返回几何图形的一个GeoSeries，表示每个几何对象在给定的距离内的所有点
    gs_buffer = GeoDf.buffer(radius)
    # 将缓冲区从UTM zone 49N 转化到WGS84大地坐标系
    gs_buffer = gs_buffer.to_crs(4326)
    # 构建缓冲区的GeoDataFrame对象
    GeoDf_buffer = gpd.GeoDataFrame(geometry=gs_buffer)

    # 缓冲区的GeoDataFrame对象与路网求交
    # 使用叠加分析overlay()，对两个GeoDataFrame中全部的矢量对象两两之间进行基于集合关系的叠加分析
    # how：字符型，用于声明空间叠加的类型，有'intersection'，'union'、'symmetric_difference'、'difference'等
    # 默认保留与df1矢量类型相同的记录，在这里即保留与road矢量类型相同的记录
    # 这里使用'intersection'，即相交
    subroad = gpd.overlay(road, GeoDf_buffer, how='intersection').to_crs(32649)
    # 遍历轨迹点，计算距离
    # unary_union，返回的shapely对象会自动对存在重叠的矢量对象进行融合
    if subroad.geometry.unary_union.is_empty:
        return mapped

    errorIndexList = []
    for index, row in trac.iterrows():
        # 对每一个数据点生成一个几何图形的序列GeoSeries
        pt = gpd.GeoSeries([Point(float(row['经度']),float(row['纬度']))],crs=4326).to_crs(32649)
        # 使用方向属性进行道路筛选，选出与数据点方向相同的道路子集，使得匹配更加精确
        Point_direction = row['方向']
        subroad_sub_bydirection = gpd.GeoDataFrame()
        for index2,row2 in subroad.iterrows():
            # 这里基于道路的方向矢量和数据点的方向矢量之间的夹角做道路方向匹配，使用到的公式为余弦定理
            # angle = (road_x * point_x) + (road_y * point_y)
            road_x = sin(row2['direction'])
            road_y = cos(row2['direction'])
            point_x = sin(row['方向'])
            point_y = cos(row['方向'])
            # 使用反三角函数acos()计算夹角
            angle = acos((road_x * point_x) + (road_y * point_y))
            # 若道路和数据点的方向与0度或180度（与道路方向反向，即沿着对向车流行驶）的夹角小于45度时认为方向是匹配的
            # 上述得到的角度是弧度制，45度角对应的是弧度为0.7853982，180度角对应的弧度为3.1415927
            if abs(angle) <= 0.7853982 or abs(angle-3.1415927) <= 0.7853982:
                subroad_sub_bydirection = subroad_sub_bydirection.append(row2)

        if len(subroad_sub_bydirection) == 0:
            print("存在错误，不存在与当前轨迹数据点方向相同的道路")
            errorIndexList.append(index)
            continue
        # 使用shapely的nearest_points方法计算点到线上的最近点，该点即为匹配点
        # shapely.ops.nearest_points(geoms1, geoms2)返回输入几何中最近点的元组, 与输入相同的顺序返回
        npts = nearest_points(pt[0],subroad.geometry.unary_union)
        # 这里我们需要的是在路网上的点，而前面知道nearest_points的返回与输入顺序相同，因此取npts中的第二项，即npts[1]
        npt = npts[1]
        # 投影变换，从UTM投影坐标系转化到WGS84坐标系
        npt = transformer_UTMtoWgs84.transform(npt.x,npt.y)
        mapped.loc[index,'经度'] = npt[1]
        mapped.loc[index,'纬度'] = npt[0]
    # print(errorIndexList)
    mapped.drop(errorIndexList)
    return mapped

# 测试for循环和apply循环的执行效率
def test_ForAndApply(test_data):
    # 开始测试for循环的执行效率
    # 经度的平均值
    test_count_avlng1 = 0
    # 纬度的平均值
    test_count_avlat1 = 0
    # 记录开始执行代码的时间，这里由于算的比较快，故使用毫秒计数
    time_test_For_start_microsecond1 = datetime.datetime.now().microsecond
    time_test_For_start_microsecond2 = time.mktime(datetime.datetime.now().timetuple())

    # 执行for循环
    for test_index, test_row in test_data.iterrows():
        test_count_avlng1 += test_row['经度']
        test_count_avlat1 += test_row['纬度']
    test_count_avlng1 = test_count_avlng1/len(data_track)
    test_count_avlat1 = test_count_avlat1/len(data_track)

    # 输出计算结果
    print("使用for循环计算的平均经度为：%f" % test_count_avlng1)
    print("使用for循环计算的平均纬度为：%f" % test_count_avlat1)

    # 记录结束执行代码的时间
    time_test_For_end_microsecond1 = datetime.datetime.now().microsecond
    time_test_For_end_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
    # 结束时间减去开始时间得到使用for循环的时间
    print('for循环测试执行一共用时：%fms' % float((time_test_For_end_microsecond2 - time_test_For_start_microsecond2) * 1000 +
                                       (time_test_For_end_microsecond1 - time_test_For_start_microsecond1) / 1000))

    # 开始测试apply循环的执行效率
    # 执行apply循环
    test_count_avlng2 = 0
    # 纬度的平均值
    test_count_avlat2 = 0
    # 记录开始执行代码的时间，这里由于算的比较快，故使用毫秒计数
    time_test_apply_start_microsecond1 = datetime.datetime.now().microsecond
    time_test_apply_start_microsecond2 = time.mktime(datetime.datetime.now().timetuple())

    test_data_sum = test_data.apply(lambda x: x.sum())
    test_count_avlng2 += test_data_sum['经度']
    test_count_avlat2 += test_data_sum['纬度']
    test_count_avlng2 = test_count_avlng2/len(data_track)
    test_count_avlat2 = test_count_avlat2/len(data_track)

    # 输出计算结果
    print("使用apply循环计算的平均经度为：%f" % test_count_avlng2)
    print("使用apply循环计算的平均纬度为：%f" % test_count_avlat2)

    # 记录结束执行代码的时间
    time_test_apply_end_microsecond1 = datetime.datetime.now().microsecond
    time_test_apply_end_microsecond2 = time.mktime(datetime.datetime.now().timetuple())
    # 结束时间减去开始时间得到使用for循环的时间
    print('apply循环测试执行一共用时：%fms' % float((time_test_apply_end_microsecond2 - time_test_apply_start_microsecond2) * 1000 +
                                         (time_test_apply_end_microsecond1 - time_test_apply_start_microsecond1) / 1000))

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“3轨迹数据插值+平滑+路网匹配”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

# 读取经过排序处理后的得到的轨迹数据csv
data_track = pd.read_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv')

# 选取某一辆出租车作为示例进行处理，修改taxiID_choose可以选择要选取的出租车ID
# 在这里使用的是随机数从轨迹数据集中任意选择一辆出租车进行分析，也可以直接手动设置taxiID_choose的值
taxiID_list = list(set(data_track['出租车ID'].tolist()))
print("当前出租车轨迹数据集涵盖出租车共 %d 辆" % len(taxiID_list))
taxiID_choose = taxiID_list[rd.sample(range(1, len(taxiID_list)-1), 1)[0]]
# taxiID_choose = 1165
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

# 对于某一条出租车轨迹，首先提取所有轨迹段
loads,no_loads,pick_up,drop_off = get_sub_trajectory(data_track_of_oneTaxi)
new_loads = pd.DataFrame()
new_no_loads = pd.DataFrame()

# 武汉市道路数据，第一个为全部的武汉市道路数据，第二个为武汉市三环以内的数据，在这里主要用到的是武汉市全部的道路数据
wuhan_All_road = gpd.GeoDataFrame.from_file('data/road/WuhanAllroad/osmWHmainRoad.shp')
wuhan_Part_road = gpd.GeoDataFrame.from_file('data/road/WuhanPartroad/WHroad.shp')

# 绘图数据准备
# 对于所有的loads和no_loads轨迹片段，分别进行插值，均值滤波
trac_all_interpolation_filter = pd.DataFrame()

'''
这里使用的是for循环配合遍历器iterrows对一段时间内轨迹的所有数据点进行处理，除此之外还可以使用Apply执行循环
Apply是pandas的一个常用函数，沿DataFrame的轴(行或列)执行功能func，从应用函数的返回类型推断出最终的返回类型，否则将取决于result_type参数
DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwds)
通常的用法是内接一个lambda匿名函数，从而对dataframe的每一行都进行循环处理，需要注意的是if或for或print等语句不能用于lambda中
简而言之，apply起到的作用就是将函数应用到由DataFrame的各列或行形成的一维数组上

这里以循环读取载客轨迹数据集loads作为测试，测试的内容为计算load数据集的所有数据点的平均经纬度
分别得到使用for循环和Apply方式循环的执行时间，结果显示使用for循环的程序执行时间和使用apply执行循环之间相差了将近4个数量级
由此可以基本推断，使用Apply方式循环的程序效率更高，耗时更少
'''
# 选取的数据集为上面的load数据集的经纬度子集
test_data = data_track[['经度','纬度']]
# 进行测试，在屏幕上输出执行的耗时
test_ForAndApply(test_data)

# 定义均值滤波的窗口大小为5
n = 5
for trac in loads:
    trac = trac.reset_index(drop=True)
    trac = interpolation(trac)
    trac_filter = mean_filter(trac,n)
    trac_all_interpolation_filter = trac_all_interpolation_filter.append(trac_filter,ignore_index=True)
for trac in no_loads:
    trac=trac.reset_index(drop=True)
    trac=interpolation(trac)
    trac_filter=mean_filter(trac,n)
    trac_all_interpolation_filter = trac_all_interpolation_filter.append(trac_filter,ignore_index=True)

# 对于所有的loads和no_loads轨迹片段，分别进行插值和地图匹配
trac_all_interpolation_mapping_with_direction = pd.DataFrame()
trac_all_interpolation_mapping_without_direction = pd.DataFrame()
dis = 10
for trac in no_loads:
    trac = trac.reset_index(drop=True)
    trac = interpolation(trac)
    # 进行附加方向属性的路网匹配
    trac_mapped_with_direction = mapping_with_direction(trac,dis,wuhan_All_road)
    trac_all_interpolation_mapping_with_direction = trac_all_interpolation_mapping_with_direction.append(trac_mapped_with_direction,ignore_index=True)
    # 进行不附加方向属性的路网匹配
    trac_mapped_without_direction = mapping_without_direction(trac, dis, wuhan_All_road)
    trac_all_interpolation_mapping_without_direction = trac_all_interpolation_mapping_without_direction.append(trac_mapped_without_direction, ignore_index=True)
for trac in loads:
    trac = trac.reset_index(drop=True)
    trac = interpolation(trac)
    # 进行附加方向属性的路网匹配
    trac_mapped_with_direction = mapping_with_direction(trac,dis,wuhan_All_road)
    trac_all_interpolation_mapping_with_direction = trac_all_interpolation_mapping_with_direction.append(trac_mapped_with_direction,ignore_index=True)
    # 进行不附加方向属性的路网匹配
    trac_mapped_without_direction = mapping_without_direction(trac, dis, wuhan_All_road)
    trac_all_interpolation_mapping_without_direction = trac_all_interpolation_mapping_without_direction.append(trac_mapped_without_direction, ignore_index=True)

# 绘制成多个子图，并保存在路径data/processed/output_picture/下
# 分割字图并设置整个图幅的大小为3000*1000
fig, axes = plt.subplots(1,3,figsize=(30,10))
plt.rcParams['font.family'] = ['SimHei']
# 绘制某编号出租车轨迹数据经过插值和滤波的结果
gdf = gpd.GeoDataFrame(
    trac_all_interpolation_filter,
    geometry=gpd.points_from_xy(trac_all_interpolation_filter['经度'], trac_all_interpolation_filter['纬度']))
# 绘图，分别指定渲染字段、颜色表、显示图例、点大小、图片大小
ax1 = gdf.plot(ax=axes[0],column='空车/重车',cmap='coolwarm',legend=True,markersize=1,figsize=(15,15))
ax1.set_ylim([30.4,30.8])
ax1.set_xlim([114.0,114.6])
# ax1.rcParams['font.family']=['SimHei']

# 绘制某编号出租车轨迹数据经过插值和不附带方向属性的地图匹配的结果
gdf = gpd.GeoDataFrame(
    trac_all_interpolation_mapping_without_direction,
    geometry=gpd.points_from_xy(trac_all_interpolation_mapping_without_direction['经度'], trac_all_interpolation_mapping_without_direction['纬度']))
ax2 = gdf.plot(ax=axes[1],column='空车/重车',cmap='coolwarm',legend=True,markersize=1,figsize=(15,15))
ax2.set_ylim([30.4,30.8])
ax2.set_xlim([114.0,114.6])
# ax2.rcParams['font.family']=['SimHei']

# 绘制某编号出租车轨迹数据经过插值和附带方向属性的地图匹配的结果
gdf = gpd.GeoDataFrame(
    trac_all_interpolation_mapping_with_direction,
    geometry=gpd.points_from_xy(trac_all_interpolation_mapping_with_direction['经度'], trac_all_interpolation_mapping_with_direction['纬度']))
ax3 = gdf.plot(ax=axes[2],column='空车/重车',cmap='coolwarm',legend=True,markersize=1,figsize=(15,15))
ax3.set_ylim([30.4,30.8])
ax3.set_xlim([114.0,114.6])
# ax3.rcParams['font.family']=['SimHei']

# 叠加武汉市路网
wuhan_All_road.plot(ax=ax1,linewidth=0.5,alpha=0.5,color='grey')
wuhan_All_road.plot(ax=ax2,linewidth=0.5,alpha=0.5,color='grey')
wuhan_All_road.plot(ax=ax3,linewidth=0.5,alpha=0.5,color='grey')

# 设置主标题和子标题
plt.suptitle('编号{taxiID}出租车轨迹数据经过数据插值等处理的结果显示'.format(taxiID=taxiID_choose),font={'family':'SimHei'})
ax1.set_title('编号{taxiID}出租车轨迹数据经过插值和滤波的结果'.format(taxiID=taxiID_choose),font={'family':'SimHei'})
ax2.set_title('编号{taxiID}出租车轨迹数据经过插值和不附带方向属性的地图匹配的结果'.format(taxiID=taxiID_choose),font={'family':'SimHei'})
ax3.set_title('编号{taxiID}出租车轨迹数据经过插值和附带方向属性的地图匹配的结果'.format(taxiID=taxiID_choose),font={'family':'SimHei'})

# 保存为图片
plt.savefig('data/processed/output_picture/出租车轨迹数据经过数据插值等处理的结果/编号{taxiID}出租车轨迹数据经过数据插值等处理的结果201811'.format(taxiID=taxiID_choose))

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("\n程序“3轨迹数据插值+平滑+路网匹配”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“3轨迹数据插值+平滑+路网匹配”的主体代码执行的时间为：%d 秒" % maincode_seconds)