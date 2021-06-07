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
# 对某辆出租车轨迹进行分段，提取上下车点，并输出载客和空载的轨迹以及上下车点【与前面的有些许区别，这里直接得到上下车点集】
def get_sub_trajectory(df):
    '''
    输入：
    df: 轨迹DataFrame。

    输出：
    loads,no_loads,ODs: 载客轨迹、空驶轨迹、上车点与下车点，实习用到后面两个。
    '''
    loads = []
    no_loads = []
    ODs = []

    # 记录每一段轨迹的开始
    idx1 = -1
    # 记录每一段轨迹的结束
    idx2 = -1
    # 记录原始的出租车状态：空车/重车
    old_status = ''

    for index, row in df.iterrows():
        status = row['空车/重车']
        # 初始化，当对索引号为0的时
        if index == 0:
            idx1 = index
            old_status = status

        # 判断状态是否转变，当发生了改变（status != old_status）时
        if status != old_status:
            # 将两次分界点之间的状态作为一段轨迹，取df的子集df[轨迹开始index1:轨迹结束idx2 + 1]
            sub_df = df[idx1:idx2 + 1]
            sub_list = sub_df.values.tolist()
            # 当状态为'重车'时，说明此时出租车是载客的状态，记录在loads轨迹集中
            if old_status == '重车':
                loads.append(sub_df)
                # sub_list[-1]表示这一段轨迹的最后一个数据点
                temp_d = sub_list[-1]
                # 将这个数据点记录为“下车点”，并添加到ODs中
                temp_d.append('下车点')
                ODs.append(temp_d)
            else:
                # 当状态不为‘重车’，即为‘空车’时，说明此时出租车是空车的状态，记录在no_loads数据集中
                no_loads.append(sub_df)
                # sub_list[0]表示这一段轨迹的第一个数据点
                temp_p = sub_list[0]
                # 将这个数据点记录为“上车点”，并添加到ODs中
                temp_p.append('上车点')
                ODs.append(temp_p)
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
    if old_status == '重车':
        loads.append(sub_df)
    else:
        no_loads.append(sub_df)
    return loads, no_loads, ODs

# 获取能够满足一天中一辆车的第一个OD点是上车点，最后一个OD点是下车点的数据区间
def find_indexes(od_list):
    daybegin = 0
    dayend = -1
    # 遍历整个od_list
    for i in range(0, len(od_list)):
        # 我们希望每一辆出租车的第一个上下车点为上车点，当点的属性为下车点时，继续
        if od_list[i][-1] == '下车点': pass
        else:
            # 如果点的属性不是下车点，则为上车点，这时将点号记录为daybegin，等待输出
            daybegin = i
            break
    for i in range(1, len(od_list)):
        # 同上，这里我们希望每一辆出租车的最后一个上下车点为下车点，当点的属性为上车点时，继续
        if od_list[len(od_list) - i][-1] == '上车点': pass
        else:
            # 如果点的属性不是下车点，则为上车点，这时将（点号+1）记录为dayend，目的是为了方便直接获取截取的子集段，等待输出
            dayend = len(od_list) - i + 1
            break
    return daybegin, dayend

# 去掉每辆车里“落单”的上车点、下车点
def align_ODs(od_list_sub):
    # 数据过少，直接抛弃掉这个出租车的数据
    if len(od_list_sub) < 2:return []
    else:
        # cur表示这辆出租车的第一个数据点的“上下车点”属性
        cur = od_list_sub[0][-1]
        # num表示需要去除下来的上下车数据点的index
        drop_nums = []
        # 从第二个数据开始，遍历整个某辆出租车的上下车点子集
        for i in range(1,len(od_list_sub)):
            # 如果和前一个数据点的“上下车点”属性相同
            if od_list_sub[i][-1] == cur:
                # 如果同为上车点的话，保留最晚的一个上车点，把较早的几个上车点index加入num
                if cur == '上车点':
                    drop_nums.append((i-1))
                # 如果同为下车点的话，保留最早的一个下车点，把较晚的几个下车点index加入num
                if cur == '下车点':
                    drop_nums.append(i)
            # 把当前数据点的“上下车点”属性记为cur，进入下一次循环
            cur = od_list_sub[i][-1]
        reserves = []
        for i in range(0,len(od_list_sub)):
            # 需要保留下来的上下车数据点，即index不在nums中
            if i not in drop_nums: reserves.append(od_list_sub[i])
        return reserves

# 清洗+对齐出租车的OD点
def filter_ODs(od_list):
    # 返回值car_sorted_ODs为经过清洗和对齐之后每一辆出租车的上下车点的字典
    # 返回值ODs_list为经过清洗和对齐之后所有出租车上下车点的列表
    # od_list的各字段名为：出租车ID,定位时间,经度,纬度,方向,速度,空车/重车
    # 利用循环遍历得到出租车的ID编号集合
    num_list = list(set([ol[0] for ol in od_list]))
    # 构建一个字典car_sorted_ODs，键即为出租车的ID，值暂时置为空
    car_sorted_ODs = {n:[] for n in num_list}

    # 以车辆为单位进行清洗，遍历上下车点列表，根据出租车ID将上下车点对应到每一个出租车ID上
    for ol in od_list: car_sorted_ODs[ol[0]].append(ol)
    # 确保一天中一辆车的第一个OD点是上车点，最后一个OD点是下车点
    for key in car_sorted_ODs:
        # 对每一辆车进行数据清洗，确保一天中一辆车的第一个OD点是上车点，最后一个OD点是下车点
        i_begin, i_end = find_indexes(car_sorted_ODs[key])
        # 根据上面获得的一天的起止点截取子集
        car_sorted_ODs[key] = car_sorted_ODs[key][i_begin:i_end]

    # 对齐序列中的“上车点”和“下车点”，去掉每辆车里“落单”的上车点、下车点
    for key in car_sorted_ODs:
        car_sorted_ODs[key] = align_ODs(car_sorted_ODs[key])

    ODs_list = []
    # 将所有出租车上下车点合在一起生成列表
    for key in car_sorted_ODs:
        for od in car_sorted_ODs[key]: ODs_list.append(od)
    return car_sorted_ODs, ODs_list

# haversine计算方法
def haversine(lonlat1, lonlat2):
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

# 过滤存在噪声点的OD点对
def drop_noise(OD_df):
    drop_indexes = []
    i = 0
    # 循环遍历整个OD_df列表
    while i < len(OD_df) - 1:
        # 当前上下车点对中只要有一个的热点类别为噪声点(-1)，将这两个点一起丢弃
        if OD_df.loc[i].tolist()[-1] == -1 or OD_df.loc[i+1].tolist()[-1] == -1:
            drop_indexes.append(i)
            drop_indexes.append(i+1)
        # 这里使循环控制变量直接+2，目的是为了每次可以取一对上下车点进行分析，如果其中一个出问题就丢掉这个上下车点对
        i += 2
    filter_df = OD_df.drop(index = drop_indexes)
    return filter_df

# 得到节点的列表
def get_nodes(df):
    '''
    节点的属性列表包含了热点的ID、经度、纬度以及访问的频次
    '''
    # 聚类（热点）类别的所有唯一值的列表[0,1,...]
    node_ids = list(set(df['热点类别'].tolist()))
    # 生成一个列表nodes，nodes内部的每一个元素都是列表，且内嵌列表的第一个元素为热点类别，例如[[0],[1],...]
    nodes = [[ids] for ids in node_ids]
    # 统计每一个热点类别下包含的节点数
    node_count = df['热点类别'].value_counts()
    # 为列表nodes添加其他的补充属性
    for node in nodes:
        node.append({
            # 聚类类别（即热点）的ID号
            'Node_class_ID': node[0],
            # 这个类别下的所有热点的经度均值
            'AverLon': round(df[df['热点类别'] == node[0]]['经度'].mean(), 6),
            # 这个类别下的所有热点的维度均值
            'AverLat': round(df[df['热点类别'] == node[0]]['纬度'].mean(), 6),
            # 这个类别下的热点数量
            'counts': node_count[node[0]]})
    # nodes的内容形如：[[0, {'Node_class_ID': 0, 'AverLon': 114.26951, 'AverLat': 30.614773, 'counts': 92}],[1,{}]...]
    return nodes

# 得到热点类别的唯一值列表
def get_edges(df):
    ids = list(set(df['热点类别'].tolist()))
    # 边的矩阵，类似于邻接矩阵
    edges = {}
    # 初始化边矩阵
    for i in range(0, len(ids)):
        for j in range(i, len(ids)):
            edges[(i, j)] = 0

    n = 0
    while n < len(df) - 1:
        # 计算边矩阵的权重，每一个上下车点对之间的路径，对应对称矩阵的一次计数，来回都计数
        e1 = (df.at[n, '热点类别'], df.at[n + 1, '热点类别'])
        e2 = (df.at[n + 1, '热点类别'], df.at[n, '热点类别'])
        if e1 in edges:
            edges[e1] += 1
        if e2 in edges:
            edges[e2] += 1
        # 取下一个上下车点对
        n += 2

    # NetworkX要求的带权边是“三元组”的形式，即(起点, 终点, 权重)
    edge_list = []
    for key in edges:
        # NetworkX有的算法要求不能有self-loop（自循环），且权重需要大于0
        if key[0] != key[1] and edges[key] > 0:
            edge_list.append((key[0], key[1], edges[key]))
    return edge_list

# 绘制网络
def paint_network(G, nodes):
    # 叠加武汉市路网
    wuhan_road = gpd.GeoDataFrame.from_file('data/road/WuhanPartroad/WHroad.shp')
    wuhan_road.plot(linewidth=0.5, alpha=0.5, color='grey')
    pos_dict = {}
    for node in nodes:
        pos_dict[node[0]] = (node[1]['AverLon'], node[1]['AverLat'])  # 以点的经纬度作为绘制的坐标
    nx.draw(G, node_size=30, pos=pos_dict, width=0.1)
    plt.title("201811 出租车OD热点空间交互网络")
    plt.savefig('data/processed/output_picture/出租车OD热点空间交互网络201811')
    # plt.show()

# 以NetworkX实现封装的社区检测算法为基础，初步探索复杂网络中的社团结构发现与分析方法
# 社区检测（community detection，又称社区发现、图聚类）即是用来揭示网络聚集行为的一种技术。
# 社区检测实际就是一种网络聚类的方法，这里的“社区”在文献中并没有一种严格的定义，我们可以将其理解为一类具有相同特性的节点的集合。
# 一般认为社团内部的点之间的连接相对稠密，而不同社团的点之间的连接相对稀疏。
def gn_community_detect(g):
    # greedy_modularity_communities 使用Clauset Newman-Moore贪婪的模块化最大化在图中查找社区
    # 返回值为每个社区的节点(在这里是聚类/热点)集，每个社区一个
    comp = community.greedy_modularity_communities(g,weight='weight')
    # 对每一个社区的节点(在这里是聚类/热点)集排序后生成元组graph_community
    graph_community = tuple(sorted(c) for c in comp)
    node_community = {}
    # 设置热点社区号的初始值为1，往后依次递增1
    cid = 1
    # 遍历每一个热点集
    for c in graph_community:
        # 遍历热点集中的每一个热点
        for v in c:
            # 逐个热点添加热点社区号
            node_community[v] = cid
        cid += 1
    return graph_community, node_community

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

# DBSCAN聚类调参
def DBSCAN_adjustParameters(distance_matrix,metric):
    '''
    Density-Based Spatial Clustering of Applications with Noise（具有噪声的基于密度的聚类方法）
    是一种基于密度的空间聚类算法，将具有足够密度的区域划分为簇，并在具有噪声的空间数据库中发现任意形状的簇，它将簇定义为密度相连的点的最大集合。
    传统的DBSCAN密度聚类算法，需要：邻域阈值(Eps)和点数阈值(min_samples)2个参数来对数据集进行聚类
    fit_predict()从要素或距离矩阵执行DBSCAN聚类，并返回聚类标签

    eps：DBSCAN算法参数，即我们的ϵ-邻域的距离阈值，和样本距离超过ϵ的样本点不在ϵ-邻域内。默认值是0.5，一般需要通过在多组值里面选择一个合适的阈值。
    eps过大，则更多的点会落在核心对象的ϵ-邻域，此时我们的类别数可能会减少，本来不应该是一类的样本也会被划为一类。
    反之则类别数可能会增大，本来是一类的样本却被划分开。

    min_samples：DBSCAN算法参数，即样本点要成为核心对象所需要的ϵ-邻域的样本数阈值。默认值是5，一般需要通过在多组值里面选择一个合适的阈值。
    通常和eps一起调参，在eps一定的情况下，min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多。
    反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少。
    '''
    result = []
    # 迭代次数，由于调参的过程耗时比较久，需要在中途给出适当的反馈
    iterateTime = 0
    # 迭代不同的eps值
    for eps in np.arange(175, 225, 1):
        # 迭代不同的min_samples值
        for min_samples in range(5, 16):
            DB_cluster_label = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(distance_matrix)
            ssw, ssb = ssw_ssb(distance_matrix, DB_cluster_label.tolist())
            # 统计各参数组合下的聚类个数（-1表示异常点）
            n_clusters = len([i for i in set(DB_cluster_label) if i != -1])
            WB = WBindex(ssw, ssb, n_clusters)
            result.append({'eps': eps,
                           'min_samples': min_samples,
                           'n_clusters': n_clusters,
                           'WBindex': WB,
                           })
            iterateTime += 1
            if iterateTime % 10 == 0:
                timeNow = time.localtime()
                print('已经测试了 %d 组参数，调参还在进行中，请稍后，当前时间为：%s' % (iterateTime,time.strftime("%Y-%m-%d %H:%M:%S", timeNow)))
    # 将迭代后的结果存储到数据框中
    result_df = pd.DataFrame(result)
    WBindex_min = result_df.loc[:, "WBindex"].min()
    best_result = result_df[result_df['WBindex'] == WBindex_min].reset_index(drop=True)
    eps_best = best_result['eps'][0]
    min_samples_best = best_result['min_samples'][0]
    return result_df, eps_best, min_samples_best

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“9构建出租车OD热点空间交互网络”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

# 读取数据，这里直接读取经过轨迹数据预处理后得到的数据
# track_data
track_data = pd.read_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv')
# 提取出track_data中的上下车点ODs数据
taxi_ODs = get_sub_trajectory(track_data)[2]

# 清洗+对齐出租车的OD点
filtered_ODs, filtered_ODs_list = filter_ODs(taxi_ODs)
# 将得到的所有出租车上下车点的列表构建DataFrame并输出中间文件保存，路径为：data/processed/output_data/
cols1 = ['TaxiID', '时间', '经度', '纬度', '速度', '方向角', '空车 / 重车', '上下车点']
ODs_df = pd.DataFrame(filtered_ODs_list, columns=cols1)
ODs_df.to_csv('data/processed/output_data/不含热点类别的出租车OD点201811.csv', index=False, encoding="utf_8_sig")
# ODs_df = pd.read_csv('data/processed/output_data/不含热点类别的出租车OD点201811.csv')

# 尝试使用整个数据集，发现数据集过大，报了如下这个错误
# numpy.core._exceptions.MemoryError: Unable to allocate 70.5 GiB for an array with shape (9462133395,) and data type float64
# 为了缩短运行时间，为了节省内存，取前12000条数据进行处理
sample_OD_df = ODs_df[:12000]

# dropna()能够找到DataFrame类型数据的空值，将空值所在的行/列删除后，将新的DataFrame作为返回值返回
# how：筛选方式：any，表示该行/列只要有一个以上的空值，就删除该行/列；all，表示该行/列全部都为空值，就删除该行/列
# axis：检查的轴：0或index，表示按行删除；1或columns，表示按列删除
lat_lon_df = sample_OD_df[['纬度', '经度']].dropna(axis=0, how='all')
# scipy.spatial.distance.pdist(X, metric='euclidean', *args, **kwargs)用来计算距离矩阵的函数，这里自定义了计算方法haversine(u,v)
# scipy.spatial.distance.squareform(X, force=’no’, checks=True)用来压缩矩阵，在这里是将一个向量格式的距离向量转换成一个方阵格式的距离矩阵
# squareform用来把一个向量格式的距离向量转换成一个方阵格式的距离矩阵，反之亦然。输入如果是矩阵的话必须是距离矩阵，如果输入的是距离向量的话，必须满足d * (d-1) / 2.
distance_matrix = squareform(pdist(lat_lon_df, (lambda u, v: haversine(u, v))))

# DBSCAN调参
result_df, eps_best, min_samples_best = DBSCAN_adjustParameters(distance_matrix,'precomputed')
# 输出调参结果
print('选用的eps参数为：%f ，选用的min_samples参数为：%d \n' % (eps_best,min_samples_best))
# DBSCAN聚类，使用上面经过调参得到的最优eps和min_samples
od_label = DBSCAN(eps=eps_best, min_samples=min_samples_best, metric='precomputed').fit_predict(distance_matrix)
# od_label = DBSCAN(eps=190, min_samples=6, metric='precomputed').fit_predict(distance_matrix)
od_label = od_label.tolist()  # 使用DBSCAN获得的热点聚类标签

# 计算噪声点的数量，在使用Sklearn进行DBSCAN聚类时，“-1”标签表示未能划分到任何类簇中的噪声点
noise = 0
for l in od_label:
    if l == -1:
        noise += 1

# 确定出租车OD热点类簇的类型
label_str = [str(ol) for ol in list(set(od_label))]
print('聚类后热点类簇Label：\n' + ', '.join(label_str))

# 增加“热点类别”字段，并将聚类结果添加到“热点类别”字段中
sample_OD_df['热点类别'] = od_label
# 过滤存在噪声点的OD点对
filtered_df = drop_noise(sample_OD_df)
# 重新设置索引
filtered_df.reset_index(drop=True, inplace=True)
filtered_df.to_csv('data/processed/output_data/含热点类别的清噪后出租车OD点201811.csv', index=False, encoding="utf_8_sig")
# filtered_df = pd.read_csv('data/processed/output_data/含热点类别的清噪后出租车OD点201811.csv')
print('清洗噪声后OD点数：' + str(len(filtered_df)))

plt.rcParams['font.family'] = ['SimHei']
# 分割字图并设置整个图幅的大小为2000*1000
fig, axes = plt.subplots(1,2,figsize=(20,10))

# 上下车点的热点聚类结果显示，设置为子图1
mpl.rcParams["font.sans-serif"] = ["SimHei"]
cluster_gdf = gpd.GeoDataFrame(filtered_df, geometry=gpd.points_from_xy(filtered_df['经度'], filtered_df['纬度']))
ax1 = cluster_gdf.plot(ax=axes[0],column='热点类别',cmap='jet',legend=True,markersize=1,figsize=(10,10))
wuhan_road = gpd.GeoDataFrame.from_file('data/road/WuhanPartroad/WHroad.shp')
wuhan_road.plot(ax=ax1, linewidth=0.5, alpha=0.5, color='grey')
ax1.set_title("201811 上下车点的热点聚类结果显示")

# 输出按聚类结果划分的各个聚类的平均经纬度以及此点的地理位置逆向解析结果
average_lon_cluster = [0] * (len(set(sample_OD_df['热点类别']))-1)
average_lat_cluster = [0] * (len(set(sample_OD_df['热点类别']))-1)
cluster_count = [0] * (len(set(sample_OD_df['热点类别']))-1)
for index, row in filtered_df.iterrows():
    # 对每个聚类（热点列表）的热点经纬度进行求和
    v = row['热点类别']
    average_lon_cluster[v] += row['经度']
    average_lat_cluster[v] += row['纬度']
    cluster_count[v] += 1
for i in range(len(average_lon_cluster)):
    # 取平均并输出结果信息
    if cluster_count[i] == 0:
        continue
    average_lon_cluster_temp = average_lon_cluster[i] / cluster_count[i]
    average_lat_cluster_temp = average_lat_cluster[i] / cluster_count[i]
    print('聚类号为 %d 的热点聚类中的热点平均经纬度为: (%f,%f)' % (i,average_lon_cluster_temp,average_lat_cluster_temp))
    print('聚类号为 %d 的热点聚类中的热点的平均地理位置逆向解析结果为：' % i)
    print(coordinatesToPosition(average_lon_cluster_temp,average_lat_cluster_temp))

# 获取节点
nodes = get_nodes(filtered_df)
# 获取边
edges = get_edges(filtered_df)
# 开始构建空网络
G = nx.Graph()
# 分别填入节点和边到网络中
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edges)
# 对网络进行拓扑可视化展示
# paint_network(G, nodes)

# 获取社区的划分结果
OD_Community1,node_community = gn_community_detect(G)
print('社区划分的个数：' + str(len(OD_Community1)) + '\n\n各个社区的所属：')
# 遍历社区集中的每一个社区，逐个输出每个社区所包括的热点类别
for c in OD_Community1:
    print(c)

# 使用迭代器遍历清洗噪声后OD点数据集
for index, row in filtered_df.iterrows():
    v = row['热点类别']
    # 根据热点类别得到这个OD点所属的热点社区，并将热点社区号作为新字段添加进去
    filtered_df.loc[index,'热点社区'] = node_community[v]

# 热点社区的空间分布，设置为子图2，和子图1一起绘制显示并保存图片在路径data/processed/output_picture/下
community_gdf = gpd.GeoDataFrame(filtered_df, geometry=gpd.points_from_xy(filtered_df['经度'], filtered_df['纬度']))
ax2 = community_gdf.plot(ax=axes[1],column='热点社区',cmap='hsv',legend=True,markersize=1,figsize=(10,10))  # 必要的绘图设置
wuhan_road = gpd.GeoDataFrame.from_file('data/road/WuhanPartroad/WHroad.shp')  # 叠加武汉市路网
wuhan_road.plot(ax=ax2, linewidth=0.5, alpha=0.5, color='grey')
ax2.set_title("201811 热点社区的空间分布")
plt.suptitle('201811 上下车点的热点聚类和热点社区的空间分布')
plt.savefig('data/processed/output_picture/上下车点的热点聚类和热点社区的空间分布201811')

plt.clf()
# 对网络进行拓扑可视化展示
paint_network(G, nodes)

# 表示节点重要性的点权，即度（Strength）
# g.degree()能够获得DegreeView对象
# 对于无向图，顶点的度是指跟顶点相连的边的数量；
# 对于有向图，顶点的图分为入度和出度，朝向顶点的边称作入度；背向顶点的边称作出度
strength = dict(G.degree(weight='weight'))

# 让matplotlib支持中文字体的显示
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置绘图的分辨率，默认绘制的图形不够清晰
plt.figure(figsize=(36,27))
# 设置x轴为各个节点的列表
# 注意除去0号节点（城市中心区域OD点稠密，形成范围较大的热点区域）
X = [i for i in range(0,len(strength))][1:]
# 设置y轴（绘制的数据）为空间交互网络各个节点的强度
Y = [strength[key] for key in strength][1:]
# 绘图
fig = plt.figure()

plt.bar(X,Y,1,color="steelblue")
plt.xlabel("节点ID",fontsize=13)
plt.ylabel("节点交互强度",fontsize=13)
# plt.grid(True)
plt.title("出租车空间交互网络节点强度",fontsize=13)
plt.savefig('data/processed/output_picture/出租车空间交互网络节点强度201811')

# 基于出租车交互网络节点强度进行探索
# 设置高强度节点的强度阈值为30
strength_threshold = 30
# 高强度节点的节点号列表
high_strength = []
# 高强度节点中的所有OD点
high_strength_points_df = pd.DataFrame()

for i in set(list(strength.keys())):
    # 将节点强度超过阈值的节点视为高强度节点，获取节点号
    if strength[i] >= strength_threshold:
        high_strength.append(list(strength.keys())[list(strength.values()).index(strength[i])])
print('\n')

for i in range(len(high_strength)):
    # 节点的经纬度
    AverLon = nodes[high_strength[i]][1]['AverLon']
    AverLat = nodes[high_strength[i]][1]['AverLat']
    print('第 %d 号节点为高强度节点，节点中各个热点的平均经纬度为：(%f,%f)' % (high_strength[i],AverLon,AverLat))
    print('此节点的在实际中的逆向地址解析结果为：')
    print(coordinatesToPosition(AverLon,AverLat))
    high_strength_points_df = high_strength_points_df.append(filtered_df[filtered_df['热点类别'] == high_strength[i]])

# 获取高强度节点中的所有OD点的GeoDataFrame以及所有OD点（包括非高强度节点中的OD点）的GeoDataFrame
# 为了防止出错，重置两个使用的dataframe的索引
high_strength_points_df = high_strength_points_df.reset_index(drop=True)
filtered_df = filtered_df.reset_index(drop=True)
high_strength_points_gdf = gpd.GeoDataFrame(high_strength_points_df, geometry=gpd.points_from_xy(high_strength_points_df['经度'], high_strength_points_df['纬度']),crs=4326)
all_points_gdf = gpd.GeoDataFrame(filtered_df, geometry=gpd.points_from_xy(filtered_df['经度'], filtered_df['纬度']),crs=4326)
# 读入武汉行政区划数据
wuhan_region = gpd.GeoDataFrame.from_file('data/武汉行政区划数据/武汉行政区划矢量图.shp')
# OD点GeoDataFrame和行政区划数据的坐标系转成 WGS84大地坐标系(4326):
wuhan_region = wuhan_region.to_crs(4326)

# 新建一个保存每个行政区划内的OD点数目的字典
ODs_in_wuhan_region_Count_high_strength = {}
ODs_in_wuhan_region_Count_all = {}

for i in range(len(wuhan_region)):
    region_geo = wuhan_region.geometry[i]
    name = wuhan_region.locname[i]
    count_temp_high_strength = 0
    count_temp_all = 0

    # 统计高强度节点中的所有OD点的空间分布情况
    for j in range(len(high_strength_points_gdf)):
        point_geo = high_strength_points_gdf.geometry[j]
        if point_geo.intersects(region_geo):
            count_temp_high_strength += 1

    # 统计所有节点（高强度、非高强度）中的所有OD点的空间分布情况
    for j in range(len(all_points_gdf)):
        point_geo = all_points_gdf.geometry[j]
        if point_geo.intersects(region_geo):
            count_temp_all += 1

    # 将统计结果按武汉行政区划保存到字典中
    ODs_in_wuhan_region_Count_high_strength[name] = count_temp_high_strength
    ODs_in_wuhan_region_Count_all[name] = count_temp_all

# 输出节点OD点的空间分布统计结果
print('\n高强度节点中的所有OD点的空间分布情况为：')
print(ODs_in_wuhan_region_Count_high_strength)
print('按照空间分布密集程度进行排序的结果为：')
print(sorted(ODs_in_wuhan_region_Count_high_strength.items(), key=lambda kv:(kv[1], kv[0]),reverse=True))

print('\n所有节点中的所有OD点的空间分布情况为：')
print(ODs_in_wuhan_region_Count_all)
print('按照空间分布密集程度进行排序的结果为：')
print(sorted(ODs_in_wuhan_region_Count_all.items(), key=lambda kv:(kv[1], kv[0]),reverse=True))
print('\n')

# 将OD点数据合并到武汉市行政区划数据中
ODCount_df = pd.DataFrame.from_dict(ODs_in_wuhan_region_Count_high_strength, orient='index', columns=['ODCount_High'])
ODCount_df = ODCount_df.append(pd.DataFrame.from_dict(ODs_in_wuhan_region_Count_all, orient='index', columns=['ODCount_All']))
ODCount_df = ODCount_df.reset_index().rename(columns={'index': 'locname'})
ODCount_gdf = gpd.GeoDataFrame(ODCount_df)
wuhan_region_withdata = wuhan_region.merge(ODCount_gdf, on='locname', how='left')

# 绘图显示
plt.rcParams['font.family'] = ['SimHei']
# 分割字图并设置整个图幅的大小为2000*1000
fig, axes = plt.subplots(1,2,figsize=(20,10))
# k为显示的颜色数量
ax1 = wuhan_region_withdata.plot(ax=axes[0],column='ODCount_High', k=8, cmap='Reds', legend=True)
ax1.set_title("高强度节点中的所有OD点的空间分布情况")
ax2 = wuhan_region_withdata.plot(ax=axes[1],column='ODCount_All', k=8, cmap='Reds', legend=True)
ax2.set_title("所有节点中的所有OD点的空间分布情况")
plt.suptitle('201811 OD点的空间分布情况')
plt.savefig('data/processed/output_picture/OD点的空间分布情况201811')
# 基于出租车交互网络节点强度进行探索到这里结束

# 基于出租车交互网络的节点中心性进行探索
# 节点介数中心系数，中介中心性指的是一个结点担任其它两个结点之间最短路的桥梁的次数。一个结点充当“中介”的次数越高，它的中介中心度就越大
node_betweenness_centrality = nx.betweenness_centrality(G)
# 节点的度中心性是它所连接的节点的分数，一个节点的节点度越大就意味着这个节点的度中心性越高，该节点在网络中就越重要
node_degree_centrality = nx.degree_centrality(G)
# 节点的接近中心性，一个点的近性中心度较高，说明该点到网络中其他各点的距离总体来说较近，反之则较远
node_closeness_centrality = nx.closeness_centrality(G, u=None, distance=None, wf_improved=True)

# 将三个中心性测度值合并成dataframe
centrality_list = [node_betweenness_centrality,node_degree_centrality,node_closeness_centrality]
centrality_df = pd.DataFrame(centrality_list)
# 重置索引名并转置
centrality_df.index = ['node_betweenness_centrality','node_degree_centrality','node_closeness_centrality']
centrality_df = centrality_df.transpose()
# 如果需要对根据两个指标进行排序的话，可以参考以下语句，下面会用到
# centrality_df.sort_values(["node_betweenness_centrality", "node_degree_centrality"], inplace=True, ascending=True)

# 设置x轴,注意除去0号节点（城市中心区域OD点稠密，形成范围较大的热点区域）
X = [i for i in range(0,len(node_betweenness_centrality))][1:]
# 设置y轴（绘制的数据）为空间交互网络各个节点的三种中心性测度
y1 = [node_betweenness_centrality[key] for key in node_betweenness_centrality][1:]
y2 = [node_degree_centrality[key] for key in node_degree_centrality][1:]
y3 = [node_closeness_centrality[key] for key in node_closeness_centrality][1:]

plt.figure(figsize=(32,24))
# 设置x轴刻度标签位置
x = np.arange(len(X))
# 每个柱子的宽度
width = 0.25
# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# x - width，x， x + width即每组数据在x轴上的位置
plt.bar(x - width, y1, width, label='betweenness_centrality')
plt.bar(x, y2, width, label='degree_centrality')
plt.bar(x + width, y3, width, label='closeness_centrality')
plt.ylabel('centrality', fontsize=36)
plt.xlabel('节点序号', fontsize=36)
plt.title('节点的三种度中心性测度柱状图显示结果', fontsize=48, pad=20)
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=X)
plt.legend()
plt.savefig('data/processed/output_picture/出租车空间交互网络节点的三种中心性测度柱状图201811')

# 柱状图看得不直观，这里使用散点图加拟合三次曲线进行分析，三个测度两两之间成一幅图
# 准备拟合的三次曲线的数据
# 介数中心性与度中心性，获取二者之间散点数据拟合的三次曲线的参数，并得到三次曲线
centrality_df.sort_values(["node_betweenness_centrality", "node_degree_centrality"], inplace=True, ascending=True)
x_betweenness_degree = np.array(list(centrality_df['node_betweenness_centrality']))
y_betweenness_degree = np.array(list(centrality_df['node_degree_centrality']))
parameter_betweenness_degree = np.polyfit(x_betweenness_degree, y_betweenness_degree, deg=3)
betweenness_degree_line3 = parameter_betweenness_degree[0] * x_betweenness_degree ** 3 + parameter_betweenness_degree[1] * x_betweenness_degree ** 2 + parameter_betweenness_degree[2] * x_betweenness_degree + parameter_betweenness_degree[3]

# 介数中心性与接近中心性，获取二者之间散点数据拟合的三次曲线的参数
centrality_df.sort_values(["node_betweenness_centrality", "node_closeness_centrality"], inplace=True, ascending=True)
x_betweenness_closeness = np.array(list(centrality_df['node_betweenness_centrality']))
y_betweenness_closeness = np.array(list(centrality_df['node_closeness_centrality']))
parameter_betweenness_closeness = np.polyfit(x_betweenness_closeness, y_betweenness_closeness, deg=3)
betweenness_closeness_line3 = parameter_betweenness_closeness[0] * x_betweenness_closeness ** 3 + parameter_betweenness_closeness[1] * x_betweenness_closeness ** 2 + parameter_betweenness_closeness[2] * x_betweenness_closeness + parameter_betweenness_closeness[3]

# 度中心性与接近中心性，获取二者之间散点数据拟合的三次曲线的参数
centrality_df.sort_values(["node_degree_centrality", "node_closeness_centrality"], inplace=True, ascending=True)
x_degree_closeness = np.array(list(centrality_df['node_degree_centrality']))
y_degree_closeness = np.array(list(centrality_df['node_closeness_centrality']))
parameter_degree_closeness = np.polyfit(x_degree_closeness, y_degree_closeness, deg=3)
degree_closeness_line3 = parameter_degree_closeness[0] * x_degree_closeness ** 3 + parameter_degree_closeness[1] * x_degree_closeness ** 2 + parameter_degree_closeness[2] * x_degree_closeness + parameter_degree_closeness[3]

# 分割成3个子图并设置整个图幅的大小为3000*1000
fig, ax = plt.subplots(1,3,figsize=(30,10))
# 第一幅子图：介数中心性与度中心性之间的相关性
ax[0].scatter(x_betweenness_degree, y_betweenness_degree)
ax[0].plot(x_betweenness_degree, betweenness_degree_line3, color='g')
ax[0].set_title("介数中心性与度中心性之间的相关性")
print('介数中心性与度中心性拟合的三次曲线参数为：')
p1 = np.poly1d(parameter_betweenness_degree, variable='x')
print(p1)

# 第二幅子图：介数中心性与接近中心性之间的相关性
ax[1].scatter(x_betweenness_closeness, y_betweenness_closeness)
ax[1].plot(x_betweenness_closeness,betweenness_closeness_line3, color='g')
ax[1].set_title("介数中心性与接近中心性之间的相关性")
print('介数中心性与接近中心性拟合的三次曲线参数为：')
p2 = np.poly1d(parameter_betweenness_closeness, variable='x')
print(p2)

# 第三幅子图：度中心性与接近中心性之间的相关性
ax[2].scatter(x_degree_closeness, y_degree_closeness)
ax[2].plot(x_degree_closeness,degree_closeness_line3, color='g')
ax[2].set_title("度中心性与接近中心性之间的相关性")
print('度中心性与接近中心性拟合的三次曲线参数为：')
p3 = np.poly1d(parameter_degree_closeness, variable='x')
print(p3)
plt.savefig('data/processed/output_picture/出租车空间交互网络节点的三种中心性测度相关拟合结果201811')

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("\n程序“9构建出租车OD热点空间交互网络”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“9构建出租车OD热点空间交互网络”的主体代码执行的时间为：%d 秒" % maincode_seconds)