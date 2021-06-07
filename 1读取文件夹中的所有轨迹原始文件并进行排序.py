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

# 判断是否相交
def inter(a,b):
    return list(set(a)&set(b))

# 抽取轨迹实习材料的原始轨迹文件夹中的txt文件
def extract_track_files(path):
    # 使用glob.glob()获取指定目录(path)下所有的存放轨迹的txt文件，
    files = [filename for filename in glob.glob(path + '*.txt')]
    return files

# 读取轨迹文件中的轨迹点
def get_track_points(filename):
    # 获取存放单个轨迹文件中的所有轨迹点
    # 由于每行中的字段可能并不完整或者存在异常字段（如“计价器故障”），无法直接对齐生成Dataframe
    lines = []
    # 这里先用二进制读入，下面再解码，比文本读入速度更快
    with open(filename, 'rb') as f:
        for fLine in f:
            # decode()将bytes类型的二进制数据转换为str类型
            # 每行最后有一个换行符，需要去掉，这里使用strip()移除字符串头尾指定的字符（默认为空格或换行符）
            fLine = fLine.decode(encoding='mbcs').strip()
            # 使用split()通过指定分隔符（例如‘,’）对字符串进行切片，以达到按照逗号分割字段的目的
            line = [item for item in fLine.split(',')]
            # 将分割的line添加到lines中
            lines.append(line)
    f.close()
    return lines

# 进行数据清理，提取分析需要的字段，并生成Dataframe
def align_points(lines):
    # 原始轨迹数据中包含的异常内容大致如下所示，若发现新的异常值，可进行补充
    anomalies = ['超速报警', '补传', '定位故障', 'LED故障', '摄像头故障', '未定位', '溜车报警', '计价器故障', '紧急报警', '超速', 'ACC关']
    # 清理掉原始数据lines中包含anomalies中异常值的行，并保留其中的前7个字段
    points = [line[:7] for line in lines if not inter(line,anomalies)]
    # 新建一个Dataframe，并将经过清理之后的轨迹点插入其中
    df = pd.DataFrame().append(points)
    # 为字段手动进行命名
    df.columns = ['出租车ID', '定位时间', '经度', '纬度', '方向', '速度', '空车/重车']
    return df

# 将轨迹数据按时间排序
def sort_points(data_track):
    # 使用sort_values()函数，按照'出租车ID','定位时间'这两个字段进行排序，且inplace=True即不创建新的对象，直接对原来的Dataframe进行修改
    data_track.sort_values(by=['出租车ID','定位时间'], inplace=True)
    # 把原来的index去掉，重新生成索引
    data_track = data_track.reset_index(drop=True)

    # 在上下车时刻，数据里会有两条仅 空车/重车 字段不同的记录，形如
    # 100 2018-11-05 00:00:00 -- -- -- -- 空车
    # 100 2018-11-05 00:00:00 -- -- -- -- 重车
    # 经过上述排序后可能造成该字段的顺序出错，需要整理
    old_index = None
    old_row = None
    # 使用iterrows()迭代器对Dataframe进行遍历，返回两个元组：索引列index、行数据row（需要注意此行的每一个字段在row中变成了行，即做了个转置）
    for index, row in data_track.iterrows():
        if old_index == None:
            old_index = index
            old_row = row
            continue
        # 判断是否出现误排，即前六个字段全部相同，只有'空车/重车'字段不相同
        if (row['出租车ID']==old_row['出租车ID'] and
            row['定位时间']==old_row['定位时间'] and
            row['经度']==old_row['经度'] and
            row['纬度']==old_row['纬度'] and
            row['方向']==old_row['方向'] and
            row['速度']==old_row['速度'] and
            row['空车/重车']!=old_row['空车/重车']):

            # 与更前一行的状态进行比较
            older_row=data_track.iloc[old_index-1]
            # 如果这一行与更前一行的数据相比，'出租车ID'和'空车/重车'都相同的话，即出现了误排
            # 现实中的解释可以为：在连续的时间内状态应该是连续的，而不会出现在同一时刻“空车-重车-空车”（乘客瞬间上车又下车）的情况
            # 而应该是“空车-空车-重车”（乘客在这一时段上车），因此需要交换两行的索引值
            if older_row['出租车ID']==row['出租车ID'] and older_row['空车/重车']==row['空车/重车']:
                # 交换这一行与前一行 两行的索引值，这并不会影响iterrows迭代器
                temp = data_track.iloc[index]
                data_track.iloc[index] = data_track.iloc[old_index]
                data_track.iloc[old_index] = temp
        old_index = index
        old_row = row
    return data_track

"""***代码正文部分***"""
# 记录开始执行代码的时间
time_maincode_start = time.localtime()
print("程序“1读取文件夹中的所有轨迹原始文件并进行排序”的主体代码开始执行的时间为：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start))

# 完成轨迹点的文件读取以及数据清理、字段分割
# data_folder_path为原始轨迹数据文件（txt）所在的目录相对路径
data_folder_path='data/raw/'
# 抽取原始轨迹文件夹中的txt文件
files = extract_track_files(data_folder_path)

# 新建一个Dataframe，命名为data_track，用来存放轨迹数据
data_track = pd.DataFrame()
# 使用循环读取所有的原始轨迹数据文件
for i in range(len(files)):
    track_points = get_track_points(files[i])
    data_track = data_track.append(align_points(track_points))
    print("读取了文件列表中的第%d个文件，当前的轨迹数据有%d条" % ((i+1),len(data_track)))

# 如果只想使用其中的某一个文件的话释放下列代码
# track_points = get_track_points(files[1])
# 进行数据清理，提取分析需要的字段
# data_track = align_points(track_points)

# 将轨迹数据按时间排序
data_track = sort_points(data_track)
# 保存经过清洗、排序等预处理后的轨迹数据
data_track.to_csv('data/processed/output_data/经过预处理后的轨迹数据201811.csv', index=False)

# 记录结束执行代码的时间
time_maincode_end = time.localtime()
print("程序“1读取文件夹中的所有轨迹原始文件并进行排序”的主体代码结束执行的时间为：%s" % time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end))
time_maincode_start = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_start),"%Y-%m-%d %H:%M:%S")
time_maincode_end = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time_maincode_end),"%Y-%m-%d %H:%M:%S")
# 结束时间减去开始时间得到运行的时间
maincode_seconds = (time_maincode_end - time_maincode_start).seconds
print("程序“1读取文件夹中的所有轨迹原始文件并进行排序”的主体代码执行的时间为：%d 秒" % maincode_seconds)