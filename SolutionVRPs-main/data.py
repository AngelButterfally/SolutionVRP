import pandas as pd
import numpy as np


def read_data(path):
    data = pd.read_csv(path, header=None)
    if len(data.columns) == 2:
        data.columns = list('XY')
    if len(data.columns) == 3:
        data.columns = list('XYQ')

    def my_point(a, b):
        return (a, b)
    data['point'] = data.apply(
        lambda row: my_point(row['X'], row['Y']), axis=1)
    locations = list(data['point'])
    lenth = len(data['point'])
    return locations, lenth


def random_generate_data(path, num=10):
    ini_data = read_data(path)
    new_list = []
    if num >= len(ini_data[0]):
        num = len(ini_data[0])-1
    if num <= 1:
        num = 1
    rand_p = np.random.choice(len(ini_data[0]), num, replace=False)
    for i in rand_p:
        new_list.append(ini_data[0][i])
    if (691, 1233) in new_list:
        new_list.remove((691, 1233))
    new_list.insert(0, (691, 1233))
    return new_list


def mannual_generate_data(path, mannual_list=[1, 2]):
    ini_data = read_data(path)
    new_list = []
    temp_list = []
    print(mannual_list)
    for i in mannual_list:
        if i >= len(ini_data[0]):
            i = len(ini_data[0])-1  # 从零开始计数所以减一
        if i <= 0:
            i = 0
        temp_list.append(i)
    temp_list = list(set(temp_list))  # 去除重复项
    print(temp_list)
    for j in temp_list:
        new_list.append(ini_data[0][j])
    if (691, 1233) in new_list:
        new_list.remove((691, 1233))
    new_list.insert(0, (691, 1233))
    print(new_list)
    return new_list


def create_data_VRP(path, vehicles_num, auto_point_num=99, auto_random_select_point=False, mannual_point_list=[1, 2]):
    """Stores the data for the problem."""
    data = {}
    if auto_random_select_point == True:
        data['locations'] = random_generate_data(path, num=auto_point_num)
    if auto_random_select_point == False:
        data['locations'] = mannual_generate_data(
            path, mannual_list=mannual_point_list)
    # data['locations'] = [(691, 1233),  (1672, 840), (1733, 927), (1369, 1824), (1850, 1119), (2089, 998), (1937, 1187), (2176, 1082), (2406, 1373), (2021, 1862), (2262, 1740), (2436, 1873), (2426, 1631), (2570, 1093), (255, 804), (518, 695), (521, 770), (688, 661), (691, 1717), (518, 1554), (518, 1715),  (961, 795), (951, 1233), (951, 1615), (875, 1824), (1042, 736), (1122, 927), (1098, 418), (1174, 608), (1167, 827), (1156, 1119), (1252, 1596), (1117, 1824), (1319, 451), (1319, 608), (1536, 840), (1627, 608), (1626, 732), (573, 1824), (824, 648), (743, 774), (956, 612), ]
    data['num_vehicles'] = vehicles_num
    data['depot'] = 0
    data['receive_point_num'] = auto_point_num
    return data
