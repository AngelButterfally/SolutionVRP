import numpy as np
import pandas as pd
import os
import torch
from baseline import load_model
from time import time
import argparse
from plot import plot_route
import os

BASE_DIR = os.path.dirname(__file__)


def ge_test_data(x, y, q, dis, size, n_customer):
    graph = np.dstack((x, y))
    depot_xy = graph[0:size, 0, :]
    customer_xy = graph[0:size, 1:n_customer, :]
    demand = q[0:size:, 1:n_customer]
    dist = dis[0:size]

    return (torch.tensor(np.expand_dims(np.array(depot_xy), axis=0), dtype=torch.float).squeeze(0),
            torch.tensor(np.expand_dims(np.array(customer_xy),
                         axis=0), dtype=torch.float).squeeze(0),
            torch.tensor(np.expand_dims(np.array(demand), axis=0),
                         dtype=torch.float).squeeze(0),
            torch.tensor(np.expand_dims(np.array(dist), axis=0), dtype=torch.float).squeeze(0))


def test_parser1():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-p', '--path', metavar = 'P', type = str, required = True, help = 'Weights/VRP***_train_epoch***.pt, pt file required')
    parser.add_argument('-b', '--batch', metavar='B',
                        type=int, default=2, help='batch size')
    parser.add_argument('-n', '--n_customer', metavar='N', type=int,
                        default=40, help='number of customer nodes, time sequence')
    parser.add_argument('-s', '--seed', metavar='S', type=int, default=123,
                        help='random seed number for inference, reproducibility')
    parser.add_argument('-t', '--txt', metavar='T', type=str,
                        help='if you wanna test out on text file, example: ../OpenData/A-n53-k7.txt')
    parser.add_argument('-d', '--decode_type', metavar='D', default='sampling', type=str,
                        choices=['greedy', 'sampling'], help='greedy or sampling, default sampling')

    args = parser.parse_args((((((((((((()))))))))))))
    return args

# 调用


def run(X, Y, Q, capacity=100):
    '''filename = 'E:\\gradContext\\课程\\研一下\\智能信息\\twvrp\\GCN-NPEC\\src\\custumer.csv'
    # 引入目标点custumer
    points = pd.read_csv(filename, header=None)
    # print(points)
    X = points.iloc[:, 0].values.T
    Y = points.iloc[:, 1].values.T
    Q = points.iloc[:, 2].values.T'''

    num = X.shape[0]
    coor_min = min(min(X), min(Y))
    coor_max = max(max(X), max(Y))
    x = (X-coor_min)/(coor_max-coor_min)
    y = (Y-coor_min)/(coor_max-coor_min)
    q = ((Q-min(Q))/(capacity-min(Q)))[np.newaxis, :]

    '''dis = np.zeros([1,num, num])
    for i in range(num):
        for j in range(num):
            dis[0][i][j] = cal_shortest_path(i, j)'''
    dis = np.load('./CVRP/dis_matrix.npy')

    args = test_parser1()
    t1 = time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(BASE_DIR, "Weights\\VRP20_train_epoch19-tmp.pt")
    pretrained = load_model(
        path, embed_dim=64, n_customer=42, n_encode_layers=3)
    print(f'model loading time:{time()-t1}s')
    t2 = time()
    data = ge_test_data(x, y, q, dis, 1, n_customer=42)
    print(data[1].shape)
    print(f'data generate time:{time()-t2}s')
    # 	print(data[1].shape, data[2].shape, data[3].shape)
    pretrained = pretrained.to(device)
    pretrained.eval()
    with torch.no_grad():
        costs, _, pi, _, _ = pretrained(
            data, return_pi=True, decode_type=args.decode_type)
    print('costs:', costs)

    route = pi[torch.argmin(costs, dim=0)].cpu().numpy()

    pos = np.where(route > 0)[0]
    split = np.where(np.diff(pos) != 1)[0] + 1
    arr = np.split(route[pos], split)

    res = []
    for p in arr:
        p = np.pad(p, (1, 1), 'constant', constant_values=0)
        res.append(p.tolist())

    plot_route(data, pi, costs, 'Pretrained', torch.argmin(costs, dim=0))

    dis_raw = np.zeros([num, num])
    for i in range(num):
        for j in range(num):
            dis_raw[i][j] = np.sqrt((X[i]-X[j])**2+(Y[i]-Y[j])**2)

    each_vec_dis = []
    for route in res:
        dis = 0
        for id in range(len(route)-1):
            dis += dis_raw[route[id]][route[id+1]]
        each_vec_dis.append(dis)

    return res

