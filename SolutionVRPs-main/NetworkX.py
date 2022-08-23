import pandas as pd
import matplotlib.pyplot as plt 
import networkx as nx  

def import_distance_information(road_data_path,point_data_path):
    #road
    road_data = pd.read_excel(road_data_path,header=None)
    road_data.fillna(0,inplace=True)
    #point
    point_data = pd.read_excel(point_data_path,header=None)
    point_data.columns = list('IXY')
    point_data.head(10)
    def my_point(a,b):
        return (a,2124-b)
    point_data['point'] = point_data.apply(lambda row: my_point(row['X'], row['Y']), axis=1)
    return [road_data, point_data]


def cal_shortest_path(from_point=0,to_point=100):
    road_data,point_data = import_distance_information('./road.xlsx', './point2.xlsx')
    #point = list(point_data['point'])  
    road  = pd.DataFrame(road_data.values)
    #建图
    G = nx.Graph() 
    G = nx.from_pandas_adjacency(road) 
    #a*算法
    if from_point < 0:
        from_point = 0
    if to_point > len(point_data['point'])-1 :
        to_point = len(point_data['point'])-1 
    min_path = nx.astar_path(G, source=from_point, target=to_point)
    min_distance = nx.astar_path_length(G, source=from_point, target=to_point)  #最短加权路径长度
    return [min_path,min_distance] 

def show_networkX_graph(figsize = (30,20)):
    road_data,point_data = import_distance_information('./road.xlsx','./point2.xlsx')
    point = list(point_data['point'])  
    road  = pd.DataFrame(road_data.values)
    #建图
    G = nx.Graph() 
    G = nx.from_pandas_adjacency(road) 
    plt.figure('1',figsize=figsize)
    # 把节点画出来
    nx.draw_networkx_nodes(G,point,node_color='b',node_size=50,alpha=0.8)
    # 把边画出来
    nx.draw_networkx_edges(G,point,width=1.0,alpha=0.5,edge_color='b')
    #把边权重画出来
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, point, edge_labels=labels, font_color='b')  
    plt.axis('on')
    plt.show()

if __name__ == '__main__':
    a = cal_shortest_path(0, 6)
    #show_networkX_graph()
    print(a[1])
