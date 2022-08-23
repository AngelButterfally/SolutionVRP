import numpy as np
import matplotlib.pyplot as plt
import torch
import math

size = 10000 
n_customer = 20 + 1

def read(filename):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(',')
            dataset.append(line)
    # print(dataset[32])
    return dataset
    

def show(dataset, num_of_epoch):
    cost = []
    epoch = []
    print(float(dataset[1][4]))
    for i in range(num_of_epoch):
        # index = i * 32 + 1
        index = (i + 1) * 32
        cost.append(float(dataset[index][4]))
        epoch.append(i + 1)
    cost = np.array(cost)
    epoch = np.array(epoch)
    plt.figure(1)
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.title("cost-epoch")
    plt.plot(epoch, cost)
    plt.savefig('cost-epoch.jpg', dpi = 300)
    plt.show()



# if __name__=="__main__":
#     filename = "./result.csv"
#     dataset = read(filename)
#     show(dataset, 300)



def get_dist(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError


if __name__=="__main__":
    delta = 0.1
    graph = np.random.rand(size, n_customer, 2)
    dist = np.zeros((size, n_customer, n_customer))
    for i in range(size):
        for j in range(n_customer):
            for k in range(n_customer):
                dist[i][j][k] = get_dist(graph[i][j], graph[i][k]) #+ 0.1 * np.random.randn(1)
    demand = np.random.rand(size, n_customer-1)
    print(demand)
    depot_demand = np.zeros((size,1))
    demand = np.concatenate((depot_demand, demand), axis = 1)
    np.savez('my-41-training.npz', graph = graph, demand = demand, dis = dist)

    print(graph.shape)
    print(demand.shape)
    print(dist.shape)