# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib import axes
import io

def draw_heatmap(file,data, xlabels, ylabels):
    # cmap=cm.Blues
    # cmap = cm.get_cmap('rainbow_r', 1000)
    # style='binary'
    style = 'ocean_r'
    # style = 'gunplot2'
    cmap = cm.get_cmap(style, 20000)
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    vmax = data[0][0]
    vmin = data[0][0]
    for i in data:
        for j in i:
            if j > vmax:
                vmax = j
            if j < vmin:
                vmin = j
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
    name=file+style+'.jpg'
    plt.savefig('d:\\z-data\\'+name)
    plt.show()

GRID_COUNT=100
global_local_trans_matrix = np.zeros( (GRID_COUNT,GRID_COUNT) )

# a = np.random.rand(10, 10)
print(global_local_trans_matrix)
# xlabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# ylabels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
xlabels = range(0,GRID_COUNT,10)
print(xlabels)
ylabels = range(0,GRID_COUNT,10)
print(ylabels)

hour_span=2
for i in range(int(24/hour_span)):
    file = "t"+str(i)+"log"
    log_out = io.open("d:\\z-data\\" + file, encoding='utf-8', mode='r')
    for i in range(GRID_COUNT):
        line = log_out.readline()
        arr = line.split(" ")
        for j in range(global_local_trans_matrix.shape[0]):
            print("i:%d,j:%d,len:%s" % (i, j, len(arr)))
            global_local_trans_matrix[i][j] = arr[j + 1]
    draw_heatmap(file,global_local_trans_matrix, xlabels, ylabels)