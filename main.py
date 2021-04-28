import matplotlib.pyplot as plt
import math
import numpy as np
import random
from matplotlib.widgets import Button


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.weights = get_random_color()

    def get_color(self):
        return (self.weights[0], self.weights[1], self.weights[2])

def get_random_color():
    return np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])


def train(cords):
    print(cords)
    print('train printed')
    
    for i in range(10): # n attempts
        input_vector = get_random_color()

        min_distance = math.sqrt(3) # sqrt((0 - 1) ** 2 + (0 - 1) ** 2 + (0 - 1) ** 2)
        best_matching_unit = None

        for node in cords:
            euclid_distance = np.linalg.norm(input_vector - node.weights)
            if euclid_distance < min_distance:
                min_distance = euclid_distance
                best_matching_unit = node

        print("")


def onclick(event):
    # print(event)

    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #   (event.button, event.x, event.y, event.xdata, event.ydata))
    
    if event.inaxes == ax:
        node = Node(event.xdata, event.ydata)
        cords.append(node)
        ax.plot(event.xdata, event.ydata, 'o', color=node.get_color())
        ax.figure.canvas.draw()
        
    if event.inaxes == btn_train_axes:
        train(cords)
    
    if event.inaxes == btn_reset_axes:
        cords.clear()
        ax.cla()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])


if __name__ == '__main__':
    plt.subplots_adjust(bottom=0.2)
    ax = plt.subplot()

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    cords = []

    ax.figure.canvas.mpl_connect('button_press_event', onclick)

    btn_train_axes= plt.axes([0.7, 0.03, 0.1, 0.075])
    btn_reset_axes= plt.axes([0.59, 0.03, 0.1, 0.075])
    btn = Button(btn_train_axes, 'Train')
    btn = Button(btn_reset_axes, 'Reset')

    plt.show()
