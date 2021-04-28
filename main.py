import matplotlib.pyplot as plt
import math
import numpy as np
import random
from matplotlib.widgets import Button


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

    def get_rgb(self):
        return (self.color[0], self.color[1], self.color[2])

    def get_weights(self):
        return self.x, self.y, self.color[0], self.color[1], self.color[2]

    def update_weights(self, a, b, input):
        ax.plot(self.x, self.y, 'o', color=(1, 1, 1))
        ax.figure.canvas.draw()
        self.x += a * b * (input[0] - self.get_weights()[0])
        self.y += a * b * (input[0] - self.get_weights()[1])
        self.color[0] += a * b * (input[2] - self.get_weights()[2])
        self.color[1] += a * b * (input[3] - self.get_weights()[3])
        self.color[2] += a * b * (input[4] - self.get_weights()[4])
        ax.plot(self.x, self.y, 'o', color=self.get_rgb())
        ax.figure.canvas.draw()


class Kohonen:
    def __init__(self, start_radius, start_learning_rate, time_constant):
        self.start_radius = start_radius
        self.start_learning_rate = start_learning_rate
        self.time_constant = time_constant

    def get_radius(self, iteration_number):
        return self.start_radius * math.exp(-iteration_number / self.time_constant)

    def get_learning_rate(self, iteration_number):
        return self.start_learning_rate * math.exp(-iteration_number / self.time_constant)
        
    def neibourghood_function(self, iteration_number, distance):
        return math.exp(-distance ** 2 / (2 * self.get_radius(iteration_number)))
        
    def train(self, nodes):
        print(nodes)
        print('train printed')
        for _ in range(10):
            for iteration_number in range(3):
                input_vector = np.array([random.uniform(0, 1) for _ in range(5)])

                best_matching_unit, min_distance = self.get_best_matching_unit(input_vector, nodes)

                a = self.get_learning_rate(iteration_number)
                b = self.neibourghood_function(iteration_number, min_distance)

                best_matching_unit.update_weights(a, b, input_vector)

    def get_best_matching_unit(self, input_vector, nodes):
        min_distance = math.sqrt(5) # n features
        best_matching_unit = None

        for node in nodes:
            euclid_distance = np.linalg.norm(input_vector - node.get_weights())
            if euclid_distance < min_distance:
                min_distance = euclid_distance
                best_matching_unit = node

        return best_matching_unit, min_distance


def onclick(event):
    # print(event)

    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #   (event.button, event.x, event.y, event.xdata, event.ydata))
    
    if event.inaxes == ax:
        node = Node(event.xdata, event.ydata)
        nodes.append(node)
        ax.plot(event.xdata, event.ydata, 'o', color=node.get_rgb())
        ax.figure.canvas.draw()
        
    if event.inaxes == btn_train_axes:
        model = Kohonen(10, 1, 1)
        model.train(nodes)
    
    if event.inaxes == btn_reset_axes:
        nodes.clear()
        ax.cla()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])


if __name__ == '__main__':
    plt.subplots_adjust(bottom=0.2)
    ax = plt.subplot()

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    nodes = []

    ax.figure.canvas.mpl_connect('button_press_event', onclick)

    btn_train_axes= plt.axes([0.7, 0.03, 0.1, 0.075])
    btn_reset_axes= plt.axes([0.59, 0.03, 0.1, 0.075])
    btn = Button(btn_train_axes, 'Train')
    btn = Button(btn_reset_axes, 'Reset')

    plt.show()
