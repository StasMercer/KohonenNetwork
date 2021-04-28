import matplotlib.pyplot as plt
import math
import numpy as np
import random
from matplotlib.widgets import Button

TIME = 1
WIDTH = 1
HEIGHT = 1

class Node:
    def __init__(self, cords):
        self.cords = np.array(cords)
        # self.weights = get_random_weights()

    def get_rgb(self):
        return (self.weights[0], self.weights[1], self.weights[2])

    def update_weights(self, a, b, input_vector):
        ax.plot(self.cords[0], self.cords[1], 'o', color='white')
        ax.figure.canvas.draw()
        for i in range(2):
        # for i in range(3):
            self.cords[i] += a * b * (input_vector[i] - self.cords[i])
            # self.weights[i] += a * b * (input_vector[i] - self.weights[i])
            # if self.weights[i] > 1:
            #     self.weights[i] = 1
            # elif self.weights[i] < 0:
            #     self.weights[i] = 0
            if self.cords[i] > 1:
                self.cords[i] = 1
            elif self.cords[i] < 0:
                self.cords[i] = 0

        ax.plot(self.cords[0], self.cords[1], 'ok')
        # ax.plot(self.cords[0], self.cords[1], 'o', color=self.get_rgb())
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
        return math.exp(-1 * (distance ** 2) / (2 * self.get_radius(iteration_number)))
        
    def train(self, nodes):
        print(nodes)
        print('train printed')

        # n_input_vectors = 3
        # input_vectors = np.array([get_random_weights() for _ in range(n_input_vectors)])
        input_vectors = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])

        for iteration_number in range(TIME):
            for input_vector in input_vectors:
                best_matching_unit = self.get_best_matching_unit(input_vector, nodes)

                learning_rate = self.get_learning_rate(iteration_number)
                
                for node in nodes:
                    # if node != best_matching_unit: # ?? is it in need
                    distance = np.linalg.norm(best_matching_unit.cords - node.cords)
                    if distance != 0:
                        neibourghood = self.neibourghood_function(iteration_number, distance)
                        node.update_weights(learning_rate, neibourghood, input_vector)

    def get_best_matching_unit(self, input_vector, nodes):
        min_distance = math.sqrt(2) # sqrt((0 - 1) ** 2 + (0 - 1) ** 2 + (0 - 1) ** 2)
        # min_distance = math.sqrt(3) # sqrt((0 - 1) ** 2 + (0 - 1) ** 2 + (0 - 1) ** 2)
        best_matching_unit = None

        for node in nodes:
            euclid_distance = np.linalg.norm(input_vector - node.cords)
            # euclid_distance = np.linalg.norm(input_vector - node.weights)
            if euclid_distance < min_distance:
                min_distance = euclid_distance
                best_matching_unit = node

        return best_matching_unit


# def get_random_weights():
#     return np.array([random.uniform(0, 1), random.uniform(0, 1)])
#     # return np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])


def onclick(event):
    # print(event)

    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #   (event.button, event.x, event.y, event.xdata, event.ydata))
    
    if event.inaxes == ax:
        node = Node((event.xdata, event.ydata))
        nodes.append(node)
        ax.plot(event.xdata, event.ydata, 'ok')
        # ax.plot(event.xdata, event.ydata, 'o', color=node.get_rgb())
        ax.figure.canvas.draw()
        
    if event.inaxes == btn_train_axes:
        start_radius = max(HEIGHT, WIDTH) / 16
        time_constant = TIME / math.log(start_radius)
        model = Kohonen(start_radius, 0.8, time_constant) # set parameters not in this function
        model.train(nodes)
    
    if event.inaxes == btn_reset_axes:
        nodes.clear()
        ax.cla()
        ax.set_xlim([0, WIDTH])
        ax.set_ylim([0, HEIGHT])


if __name__ == '__main__':
    plt.subplots_adjust(bottom=0.2)
    ax = plt.subplot()

    ax.set_xlim([0, WIDTH])
    ax.set_ylim([0, HEIGHT])

    nodes = []

    ax.figure.canvas.mpl_connect('button_press_event', onclick)

    btn_train_axes= plt.axes([0.7, 0.03, 0.1, 0.075])
    btn_reset_axes= plt.axes([0.59, 0.03, 0.1, 0.075])
    btn = Button(btn_train_axes, 'Train')
    btn = Button(btn_reset_axes, 'Reset')

    plt.show()
