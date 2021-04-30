import matplotlib.pyplot as plt
import math
import numpy as np
import random
from matplotlib.widgets import Button
from numpy.lib.function_base import append

TIME = 10
WIDTH = 1
HEIGHT = 1


class Kohonen:
    def __init__(self, start_radius, start_learning_rate, time_constant):
        self.start_radius = start_radius
        self.start_learning_rate = start_learning_rate
        self.time_constant = time_constant
        self.nx, self.ny = (10, 10)
        self.weights = np.meshgrid(np.linspace(0.25, 0.75, self.nx), np.linspace(0.25, 0.75, self.ny))

        print(self.weights)

        # draw kohonen layer nodes
        plt.plot(self.weights[0], self.weights[1], 'ok')

        # draw horizontal connections
        for j in range(self.ny):
            for i in range(self.nx - 1):
                plt.plot([self.weights[0][j][i], self.weights[0][j][i + 1]], [self.weights[1][j][i], self.weights[1][j][i + 1]], '-k')
        
        # draw vertical connections
        for j in range(self.nx):
            for i in range(self.ny - 1):
                plt.plot([self.weights[0][i][j], self.weights[0][i + 1][j]], [self.weights[1][i][j], self.weights[1][i + 1][j]], '-k')

    def reset_weights(self):
        self.weights = np.meshgrid(np.linspace(0.25, 0.75, self.nx), np.linspace(0.25, 0.75, self.ny))

    # calculate current 
    def get_learning_rate(self, iteration_number):
        return self.start_learning_rate * math.exp(-1 * iteration_number / self.time_constant)
    
    # calculate current radius for neibourghood_function
    def get_radius(self, iteration_number):
        return self.start_radius * math.exp(-1 * iteration_number / self.time_constant)
    
    # calculate value for neighbourhood function
    def neibourghood_function(self, iteration_number, distance):
        return math.exp(-1 * (distance ** 2) / (2 * self.get_radius(iteration_number)))
        
    def train(self, input_vectors):
        print(input_vectors[0])
        print('train printed')

        # for every iteration in defined TIME
        for iteration_number in range(TIME):

            # for every input vector in input data
            for point in input_vectors:

                # find the best matching node
                best_matching_unit = self.get_best_matching_unit(point)

                # calculate learning rate for current iteration
                learning_rate = self.get_learning_rate(iteration_number)

                # for every node in Kohonen layer
                for i in range(self.nx):
                    for j in range(self.ny):
                        x, y = self.weights[0][i, j], self.weights[1][i, j]

                        # calculate euclid distance between current node and best matching node
                        distance = np.linalg.norm(best_matching_unit - np.array([x, y]))

                        # calculate neibourghood value for current node and best matching node
                        neibourghood = self.neibourghood_function(iteration_number, distance)

                        # update weights
                        self.weights[0][i, j], self.weights[1][i, j] = self.update_weight([x, y], learning_rate, neibourghood, point)
            
            # clear screen
            ax.cla()
            
            # draw input data
            for x, y in input_vectors:
                ax.plot(x, y, 'or')
            
            # draw Kohonen layer
            ax.plot(self.weights[0], self.weights[1], 'ok')
            for j in range(self.ny):
                for i in range(self.nx - 1):
                    ax.plot([self.weights[0][j][i], self.weights[0][j][i + 1]], [self.weights[1][j][i], self.weights[1][j][i + 1]], '-k')
            for j in range(self.nx):
                for i in range(self.ny - 1):
                    ax.plot([self.weights[0][i][j], self.weights[0][i + 1][j]], [self.weights[1][i][j], self.weights[1][i + 1][j]], '-k')
            ax.set_xlim([0, WIDTH])
            ax.set_ylim([0, HEIGHT])
            ax.figure.canvas.draw()

    def get_best_matching_unit(self, point):
        min_distance = math.sqrt(len(point))
        best_matching_unit = None

        for vec_x, vec_y in zip(self.weights[0], self.weights[1]):
            for x, y in zip(vec_x, vec_y):
                
                euclid_distance = np.linalg.norm((np.array(point) - np.array([x, y])))
                
                if euclid_distance < min_distance:
                    min_distance = euclid_distance
                    best_matching_unit = [x,y]
        
        return np.array(best_matching_unit)

    def update_weight(self, cords, a, b, point):
        # for x axes
        cords[0] += a * b * (point[0] - cords[0])
        if cords[0] > WIDTH:
            cords[0] = WIDTH
        elif cords[0] < 0:
            cords[0] = 0
        
        # for y axes
        cords[1] += a * b * (point[1] - cords[1])
        if cords[1] > HEIGHT:
            cords[1] = HEIGHT
        elif cords[1] < 0:
            cords[1] = 0
        
        return cords[0], cords[1]


def onclick(event):
    # print(event)

    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #   (event.button, event.x, event.y, event.xdata, event.ydata))
    
    if event.inaxes == ax:
        
        input_vector.append([event.xdata, event.ydata])
        ax.plot(event.xdata, event.ydata, 'or')
        ax.figure.canvas.draw()
        
    if event.inaxes == btn_train_axes:
         # set parameters not in this function
        model.train(input_vector)
    
    if event.inaxes == btn_reset_axes:
        input_vector.clear()
        model.reset_weights()
        ax.cla()
        ax.plot(model.weights[0], model.weights[1], 'ok')
        for j in range(model.ny):
            for i in range(model.nx - 1):
                ax.plot([model.weights[0][j][i], model.weights[0][j][i + 1]], [model.weights[1][j][i], model.weights[1][j][i + 1]], '-k')
        for j in range(model.nx):
            for i in range(model.ny - 1):
                ax.plot([model.weights[0][i][j], model.weights[0][i + 1][j]], [model.weights[1][i][j], model.weights[1][i + 1][j]], '-k')
        ax.set_xlim([0, WIDTH])
        ax.set_ylim([0, HEIGHT])


if __name__ == '__main__':
    plt.subplots_adjust(bottom=0.2)
    ax = plt.subplot()
    a = [1, 2, 3]
    for ai in a:
        a[0] = 4
    print(a)
    nodes = []
    ax.set_xlim([0, WIDTH])
    ax.set_ylim([0, HEIGHT])
    
    input_vector = []
    start_radius = max(HEIGHT, WIDTH) / 32
    time_constant = TIME / math.log(start_radius) * 32
    model = Kohonen(start_radius, 0.1, time_constant)

    ax.figure.canvas.mpl_connect('button_press_event', onclick)

    btn_train_axes= plt.axes([0.7, 0.03, 0.1, 0.075])
    btn_reset_axes= plt.axes([0.59, 0.03, 0.1, 0.075])
    btn = Button(btn_train_axes, 'Train')
    btn = Button(btn_reset_axes, 'Reset')

    plt.show()
