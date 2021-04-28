import math


class Kohonen:
    def __init__(start_radius, start_learning_rate, time_constant):
        self.start_radius = start_radius
        self.start_learning_rate = start_learning_rate
        self.time_constant = time_constant

    def get_radius(iteration_number):
        return self.start_radius * math.exp(-iteration_number / self.time_constant)

    def get_learning_rate(iteration_number):
        return self.start_learning_rate * math.exp(-iteration_number / self.time_constant)
        
    def neibourghood_function(iteration_number):
        return math.exp(-self.distance ** 2 / (2 * get_radius(iteration_number)))
