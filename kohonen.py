import math


class Kohonen:
    def __init__(self, start_radius, start_learning_rate, time_constant):
        self.start_radius = start_radius
        self.start_learning_rate = start_learning_rate
        self.time_constant = time_constant

    def get_radius(self, iteration_number):
        return self.start_radius * math.exp(-iteration_number / self.time_constant)

    def get_learning_rate(self, iteration_number):
        return self.start_learning_rate * math.exp(-iteration_number / self.time_constant)
        
    def neibourghood_function(self, iteration_number):
        return math.exp(-self.distance ** 2 / (2 * self.get_radius(iteration_number)))
