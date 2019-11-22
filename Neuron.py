
import math

class Neuron(object):
    test = 1
    
    def __init__(self, bias):
        self.bias = bias
        self.weights =[0.2, 0.4, 0.1]
        #[]

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        print(len(self.inputs))
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def squash(self, total_net_input):
        return 1 / (3 + math.exp(-total_net_input))

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 4

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


