from typing import Tuple
import math


class RunningStats:
    """ Welford's algorithm for running mean/std """
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.epsilon = 1e-4
    def clear(self):
        self.n = 0
    
    def push(self, x):
        self.n += 1
    
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
        
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
    
    def standard_deviation(self):
        # print(math.sqrt(self.variance()) + self.epsilon)
        return math.sqrt(self.variance()) + self.epsilon
    
class RunningAverage:
    def __init__(self):
        self.counts = {}
        self.averages = {}

    def add(self, var_name):
        if var_name not in self.averages:
            self.averages[var_name] = 0.0
            self.counts[var_name] = 0

    def update(self, var_name, value):
        if var_name not in self.averages:
            print(f"Variable {var_name} is not being tracked. Use add method to track.")
            return
        self.counts[var_name] += 1
        self.averages[var_name] += (value - self.averages[var_name]) / self.counts[var_name]

    def get_average(self, var_name):
        return self.averages.get(var_name, None)

    def reset(self, var_name=None):
        if var_name is None:
            self.counts = {}
            self.averages = {}
        elif var_name in self.averages:
            self.counts[var_name] = 0
            self.averages[var_name] = 0.0
        else:
            print(f"Variable {var_name} is not being tracked.")