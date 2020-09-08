from scipy import misc
# import matplotlib.pyplot as plt
import numpy as np
import os

# a=[1,2,5,3]
# b=a.sort()
# a.reverse()
a=os.environ.get('CPU_NUM')
print(a)


def add(x):
    return x+1

print(add(1))