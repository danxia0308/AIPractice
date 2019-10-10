import colorsys
import random
import numpy as np
import matplotlib.pyplot as plt

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

colors=random_colors(10, bright=False)
img=np.zeros(shape=(256,256,3))
for c in range(3):
    img[:,:,c]=colors[1][c]
plt.imshow(img)
plt.show()