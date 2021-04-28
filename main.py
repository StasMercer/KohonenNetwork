import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button



plt.subplots_adjust(bottom=0.2)
ax = plt.subplot()

ax.set_xlim([0, 10])
ax.set_ylim([0, 10])


def train(cords):
    print(cords)
    print('train printed')

cords = []

def onclick(event):
    # print(event)

    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #   (event.button, event.x, event.y, event.xdata, event.ydata))
    
    
    if event.inaxes == ax:
        cords.append([event.xdata, event.ydata])
        ax.plot(event.xdata, event.ydata, 'or')
        ax.figure.canvas.draw()
        
    if event.inaxes == btn_train_axes:
        train(cords)
    
    if event.inaxes == btn_reset_axes:
        cords.clear()
        ax.cla()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
 
ax.figure.canvas.mpl_connect('button_press_event', onclick)

btn_train_axes= plt.axes([0.7, 0.03, 0.1, 0.075])
btn_reset_axes= plt.axes([0.59, 0.03, 0.1, 0.075])
btn = Button(btn_train_axes, 'Train')
btn = Button(btn_reset_axes, 'Reset')

plt.show()
