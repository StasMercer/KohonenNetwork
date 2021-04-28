import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
cords = np.array([], dtype='float32')


def train(event):
    # print(event)
    print('train printed')

ax = plt.subplot(111)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])



def onclick(event):
    print(event.guiEvent)
    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #   (event.button, event.x, event.y, event.xdata, event.ydata))

    np.append(cords, np.array([event.xdata, event.ydata]))

    ax.plot(event.xdata, event.ydata, 'or')
    plt.show()


btn_axes= plt.axes([0.7, 0.01, 0.1, 0.075])
btn = Button(btn_axes, 'Train')
btn.on_clicked(train)
plt.connect('button_press_event', onclick)
plt.show()
