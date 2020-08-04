import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

def set_figscale(fig, ax):
    x0, y0, dx, dy = ax.get_position().bounds
    w = 3 * max(dx, dy) /dx
    h = 3 * max(dx, dy) /dy
    fig.set_size_inches((w, h))

def draw_circle(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    x, y = np.cos(t), np.sin(t)
    ax.plot(x*1.0, y*1.0, 'k')
    axis = 1.01
    ax.axis([-axis, axis, -axis, axis])

def plot_light(x, y, save_name, c=None):
    fig, ax = plt.subplots()

    if c is None:
        ax.scatter(x, y, s=6) 
    else:
        #c = np.linspace(0,1, 96)
        plt.scatter(x, y, c=c, cmap='jet', vmin=0, vmax=1)
    draw_circle(ax)

    ax.axis('off') 
    set_figscale(fig, ax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(save_name, bbox_inches=extent)
    plt.close()
