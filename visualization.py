import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import load_model
from utils import load_test
import os
import numpy as np

def create_video(name, folder, fps=1):
    os.system("ffmpeg -pattern_type glob -i \"{}*.png\" -r {} -vcodec mpeg4 -y {}{}.mp4".format(folder, fps, folder, name))


def plot_slice(x, y, save_loc):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(x, cmap='hot')
    ax1.title.set_text('Ground Truth')

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(y, cmap='hot')
    ax2.title.set_text('Prediction')
    plt.savefig(save_loc)
    plt.close(fig)


def plot_day(X, Y, save_dir):
    X = np.squeeze(X)
    Y = np.squeeze(Y)
    for idx, (x, y) in enumerate(zip(X, Y)):

        filename = os.path.join(save_dir, "step_{}.png".format(idx))
        plot_slice(x, y, save_loc=filename)


if __name__ == "__main__":

    X, Y = load_test()
    model = load_model("model_01.h5")

    os.mkdir("imgs")
    for idx, (x, y) in enumerate(zip(X, Y)):
        y_pred = model.predict(x)
        dirname = "imgs/day_{:03d}/".format(idx)
        os.mkdir(dirname)
        plot_day(y, y_pred, save_dir=dirname)

