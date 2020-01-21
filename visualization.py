import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import load_test
import os, sys


def create_video(name, folder, fps=12):
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

def plot_day(X, Y, save_dir):



if __name__ == "__main__":
    usage = "python3 visualization.py [GHI|GFS] [latbounds] [lonbounds] [timestart] [timestop]\n" \
            "example: \n" \
            "$ python3 visualization.py GHI 18, -160,-155 2013/02/28  2013/03/14\n"
    if len(sys.argv) != 6:
        print(usage)
        exit(1)
    dataset = sys.argv[1]
    latbounds = read_latlong_args(sys.argv[2])
    lonbounds = read_latlong_args(sys.argv[3])
    timestart = get_datetime(sys.argv[4])
    timestop = get_datetime(sys.argv[5])

    catalog = pd.read_csv("catalog.csv", parse_dates=['time_stamp'])
    mask = (catalog['time_stamp'] > timestart) & (catalog['time_stamp'] < timestop)

    files = catalog.loc[mask][dataset].dropna().tolist()
    files = sorted(files)
    folder = "../visuals/"

    plot_catalog_imgs(files, folder, dataset, latbounds, lonbounds)
    create_video("vid", folder)
