import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from extract_utils import *
import os, sys


def plot_img(data, file):
    plt.ioff()

    plt.savefig(file)
    plt.close()


def plot_catalog_imgs(files, folder, dataset, latbounds, lonbounds):
    if dataset == "GFS":
        extract = extract_grib
        variable = "Precipitable water"

    elif dataset == "GHI":
        extract = extract_nc
        variable = "GHI"

    else:
        print("Invalid Dataset: ", dataset, len(dataset), type(dataset))

    for idx, f in enumerate(files):
        print(f)
        data = extract(f, variable, latbounds, lonbounds)
        name = "image{:05d}.png".format(idx)
        plot_img(data, folder + name)


def create_video(name, folder, fps=12):
    os.system("ffmpeg -pattern_type glob -i \"{}*.png\" -r {} -vcodec mpeg4 -y {}{}.mp4".format(folder, fps, folder, name))


def read_latlong_args(args):
    return tuple([int(s) for s in args.split(",")])


def get_datetime(args):
    args = args.split("/")
    return datetime(year=int(args[0]), month=int(args[1]), day=int(args[2]))


def long_vid(daily_frames):
    directory = "../visuals/"
    total = len(daily_frames)
    for idx_d, day in enumerate(daily_frames):
        print_progress_bar(idx_d, total)
        for idx_f, frames in enumerate(day):
            filename = directory + "img_{:03d}_{:03d}.png".format(idx_d, idx_f)
            plot_img(frames, filename)

    create_video("all", folder=directory)


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
