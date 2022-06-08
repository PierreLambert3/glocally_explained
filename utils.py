import sys
import numpy as np
from scipy.cluster.vq import kmeans2

def luminosity_change(c, strength): # negative for darker, positive for lighter
    total = np.sum(c)
    if strength > 0:
        c0 = min(255,int(strength*c[0]/total) + c[0])
        c1 = min(255,int(strength*c[1]/total) + c[1])
        c2 = min(255, int(strength*c[2]/total) + c[2])
    else:
        c0 = max(0,int(strength*c[0]/total) + c[0])
        c1 = max(0,int(strength*c[1]/total) + c[1])
        c2 = max(0, int(strength*c[2]/total) + c[2])
    return np.array([c0, c1, c2])

def get_gui_config(argv):
    config = {"print mode": False, "windowed": False, "resolution": (1024, 768), "frame time": 0.025}
    try:
        with open('config.txt', 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                windowed_split = line.split('- windowed')
                if len(windowed_split) > 1 and windowed_split[1][0] != '_':
                    value = windowed_split[1].split('=')[1].split('#')[0].strip()
                    windowed = ("True" in value or "true" in value or "1" in value)
                    config["windowed"] = windowed

                windowed_resolution_split = line.split('- windowed_resolution')
                if len(windowed_resolution_split) > 1:
                    value = windowed_resolution_split[1].split('=')[1].split('#')[0].strip()
                    w = int(value.split("x")[0].strip())
                    h = int(value.split("x")[1].strip())
                    config["resolution"] = (w, h)

                printable_split = line.split('- printable')
                if len(printable_split) > 1:
                    value = printable_split[1].split('=')[1].split('#')[0].strip()
                    printable = ("True" in value or "true" in value or "1" in value)
                    config["print mode"] = printable

                frame_time_split = line.split('- target frame time (seconds)')
                if len(frame_time_split) > 1:
                    value = frame_time_split[1].split('=')[1].split('#')[0].strip()
                    frame_time = float(value)
                    config["frame time"] = frame_time

                shortcut_split = line.split('- main_screen_key')
                if len(shortcut_split) > 1:
                    value = shortcut_split[1].split('=')[1].split('#')[0].strip()
                    config["main screen key"] = value
                shortcut_split = line.split('- relQA_screen_key')
                if len(shortcut_split) > 1:
                    value = shortcut_split[1].split('=')[1].split('#')[0].strip()
                    config["relQA screen key"] = value
                shortcut_split = line.split('- absQA_screen_key')
                if len(shortcut_split) > 1:
                    value = shortcut_split[1].split('=')[1].split('#')[0].strip()
                    config["absQA screen key"] = value

                regr_color_split = line.split('- regr_var_color_with_span')
                if len(regr_color_split) > 1:
                    value = regr_color_split[1].split('=')[1].split('#')[0].strip()
                    regr_color = ("True" in value or "true" in value or "1" in value)
                    config["regr color with span"] = regr_color

                GPU_feature_sel = line.split('- DL_feature_selection_on_GPU')
                if len(GPU_feature_sel) > 1:
                    value = GPU_feature_sel[1].split('=')[1].split('#')[0].strip()
                    boolean = ("True" in value or "true" in value or "1" in value)
                    config["DL_feature_selection_on_GPU"] = boolean

                default_N = line.split('- default_N')
                if len(default_N) > 1:
                    value = default_N[1].split('=')[1].split('#')[0].strip()
                    config["default_N"] = int(value.strip())
    except:
        die("error when reading config file. \nmake sure the config file is at the root folder (next to main.py) \nalso make sure that each element is written in the format : \" - windowed            = False    #  If passing arguments \"-w\" then windowed is set to True.\"")

    if '-p' in str(argv):
        config["print mode"] = True
    if '-w' in str(argv):
        config["windowed"] = True
    return config

def die(message):
    print("\n\n    ERROR :")
    print(message)
    print()
    sys.exit()

# Kmeans on a randomly populated RGB space
def random_colors(N_colors):
    L = 14
    R = 25
    data = np.zeros(((R-L)**3, 3))
    observation = 0
    np.random.seed(7)
    for r in range(L, R):
        for g in range(L, R):
            for b in range(L, R):
                data[observation]=np.array([r*10*max(np.random.uniform(),np.random.uniform()),g*10*max(np.random.uniform(),np.random.uniform()),b*10*max(np.random.uniform(),np.random.uniform())])
                observation+=1
    codebook, _ = kmeans2(data, N_colors)
    colors = []
    for c in codebook:
        colors.append(np.array([min(max(int(c[0]), 4), 254),  min(max(int(c[1]), 4), 254),  min(max(int(c[2]), 4), 254)]))
    np.random.seed(None)
    return np.array(colors)
