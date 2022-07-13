import sys, os
from utils import get_gui_config, luminosity_change
from engine.gui.shared_variable import Shared_variable
from engine.gui.gui import Gui, Id_generator
from engine.gui.container import Container
from engine.gui.window import Window
from engine.screen_managers.main_manager import Main_manager
import pygame, threading
import numpy as np
from data_fetchers import *



def get_all_data(dataset_name):
    if dataset_name == 'airfoil':
        Xhd, Y, feature_names = get_airfoil()
    elif dataset_name == 'satellite':
        Xhd, Y, feature_names = get_satellite()
    elif dataset_name == 'winequality':
        Xhd, Y, feature_names = get_winequality()
    # add your own get_myDataset() function here in a new 'elif'
    # write get_myDataset() in data_fetchers.py for some examples
    else:
        print('wrong dataset name: '+str(dataset_name)+ ' \n\n\n\n\n\n'); 1/0
    return Xhd, get_Xld(dataset_name, Xhd), Y, feature_names

def main():
    dataset_names = ['airfoil', 'satellite', 'winequality']
    Xhd, Xld, Y, feature_names= get_all_data(dataset_names[0])

    worker_args = [{'Xhd':Xhd, 'Y':Y, 'Xld':Xld, 'feature_names': feature_names}]
    run_both(gui_args = "-w" if "-w" in sys.argv else "", worker_function=explain_things, worker_args=worker_args)




def explain_things(dict):
    Xhd, Y, Xld, feature_names = dict['Xhd'], dict['Y'], dict['Xld'], dict['feature_names']
    Y_colours = label_colours(Y)
    Y_colours_expl = np.tile(np.array([213., 60., 245.]), Xld.shape[0]).reshape((Xld.shape[0], 3))


    methods = ['pca', 'biot']
    expl_method = methods[1]
    threshold = 10. # try using method = pca for a quick estimation of the threshold then set method = 'biot'
    min_support = 10 # when partitining the LD space, if |local_sample| < min_support the stop recursive split, even if error(Xld_hat) is under the threshold

    event_manager = dict['manager']
    event_manager.receive_dataset(Xhd, Xld, Y, Y_colours, Y_colours_expl, feature_names=feature_names)
    event_manager.method = expl_method
    event_manager.explain_full_dataset(algo=event_manager.method, threshold=threshold, min_support=min_support)


def run_both(gui_args, worker_function, worker_args):
    gui = build_gui(gui_args)
    worker_args[0]['manager']            = gui.main_manager
    worker_args[0]['scatterplot redraw'] = gui.main_manager.scatterplot_redraw_listener
    worker_thread = threading.Thread(target=worker_function, args=worker_args)
    worker_thread.start()
    gui.routine()



def make_main_screen(theme, config, uid_generator):
	main_window = Window((0,0), config["resolution"], "main window", close_on_click_outside=False, uid_generator=uid_generator, color=theme["main"]["color"], background_color=theme["main"]["background"])
	manager     = Main_manager(config, main_window, theme["main"], uid_generator)
	return {"manager":manager, "window":main_window}


def make_theme(print_mode):
	main_theme  = {"background" : np.array([15,0,5]), "color" : np.array([180, 80,10])}
	if print_mode:
		main_theme["background"]  = np.array([255,255,255])
		main_theme["color"]  = luminosity_change(main_theme["color"], -300)
	return {"main" : main_theme}


def build_gui(gui_args):
	uid_generator = Id_generator()
	config        = get_gui_config(gui_args)
	display       = init_screen(config)
	running_flag  = Shared_variable(True)
	theme		  = make_theme(config["print mode"])
	main_screen   = make_main_screen(theme, config, uid_generator)
	return Gui(display, theme, config, running_flag, main_screen)

def init_screen(config):
	if not config["windowed"]:
		pygame.display.init()
		screen_info = pygame.display.Info()
		config["resolution"] = (screen_info.current_w, screen_info.current_h)
		return pygame.display.set_mode(config["resolution"], pygame.FULLSCREEN)
	else:
		return pygame.display.set_mode(config["resolution"])

def Kmeans_colours(Y):
    from sklearn.cluster import KMeans
    K = np.unique(Y).shape[0]
    sample = np.random.uniform(size = 4000)*0.9 + 0.1
    return KMeans(n_clusters=K).fit(sample).cluster_centers_

def label_colours(Y):
    unq_Y = np.unique(Y)
    is_classification = True
    if type(Y[0].item()) == float:
        is_classification = False
    if is_classification:
        return Kmeans_colours(Y)
    else:
        ymin, ymax = np.min(Y), np.max(Y)
        Y -= ymin
        if ymax - ymin > 1e-9:
            Y /= (ymax - ymin)
        c2 = np.array([190, 150, 0])
        c1 = np.array([0, 40, 250])
        return (Y*c1[:,None] + (1-Y)*c2[:,None]).T

def get_Xld(dataset_name, Xhd):
    Xld = None
    try:
        Xld = np.load("saved_Xld/"+str(dataset_name)+".npy")
    except:
        print("no saved embedding found, computing a tSNE embedding and saving it for later...")
        from sklearn.manifold import TSNE
        Xld = TSNE(n_components=2, init='pca').fit_transform(Xhd)
        np.save("saved_Xld/"+str(dataset_name)+".npy", Xld)
    return Xld

if __name__ == "__main__":
    main()
