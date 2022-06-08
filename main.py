import sys, os
from utils import get_gui_config, luminosity_change
from engine.gui.shared_variable import Shared_variable
from engine.gui.gui import Gui, Id_generator
from engine.gui.container import Container
from engine.gui.window import Window
from engine.screen_managers.main_manager import Main_manager
import pygame, threading
import numpy as np



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


def run_both(gui_args, worker_function, worker_args):
    gui = build_gui(gui_args)

    worker_args[0]['scatterplot redraw'] = gui.main_manager.scatterplot_redraw_listener
    worker_thread = threading.Thread(target=worker_function, args=worker_args)
    worker_thread.start()

    gui.routine()

def worker_function(dict):
    X, Y = dict['X'], dict['Y']
    N, M = X.shape

    sample = X[:100]
    center = selected_points[10]
    Local_explanation(sample, center=None)




def worker_function_test(dict):
    X = dict['X']
    colours = (np.random.uniform(size=(X.shape[0], 3))*254.).astype(int)
    momentums = np.zeros_like(X)
    listener = dict['scatterplot redraw']
    import time
    for i in range(80):
        # update the point locations
        grads = np.random.normal(size=X.shape)
        momentums = 0.9*momentums - grads
        X += 0.05 * momentums
        # notify the screen for a redraw
        listener.notify((X, colours))
        time.sleep(0.05)


if __name__ == "__main__":

    worker_args = [{'X':np.random.uniform(size=(1000, 2)), 'Y':np.zeros(1000, dtype=int)}]
    run_both(gui_args = "-w" if "-w" in sys.argv else "", worker_function=worker_function, worker_args=worker_args)
