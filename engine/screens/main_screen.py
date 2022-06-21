from engine.gui.container import Container
from engine.gui.scatterplot import Explained_scatterplot, Axis_explained
from engine.gui.window import Window
from engine.gui.selector import  Button, Mutex_with_title, String_selector, Number_selector, Mutex_choice, Scrollable_bundle
from engine.gui.listener import Listener
from engine.gui.event_ids import *
from utils import random_colors
from utils import luminosity_change
import numpy as np

class Main_screen():
    def __init__(self, theme, window, manager):
        self.manager = manager
        self.color = theme["color"]; self.background_color = theme["background"]
        self.eight_colors = random_colors(8)
        self.main_view   = Container((0,0), (1., 1.), "main_area", window, manager.uid_generator, color=theme["color"], background_color=theme["background"], filled=False)
        self.main_view.add_leaf(Explained_scatterplot("the scatterplot", pos_pct=(0., 0.), dim_pct=(0.7,1.), parent=self.main_view, uid_generator=manager.uid_generator, color=self.color, manager=manager))

        cont_features = Container((0.7,0.), (0.3, 1.), "features area", self.main_view, manager.uid_generator, color=theme["color"], background_color=theme["background"])
        cont_features.add_leaf(Axis_explained("ax1", (0., 0.), (1., 0.495), parent=cont_features, uid_generator=manager.uid_generator, color=np.array([220, 120, 0]), manager=manager))
        cont_features.add_leaf(Axis_explained("ax2", (0., 0.505), (1., 0.495), parent=cont_features, uid_generator=manager.uid_generator, color=np.array([85, 20, 240]), manager=manager))
        self.main_view.add_container(cont_features)
        window.add_container(self.main_view)


    def schedule_draw(self, to_redraw, all = False):
        if all:
            self.main_view.schedule_draw(to_redraw)

    def ctrl_press(self, dataset_name):
        print("ctrl press in Main_screen() class")

    def ctrl_unpress(self, dataset_name):
        print("ctrl unpress in Main_screen() class")
