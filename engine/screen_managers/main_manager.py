from engine.screen_managers.manager import Manager
from engine.gui.listener import Listener
from engine.gui.event_ids import *
from utils import random_colors
import threading
import numpy as np
from engine.screens.main_screen import Main_screen
from expl_model import Local_explanation_wrapper
from sklearn.neighbors import KDTree
from numba import njit

@njit
def thoughtfull_name(Xld, colours, orig_colours, errors, err_min, span):
    for pt in range(Xld.shape[0]):
        colours[pt] = orig_colours[pt] * (errors[pt]-err_min)/span

class Main_manager(Manager):
    def __init__(self, config, main_window, theme, uid_generator):
        super(Main_manager, self).__init__("main", initial_state=True)
        # self.type_identifier = "main manger"
        self.active  = True
        self.deleted = False
        self.theme = theme

        self.main_window   = main_window
        self.main_window.awaiting_key_press.append(self)
        self.uid_generator = uid_generator
        self.screen        = Main_screen(theme, main_window, self) # builds the screen
        self.lock = threading.Lock()
        self.main_view = self.main_window.containers[0]


        self.scatterplot_redraw_listener = Listener(REDRAW_THINGS, [self])
        self.scatterplot = self.main_view.leaves[0]
        self.ax1_explanation = self.main_view.containers[0].leaves[0]
        self.ax2_explanation = self.main_view.containers[0].leaves[1]
        self.ctrl_pressed = False
        self.Xhd, self.Xld, self.KDtree_ld, self.Y, self.Y_colours = None,None,None,None,None
        self.has_dataset = False
        self.K_select = 150

    def receive_dataset(self, Xhd, Xld, Y, Y_colours):
        self.scatterplot.set_points(Xld, Y_colours)
        self.Xhd = Xhd
        self.Xld = Xld
        self.KDtree_ld = KDTree(Xld, leaf_size=2)
        self.Y = Y
        self.has_dataset = True
        feature_colours = np.tile(np.array((100., 50., 150.)), Xhd.shape[1]).reshape((-1, 3)) + np.random.uniform(size=(Xhd.shape[1],3))*100
        self.ax1_explanation.receive_features(["variable "+str(i) for i in range(Xhd.shape[1])], feature_colours)
        self.ax2_explanation.receive_features(["variable "+str(i) for i in range(Xhd.shape[1])], feature_colours)

    def select_explanation(self, explanation_idx):
        if explanation_idx == -1:
            return
        self.scatterplot.selected_explanation = explanation_idx
        explanation = self.scatterplot.local_explanations[explanation_idx]
        errors = explanation.compute_errors(self.Xhd, self.Xld)
        err_min, err_max = np.min(errors), np.max(errors)
        span = err_max - err_min

        colours      = self.scatterplot.Y_colours
        orig_colours = self.scatterplot.orig_Y_colours
        thoughtfull_name(self.Xld, colours, orig_colours, errors, err_min, span)

        self.ax1_explanation.receive_explanation(explanation, 0)
        self.ax2_explanation.receive_explanation(explanation, 1)


    def click(self, pos, left_click):
        if not self.has_dataset:
            return

        if left_click:
            closest = self.scatterplot.closest_explanation(self.scatterplot.px_pos_to_LD(pos))
            self.select_explanation(closest)
        else:
            pos_in_LD = self.scatterplot.px_pos_to_LD(pos)
            if not self.ctrl_pressed:
                neighbours = self.KDtree_ld.query(pos_in_LD.reshape((1,2)), k=self.K_select, return_distance=False)[0]
                self.new_explanation(neighbours)
            else:
                self.scatterplot.delete_explanation(pos_in_LD)

    def new_explanation(self, neighbours_idx):
        explanation = Local_explanation_wrapper(neighbours_idx, self.Xld, self.Xhd)
        self.scatterplot.add_explanation(explanation)

    def wake_up(self, prev_manager):
        super(Main_manager, self).wake_up(prev_manager)

    def get_notified(self, event_class, id, value, to_redraw = []):
        with self.lock:
            if event_class == REDRAW_THINGS:
                self.redraw_things()
            elif event_class == CTRL_KEY_CHANGE:
                is_pressed, pos = value
                self.ctrl_pressed = is_pressed
                self.scatterplot.selected_explanation = -1
                self.scatterplot.Y_colours = self.scatterplot.orig_Y_colours.copy()
                self.redraw_things()

            elif event_class == SCATTERPLOT_LEFT_CLICK:
                self.click(value[0], left_click=True)
                self.redraw_things()
            elif event_class == SCATTERPLOT_RIGHT_CLICK:
                self.click(value[0], left_click=False)
                self.redraw_things()
            elif event_class == SCATTERPLOT_HOVER:
                return
            else:
                print("unrecognised event received by main_manager: ", event_class, "  (see correspondance in event_ids.py)")



    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        if pressed_keys[0] == 'o':
            print("pressed o")
        return False

    def redraw_things(self):
        self.ask_redraw(self.main_view)
