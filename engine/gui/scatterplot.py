from engine.gui.element import Element, Text
from engine.gui.selector import Button
from engine.gui.listener import Listener
from engine.gui.event_ids import *
import pygame
import numpy as np
import threading
from matplotlib.path import Path
from scipy.linalg import norm

class Feature_bar_thing(Element):
    def __init__(self, coeff, maxabs, name, pos_pct, dim_pct, parent, uid, color, manager):
        super(Feature_bar_thing, self).__init__(pos_pct, dim_pct, name, parent, uid=uid, color=color)
        self.coeff = coeff
        self.maxabs = maxabs
        self.add_text([(0.5,0.1), 2, (0.8,0.8), 18, self.name, self.color]) #  [pos_pct, anchor_id, max_dim, font_size, string, color]
        self.selected = False

    def draw(self, screen):
        if self.selected:
            pygame.draw.rect(screen, self.background_color*2, self.bounding_rect, 0)
            pygame.draw.rect(screen, self.background_color, self.bounding_rect, 2)
            pygame.draw.rect(screen, self.color, self.bounding_rect, 1)
        else:
            pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)

        pos = np.array(self.abs_pos)
        dim = np.array(self.dim)
        a1 = np.array([0., 1.])
        center = pos + 0.5*dim
        L = dim[0]/2

        # middle line
        pygame.draw.line(screen, self.color, center-(0.18*a1*dim[1]), center+(0.15*a1*dim[1]), 1)

        p1 = center[0] + (self.coeff/self.maxabs)*(0.4*dim[0]), center[1]
        pygame.draw.line(screen, self.color*0.5, p1, center, 8)
        super(Feature_bar_thing, self).draw(screen)




        # p1 = self.abs_pos[0], self.abs_pos[1]
        # p2 = self.abs_pos
        # print(self.abs_pos)
        # print(self.parent.abs_pos)
        # print(" -----")
        # pygame.draw.line(screen, self.color, p1, p2, 4)


class Axis_explained(Element):
    def __init__(self, name, pos_pct, dim_pct, parent, uid_generator, color, manager):
        super(Axis_explained, self).__init__(pos_pct, dim_pct, name, parent, uid=uid_generator.get(), color=color)
        self.Nshown = 5
        self.M = 0
        self.features_labels = []
        self.draw_features = False
        self.subthings = []
        self.uid_generator = uid_generator
        self.manager = manager

        self.add_listeners({"Lclick" : AXIS_LEFT_CLICK,
                            "Rclick" : AXIS_RIGHT_CLICK}, to_notify=manager)

    def receive_features(self, features_list, features_colours):
        self.M = len(features_list)
        self.features_labels = features_list.copy()
        self.features_colours = features_colours

    def receive_explanation(self, explanation, axis_nb):
        self.draw_features = True
        if axis_nb == 0:
            component = explanation.get_features_coeff()[0]
        else:
            component = explanation.get_features_coeff()[1]
        abs_comp = np.abs(component)
        maxabs = np.max(abs_comp)
        importance = np.argsort(abs_comp)

        for i in range(min(self.Nshown, self.M)):
            coeff = component[importance[-(i+1)]]
            var_name = self.features_labels[importance[-(i+1)]]
            pos = (0.02, 0.025 + 0.985*i*1./min(self.Nshown, self.M))
            dim = (0.98, 0.9*1/min(self.Nshown, self.M))
            self.subthings.append(Feature_bar_thing(coeff, maxabs, var_name, pos, dim, self, self.uid_generator.get(), self.features_colours[importance[-(i+1)]].astype(int), self.manager))

    def draw(self, screen):
        pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)
        pygame.draw.rect(screen, self.color, self.bounding_rect, 1)
        if not self.draw_features:
            return
        for feat in self.subthings:
            feat.draw(screen)

    def select_feature(self, feature_name, left_click=True):
        self.unselect_features()
        for feat in self.subthings:
            if feat.name == feature_name:
                feat.selected = True

    def unselect_features(self, left_click=True):
        for feat in self.subthings:
            feat.selected = False

    def whom_was_clicked(self, mouse_pos, windows, to_redraw):
        var_name = None
        for feat in self.subthings:
            if feat.point_is_inside(mouse_pos):
                var_name = feat.name
        return var_name

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.point_is_inside(mouse_pos):
            var_name = self.whom_was_clicked(mouse_pos, windows, to_redraw)
            self.on_Lclick_listener.notify((var_name, windows), to_redraw)
            return True
        return False

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.point_is_inside(mouse_pos):
            var_name = self.whom_was_clicked(mouse_pos, windows, to_redraw)
            self.on_Rclick_listener.notify((var_name, windows), to_redraw)
            return True
        return False

class Explained_scatterplot(Element):
    def __init__(self, name, pos_pct, dim_pct, parent, uid_generator, color, manager):
        super(Explained_scatterplot, self).__init__(pos_pct, dim_pct, name, parent, uid=uid_generator.get(), color=color)
        self.listen_to(["Lclick","Rclick","hover"])
        self.uid_generator = uid_generator
        self.lock = threading.Lock()

        self.add_listeners({"Lclick" : SCATTERPLOT_LEFT_CLICK,
                            "Rclick" : SCATTERPLOT_RIGHT_CLICK,
                            "hover"  : SCATTERPLOT_HOVER}, to_notify=manager)
        self.draw_done_listener = Listener(SCATTERPLOT_DRAWING_DONE, [manager])
        self.manager = manager
        '''
        ~~~~~~~~~~~~~~  scatterplot part  ~~~~~~~~~~~~~~
        '''
        self.X_LD      = None
        self.X_LD_px   = None
        self.Y_colours = None
        self.Y_colours_expl = None
        self.orig_Y_colours = None


        self.drawing = False
        self.draw_list = []

        self.local_explanations = []   # contains Local_explanation_wrapper()
        self.centers_in_px      = []
        self.components1_in_px  = []
        self.components2_in_px  = []
        self.selected_explanation = -1

        self.selected_feature = None

        '''
        ~~~~~~~~~~~~~~  pixel stuff  ~~~~~~~~~~~~~~
        '''
        self.x_axis_length, self.y_axis_length = 1., 1.
        self.anchor = (0.0, 1.) # bottom-left position
        self.px_to_LD_offsets = None
        self.px_to_LD_coefs   = None
        self.scale_to_square()    # makes the adjstments to have axis of equal lengths in terms of pixels
        self.update_pos_and_dim() # updates the positions of X and such according to the adjusted axis

    def draw(self, screen):
        if  self.X_LD_px is None:
            return
        pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)

        # dot_colours = self.Y_colours_expl
        dot_colours = self.Y_colours

        thickness = 1
        m1 = np.array([0, 1])
        m2 = np.array([1, 0])
        '''
        draw the scatterplot
        '''
        N = self.X_LD_px.shape[0]
        coord = self.X_LD_px
        for i in range(N):
            pygame.draw.circle(screen, dot_colours[i], coord[i], thickness)

        if self.drawing:
            for i in range(len(self.draw_list)):
                pygame.draw.circle(screen, (np.array([100,200,100])), self.draw_list[i], thickness)

        '''
        draw the explanations
        '''
        Nexplanations = len(self.local_explanations)
        sel_idx = -1
        if self.selected_feature is not None and self.feature_names is not None:
            for i in range(len(self.feature_names)):
                if self.feature_names[i] == self.selected_feature:
                    sel_idx = i

        center_colour = np.array([161., 94., 249.])
        center_colour2 = np.array([249., 151., 94.])
        comp1_colour  = np.array([161., 94., 249.])
        comp2_colour  = np.array([249., 151., 94.])
        lavender = np.array([80, 0, 140])
        # lavender = np.array([35, 240, 50])
        a1 = np.array([1., -1.])
        selected_idx = -1
        for i in range(Nexplanations):

            expl = self.local_explanations[i]
            expl_W1, expl_W2 = expl.get_features_coeff()
            expl_W1 = np.abs(expl_W1) / np.sum(np.abs(expl_W1))
            expl_W2 = np.abs(expl_W2) / np.sum(np.abs(expl_W2))



            # print(np.round( , 2))
            # print(np.round(np.abs(expl_W2) / np.sum(np.abs(expl_W2)) , 2))

            center = self.centers_in_px[i]
            comp1  = self.components1_in_px[i]
            comp2  = self.components2_in_px[i]
            p1 = center + comp1*a1 * 60
            p2 = center + comp2*a1 * 60
            if i == self.selected_explanation:
                pygame.draw.aaline(screen, comp1_colour, center, p1, 4)
                pygame.draw.aaline(screen, comp2_colour, center, p2, 4)
                pygame.draw.circle(screen, center_colour2, center, 6)
                selected_idx = i
            else:
                pygame.draw.aaline(screen, comp1_colour, center, p1, 2)
                pygame.draw.aaline(screen, comp2_colour, center, p2, 2)
                pygame.draw.circle(screen, center_colour, center, 4)

            # seleted feature importance
            v1 = (p1 - center) * expl_W1[sel_idx]
            v2 = (p2 - center) * expl_W2[sel_idx]
            pygame.draw.line(screen, comp1_colour, center, center+v1, 5)
            pygame.draw.line(screen, comp2_colour, center, center+v2, 5)

        if selected_idx != -1:
            sample_idxs = self.local_explanations[selected_idx].sample_idx
            for idx in sample_idxs:
                pygame.draw.circle(screen, lavender, coord[idx], 3, 1)


    def delete(self):
        with self.lock:
            super(Explained_scatterplot, self).delete()

    def add_explanation(self, explanation):
        self.local_explanations.append(explanation)
        self.centers_in_px.append(self.LD_pos_to_px(explanation.center))
        self.components1_in_px.append(explanation.axis2d[0])
        self.components2_in_px.append(explanation.axis2d[1])

    def closest_explanation(self, pos):
        dwinner, iwinner = 1e12, -1
        for i in range(len(self.local_explanations)):
            expl_pos = self.local_explanations[i].center
            d = np.sum((pos-expl_pos)**2)
            if d < dwinner:
                dwinner = d
                iwinner = i
        return iwinner

    def delete_explanation(self, pos):
        if len(self.local_explanations) == 0: return
        iwinner = self.closest_explanation(pos)
        del self.local_explanations[iwinner]
        del self.centers_in_px[iwinner]
        del self.components1_in_px[iwinner]
        del self.components2_in_px[iwinner]

    def set_points(self, X_LD, Y_colours, Y_colours_expl, feature_names):
        self.px_to_LD_coefs   = None
        self.px_to_LD_offsets = None
        self.feature_names = feature_names
        if X_LD is not None:
            self.X_LD       = X_LD
            self.X_LD_px    = np.zeros_like(self.X_LD)
        if Y_colours is not None:
            self.Y_colours  = Y_colours
            self.Y_colours_expl = Y_colours_expl
            self.orig_Y_colours = Y_colours.copy()
        self.rebuild_points()

    def px_pos_to_LD(self, mouse_pos, no_reshape = False):
        if self.X_LD is None or self.px_to_LD_coefs is None:
            return
        out = np.array([mouse_pos[0]*self.px_to_LD_coefs[0] + self.px_to_LD_offsets[0], mouse_pos[1]*self.px_to_LD_coefs[1] + self.px_to_LD_offsets[1]])
        if no_reshape:
            return out
        else:
            return out.reshape((1, 2))

    def LD_pos_to_px(self, Xi):
        if self.X_LD is None or self.px_to_LD_coefs is None:
            return
        # return np.array([(Xi[0]-self.px_to_LD_offsets[0])/self.px_to_LD_coefs[0], ])
        return (Xi - self.px_to_LD_offsets) / self.px_to_LD_coefs

    def rebuild_points(self):
        if self.X_LD is None or self.Y_colours is None:
            return

        ax_px_len = self.x_axis_length*self.dim[0]
        ax1_min, ax2_min = np.min(self.X_LD, axis=0)
        ax1_max, ax2_max = np.max(self.X_LD, axis=0)

        ax1_wingspan = ax1_max - ax1_min + 1e-6
        ax2_wingspan = ax2_max - ax2_min + 1e-6
        x_offset = self.abs_pos[0] + self.anchor[0]*self.dim[0]
        y_offset = self.abs_pos[1] + self.anchor[1]*self.dim[1]

        idx = 0
        for obs in self.X_LD:
            self.X_LD_px[idx][0] = ax_px_len*((obs[0]-ax1_min)/ax1_wingspan)+x_offset
            self.X_LD_px[idx][1] = y_offset-ax_px_len*((obs[1]-ax2_min)/ax2_wingspan)
            idx += 1

        self.px_to_LD_offsets = (-x_offset*ax1_wingspan/ax_px_len +ax1_min, y_offset*ax2_wingspan/ax_px_len +ax2_min)
        self.px_to_LD_coefs   = (ax1_wingspan/ax_px_len, -ax2_wingspan/ax_px_len)


    def scale_to_square(self):
        self.px_to_LD_coefs   = None
        self.px_to_LD_offsets = None
        axis_target_length = 1.
        yx_ratio = self.dim[1]/self.dim[0]
        if yx_ratio < 1: # we need to separate both cases because we can only scale an axis by getting it smaller (or else we would have an axis bigger that the parent container)
            self.x_axis_length = axis_target_length*yx_ratio
            self.y_axis_length = axis_target_length
        else:
            self.x_axis_length = axis_target_length
            self.y_axis_length = axis_target_length/yx_ratio
        self.rebuild_points()

    def update_pos_and_dim(self):
        self.dim     = (self.parent.dim[0]*self.dim_pct[0], self.parent.dim[1]*self.dim_pct[1])
        self.rel_pos = (self.parent.dim[0]*self.pos_pct[0], self.parent.dim[1]*self.pos_pct[1])
        self.abs_pos = (self.parent.abs_pos[0]+self.rel_pos[0], self.parent.abs_pos[1]+self.rel_pos[1])
        self.background_rect = pygame.Rect(self.abs_pos, self.dim)
        self.bounding_rect   = pygame.Rect((self.abs_pos[0], self.abs_pos[1]), (self.dim[0], self.dim[1]+5))
        self.scale_to_square()



    def point_is_within_plot(self, pos):
        x_relative = pos[0]-self.abs_pos[0]-self.anchor[0]*self.dim[0]
        y_relative = -(pos[1]-self.abs_pos[1]-self.anchor[1]*self.dim[1])
        if x_relative > 0 and x_relative < self.x_axis_length*self.dim[0]:
            if y_relative > 0 and y_relative < self.y_axis_length*self.dim[1]:
                return True
        return False




    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.point_is_within_plot(mouse_pos):
            self.on_Lclick_listener.notify((mouse_pos, windows), to_redraw)
            return True
        return False

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.point_is_within_plot(mouse_pos):
            self.on_Rclick_listener.notify((mouse_pos, windows), to_redraw)
            self.drawing = True
            self.draw_list = [mouse_pos]
            self.schedule_awaiting(awaiting_mouse_move)
            self.schedule_awaiting(awaiting_mouse_release)
            return True
        return False


    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.point_is_within_plot(mouse_pos):
            if self.drawing:
                self.draw_list.append(mouse_pos)
            # self.on_hover_listener.notify((mouse_pos), to_redraw)
            self.schedule_draw(to_redraw)
            return True
        return True

    def compute_which_points_are_inside(self, draw_trajectory):
        idxs = np.where(Path(draw_trajectory).contains_points(self.X_LD_px))[0]
        return np.array(idxs)

    def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
        self.drawing = False
        if len(self.draw_list) > 3:
            self.draw_list.append(self.draw_list[0])
            inside = self.compute_which_points_are_inside(self.draw_list)
            if inside.shape[0] < 10:
                return True
            self.draw_done_listener.notify((inside), to_redraw)
        return True

    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        return True

    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        return True
