from engine.gui.element import Element, Text
from engine.gui.selector import Button
from engine.gui.listener import Listener
from engine.gui.event_ids import *
import pygame
import numpy as np
import threading
from matplotlib.path import Path

class Scatterplot(Element):
    def __init__(self, name, pos_pct, dim_pct, parent, uid_generator, color, manager):
        super(Scatterplot, self).__init__(pos_pct, dim_pct, name, parent, uid=uid_generator.get(), color=color)
        self.listen_to(["Lclick","Rclick","hover"])
        self.uid_generator = uid_generator
        self.lock = threading.Lock()

        self.add_listeners({"Lclick" : SCATTERPLOT_LEFT_CLICK,
                            "Rclick" : SCATTERPLOT_RIGHT_CLICK,
                            "hover"  : SCATTERPLOT_HOVER}, to_notify=manager)

        '''
        ~~~~~~~~~~~~~~  scatterplot part  ~~~~~~~~~~~~~~
        '''
        self.X_LD      = None
        self.X_LD_px   = None
        self.Y_colours = None

        '''
        ~~~~~~~~~~~~~~  pixel stuff  ~~~~~~~~~~~~~~
        '''
        self.x_axis_length, self.y_axis_length = 1., 1.
        self.anchor = (0.0, 1.) # bottom-left position
        self.px_to_LD_offsets = None
        self.px_to_LD_coefs   = None
        self.scale_to_square()    # makes the adjstments to have axis of equal lengths in terms of pixels
        self.update_pos_and_dim() # updates the positions of X and such according to the adjusted axis

    def delete(self):
        with self.lock:
            super(Scatterplot, self).delete()

    def set_points(self, X_LD, Y_colours):
        self.px_to_LD_coefs   = None
        self.px_to_LD_offsets = None
        if X_LD is not None:
            self.X_LD       = X_LD
            self.X_LD_px    = np.zeros_like(self.X_LD)
        if Y_colours is not None:
            self.Y_colours  = Y_colours
        self.rebuild_points()

    def px_pos_to_LD(self, mouse_pos):
        if self.X_LD is None or self.px_to_LD_coefs is None:
            return
        return np.array([mouse_pos[0]*self.px_to_LD_coefs[0] + self.px_to_LD_offsets[0], mouse_pos[1]*self.px_to_LD_coefs[1] + self.px_to_LD_offsets[1]]).reshape((1, 2))

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

    def draw(self, screen):
        if  self.X_LD_px is None:
            return
        pygame.draw.rect(screen, self.background_color, self.bounding_rect, 0)

        thickness = 2
        m1 = np.array([0, 1])
        m2 = np.array([1, 0])
        '''
        draw the scatterplot
        '''
        N = self.X_LD_px.shape[0]
        coord = self.X_LD_px
        for i in range(0, N):
            pygame.draw.line(screen, self.Y_colours[i], coord[i]-m1, coord[i]+m1, thickness)
            pygame.draw.line(screen, self.Y_colours[i], coord[i]-m2, coord[i]+m2, thickness)


    def point_is_within_plot(self, pos):
        x_relative = pos[0]-self.abs_pos[0]-self.anchor[0]*self.dim[0]
        y_relative = -(pos[1]-self.abs_pos[1]-self.anchor[1]*self.dim[1])
        if x_relative > 0 and x_relative < self.x_axis_length*self.dim[0]:
            if y_relative > 0 and y_relative < self.y_axis_length*self.dim[1]:
                return True
        return False




    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        print("clicked in the")
        if self.point_is_within_plot(mouse_pos):
            self.on_Lclick_listener.notify((mouse_pos, windows), to_redraw)
            return True
        return False

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        if self.point_is_within_plot(mouse_pos):
            self.on_Rclick_listener.notify((mouse_pos, windows), to_redraw)
            return True
        return False


    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        if self.point_is_within_plot(mouse_pos):
            self.on_hover_listener.notify((mouse_pos), to_redraw)
            return True
        return True

    def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
        return True

    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        return True

    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        return True


class Explainable_scatterplot(Scatterplot):
    def __init__(self, name, pos_pct, dim_pct, parent, uid_generator, color, manager):
        super(Explainable_scatterplot, self).__init__(pos_pct, dim_pct, name, parent, uid=uid_generator.get(), color=color)
