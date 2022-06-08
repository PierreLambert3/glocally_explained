import pygame
from engine.gui.listener import Listener

class Element:
    def __init__(self, position, dimension, name, parent, uid, color=None, background_color=None, \
        on_draw_listener=None, on_Lclick_listener=None, on_Rclick_listener=None, on_hover_listener=None, on_scroll_listener=None,\
        listening_hover=False, listening_Lclick=False, listening_Rclick=False, listening_scroll=False, attach_to_parent=False):
        self.deleted = False
        self.uid    = uid
        self.parent = parent
        self.name   = name
        self.hidden = False
        self.already_drawn = False # checked when to_redraw is emptied, in case element is added twice to the list
        self.update_colors(color, background_color)
        self.set_initial_positions(position, dimension)

        # ---------------  shapes: encoded as schematics, and pre-built for quick drawing  ---------------
        self.rects_schematics = [] # [[pos_pct, dim_pct, thickness, color], ... ]
        self.lines_schematics = [] # [[pos_pct1, pos_pct2, thickness, color], ... ]
        self.texts_schematics = [] # [[pos_pct, anchor_id, max_dim, font_size, string , color], ... ]
        self.rects = [] # [rectangle, ... ]
        self.lines = [] # [[abs_pos1, abs_pos2], ... ]
        self.texts = [] # [Text_class, ... ]
        self.bounding_rect = pygame.Rect(self.abs_pos, self.dim)

        # ---------------  only propagate X if listening to X ---------------
        self.default_listening_hover  = listening_hover
        self.default_listening_Lclick = listening_Lclick
        self.default_listening_Rclick = listening_Rclick
        self.default_listening_scroll = listening_scroll
        self.listening_hover   = self.default_listening_hover
        self.listening_Lclick  = self.default_listening_Lclick
        self.listening_Rclick  = self.default_listening_Rclick
        self.listening_scroll  = self.default_listening_scroll

        # ---------------  ignore the next on_awaiting_X --------------- (use case: we hide something, but it is awaiting_X, when X happens that thing would redraw itself)
        self.stop_awaiting_hover = False
        self.stop_awaiting_click = False
        self.stop_awaiting_key   = False

        # ---------------  listeners: they notify other elements of a certain event ---------------
        self.notify_on_draw   = False
        self.notify_on_Lclick  = False
        self.notify_on_Rclick = False
        self.notify_on_hover  = False
        self.notify_on_scroll = False
        self.on_draw_listener   = on_draw_listener
        self.on_Lclick_listener = on_Lclick_listener
        self.on_Rclick_listener = on_Rclick_listener
        self.on_hover_listener  = on_hover_listener
        self.on_scroll_listener = on_scroll_listener
        self.update_listener_booleans()

        if attach_to_parent:
            self.parent.add_leaf(self)


    def delete(self):
        self.deleted = True
        self.stop_awaiting_hover = True
        self.stop_awaiting_click = True
        self.stop_awaiting_key   = True

        self.bounding_rect = None
        for t in range(len(self.texts)):
            if self.texts[t] is not None:
                self.texts[t].delete()
                self.texts[t] = None
        self.texts = []

        for l in range(len(self.lines)):
            self.lines[l] = None
        self.lines = []

        for r in range(len(self.rects)):
            self.rects[r] = None
        self.rects = []

        self.rects_schematics = []
        self.texts_schematics = []
        self.lines_schematics = []
        if self.on_draw_listener is not None:
            self.on_draw_listener.delete()
        if self.on_Lclick_listener is not None:
            self.on_Lclick_listener.delete()
        if self.on_Rclick_listener is not None:
            self.on_Rclick_listener.delete()
        if self.on_hover_listener is not None:
            self.on_hover_listener.delete()
        if self.on_scroll_listener is not None:
            self.on_scroll_listener.delete()
        self.on_draw_listener = None
        self.on_Lclick_listener = None
        self.on_Rclick_listener = None
        self.on_hover_listener = None
        self.on_scroll_listener = None
        self.parent = None

    def reset(self):
        return

    # either set colors to specific values, or inherit colors from parent
    def update_colors(self, color=None, background_color=None):
        if color is None:
            self.color = self.parent.color
        else:
            self.color = color
        if background_color is None:
            self.background_color = self.parent.background_color
        else:
            self.background_color = background_color

    def set_initial_positions(self, position, dimension):
        self.dim_pct = dimension
        self.dim     = (int(self.parent.dim[0]*self.dim_pct[0]), int(self.parent.dim[1]*self.dim_pct[1]))
        self.pos_pct = position
        self.rel_pos = (int(self.parent.dim[0]*self.pos_pct[0]), int(self.parent.dim[1]*self.pos_pct[1]))
        self.abs_pos = (int(self.parent.abs_pos[0]+self.rel_pos[0]), int(self.parent.abs_pos[1]+self.rel_pos[1]))

    def disable(self):
        self.hidden = True
        self.listening_hover  = False
        self.listening_Lclick  = False
        self.listening_Rclick  = False
        self.listening_scroll = False
        self.stop_awaiting_hover = True
        self.stop_awaiting_click = True
        self.stop_awaiting_key   = True

    def enable(self):
        self.hidden = False
        self.listening_hover  = self.default_listening_hover
        self.listening_Lclick  = self.default_listening_Lclick
        self.listening_Rclick  = self.default_listening_Rclick
        self.listening_scroll = self.default_listening_scroll
        self.stop_awaiting_hover = False
        self.stop_awaiting_click = False
        self.stop_awaiting_key   = False

    def update_listener_booleans(self):
        if self.on_draw_listener is not None:
            self.notify_on_draw = True
        if self.on_Lclick_listener is not None:
            self.notify_on_Lclick = True
        if self.on_Rclick_listener is not None:
            self.notify_on_Rclick = True
        if self.on_hover_listener is not None:
            self.notify_on_hover = True
        if self.on_scroll_listener is not None:
            self.notify_on_scroll = True

    # event_types --> something like {"Lclick":LISTENER_ID1, "hover":LISTENER_ID2}
    #                 this allows for more precise IDs for the event and thus reduces the work on the side of the manager
    def add_listeners(self, event_types, to_notify):
        for event_type in event_types:
            EVENT_ID = event_types[event_type] #  {"event name": ID_OF_THE_EVENT}
            if event_type == "draw":
                self.on_draw_listener = Listener(EVENT_ID, [to_notify])
                self.notify_on_draw = True
            elif event_type == "Lclick":
                self.default_listening_Lclick = True
                self.listening_Lclick         = True
                self.on_Lclick_listener = Listener(EVENT_ID, [to_notify])
                self.notify_on_Lclick = True
            elif event_type == "Rclick":
                self.default_listening_Rclick = True
                self.listening_Rclick         = True
                self.on_Rclick_listener = Listener(EVENT_ID, [to_notify])
                self.notify_on_Rclick = True
            elif event_type == "hover":
                self.default_listening_hover = True
                self.listening_hover         = True
                self.on_hover_listener = Listener(EVENT_ID, [to_notify])
                self.notify_on_hover = True
            elif event_type == "scroll":
                self.default_listening_scroll = True
                self.listening_scroll         = True
                self.on_scroll_listener = Listener(EVENT_ID, [to_notify])
                self.notify_on_scroll = True
            else:
                print("\n\t UNRECOGNISED EVENT_TYPE IN ADD_LISTENER():  ", event_type)
                1/0

    def listen_to(self, event_list, listener = None):
        for e in event_list:
            if e == "hover":
                self.default_listening_hover = True
                self.listening_hover         = True
            elif e == "Lclick":
                self.default_listening_Lclick = True
                self.listening_Lclick         = True
            elif e == "Rclick":
                self.default_listening_Rclick = True
                self.listening_Rclick         = True
            elif e == 'scroll':
                self.default_listening_scroll = True
                self.listening_scroll         = True
            else:
                1/0

    def schedule_draw(self,  to_redraw):
        # if we have an on_draw listener, we dont draw and let the notified things do the drawing
        if self.notify_on_draw:
            self.on_draw_listener.notify(True, to_redraw)
        else:
            to_redraw.append(self)
            self.already_drawn = False

    def schedule_awaiting(self, awaiting_X):
        if not self.is_already_in(awaiting_X):
            awaiting_X.append(self)

    def draw(self, screen): # return True if not a hidden element
        self.already_drawn = True
        if self.hidden:
            return False
        for i, r in enumerate(self.rects):
            pygame.draw.rect(screen, self.rects_schematics[i][3], r, self.rects_schematics[i][2])
        for i, r in enumerate(self.lines):
            pygame.draw.line(screen, self.lines_schematics[i][3], self.lines[i][0], self.lines[i][1], self.lines_schematics[i][2])
        for txt in self.texts:
            txt.draw(screen)
        return True

    def flip_colors(self):
        tmp = self.color
        self.color = self.background_color
        self.background_color = tmp

    def pct_to_px(self, pct_of_self): # pct of self: (x_pct, y_pct)
        return (self.abs_pos[0] + self.dim[0]*pct_of_self[0], self.abs_pos[1] + self.dim[1]*pct_of_self[1])

    def point_is_inside(self, p): # p in absolute value
        return (p[0]>=self.abs_pos[0] and p[1]>=self.abs_pos[1] and p[0]<=self.abs_pos[0]+self.dim[0] and p[1]<=self.abs_pos[1]+self.dim[1])

    def get_notified(self, listener_class, listener_id, listener_value, to_redraw):
        return

    def is_already_in(self, L):
        for e in L:
            if e.uid == self.uid:
                return True
        return False

    # propagate_X -> only called if i'm hovered/clicked/scrolled, no need to check with self.point_is_inside()
    # also it's the parent that checks wether i'm listening_X
    # return:   False if you want the parent container to continue popagating to others
    #           True  if you want the parent container to stop propagation
    def propagate_hover(self, to_redraw, mouse_pos, pressed_special_keys, awaiting_mouse_move):
        return True # stop parent from propagating further

    def propagate_Lmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        return True

    def propagate_Rmouse_down(self, to_redraw, windows, mouse_pos, mouse_button_status, pressed_special_keys, awaiting_mouse_move, awaiting_mouse_release):
        return True

    def propagate_scroll(self, to_redraw, mouse_pos, scroll, pressed_special_keys):
        return True

    def on_awaited_mouse_move(self, to_redraw, mouse_positions, mouse_button_status, pressed_special_keys):
        return True # if true: can remove else: keep in awaiting

    def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
        return True

    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        return True

    def update_pos_and_dim(self):
        self.dim     = (self.parent.dim[0]*self.dim_pct[0], self.parent.dim[1]*self.dim_pct[1])
        self.rel_pos = (self.parent.dim[0]*self.pos_pct[0], self.parent.dim[1]*self.pos_pct[1])
        self.abs_pos = (self.parent.abs_pos[0]+self.rel_pos[0], self.parent.abs_pos[1]+self.rel_pos[1])
        self.rebuild_shapes()
        self.bounding_rect = pygame.Rect(self.abs_pos, self.dim)

    def add_rect(self, schematic):  # [pos_pct, dim_pct, thickness, color]
        pos = self.abs_pos[0]+schematic[0][0]*self.dim[0], self.abs_pos[1]+schematic[0][1]*self.dim[1]
        dim = self.dim[0]*schematic[1][0], self.dim[1]*schematic[1][1]
        self.rects_schematics.append(schematic)
        self.rects.append(pygame.Rect(pos, dim))

    def add_line(self, schematic): # [pos_pct1, pos_pct2, thickness, color]
        pos1 = self.abs_pos[0]+schematic[0][0]*self.dim[0], self.abs_pos[1]+schematic[0][1]*self.dim[1]
        pos2 = self.abs_pos[0]+schematic[1][0]*self.dim[0], self.abs_pos[1]+schematic[1][1]*self.dim[1]
        self.lines_schematics.append(schematic)
        self.lines.append([pos1, pos2])

    def add_text(self, schematic): # [pos_pct, anchor_id, max_dim, font_size, string, color]
        txt = Text(schematic[0], schematic[1], schematic[2], schematic[3], schematic[4], self, color=schematic[5])
        self.texts_schematics.append(schematic)
        self.texts.append(txt)

    def rebuild_rects(self):
        self.rects.clear()
        for schematic in self.rects_schematics:
            pos = self.abs_pos[0]+schematic[0][0]*self.dim[0], self.abs_pos[1]+schematic[0][1]*self.dim[1]
            dim = self.dim[0]*schematic[1][0], self.dim[1]*schematic[1][1]
            self.rects.append(pygame.Rect(pos, dim))

    def rebuild_lines(self):
        self.lines.clear()
        for schematic in self.lines_schematics:
            pos1 = self.abs_pos[0]+schematic[0][0]*self.dim[0], self.abs_pos[1]+schematic[0][1]*self.dim[1]
            pos2 = self.abs_pos[0]+schematic[1][0]*self.dim[0], self.abs_pos[1]+schematic[1][1]*self.dim[1]
            self.lines.append([pos1, pos2])

    def rebuild_texts(self):
        for t in self.texts:
            if t is not None:
                t.delete()
        self.texts.clear()
        for schematic in self.texts_schematics:
            txt = Text(schematic[0], schematic[1], schematic[2], schematic[3], schematic[4], self, color=schematic[5])
            self.texts.append(txt)

    def rebuild_shapes(self):
        self.rebuild_rects()
        self.rebuild_lines()
        self.rebuild_texts()

    def describe(self): # used in debugging
        print("I am ", self.name, "   hidden = ", self.hidden," \n abs_pos: ", self.abs_pos, "  dim: ", self.dim, "\t  percent pos: ", self.pos_pct, " percent dim: ", self.dim_pct)


class Text(Element):
    def __init__(self, position, anchor_id, max_dim_pct, font_size, name, parent, color=None, draw_background=False):
        pygame.font.init()
        if color is None:
            color = parent.color
        self.name = name
        # find true dimension and adjust font size if too large
        self.draw_background = draw_background
        self.f_sz = font_size
        self.max_dim_pct = max_dim_pct
        self.target_font_size = font_size
        self.font = pygame.font.SysFont(None, self.f_sz)
        self.txt_img = self.font.render(name, True, color) # 2nd arg is antialiasing
        while self.txt_img.get_width() > parent.dim[0]*max_dim_pct[0]:
            self.font = pygame.font.SysFont(None, max(10, self.f_sz))
            self.txt_img = self.font.render(name, True, color)
            self.f_sz -= 2
            if self.f_sz < 10:
                break
        if self.txt_img.get_width() > parent.dim[0]*max_dim_pct[0]: # not enough space: text becomes ".."
            name = ".."
            self.txt_img = self.font.render(name, True, color)
        actual_dim = (float(self.txt_img.get_width())/parent.dim[0],float(self.txt_img.get_height())/parent.dim[1])
        # find actual position of upper left corner (according to anchor point and dim)
        abs_upper_left = self.get_abs_upper_left(anchor_id,  parent.pct_to_px(position),  (self.txt_img.get_width(), self.txt_img.get_height()))
        pos_pct = ((abs_upper_left[0]-parent.abs_pos[0])/parent.dim[0], (abs_upper_left[1]-parent.abs_pos[1])/parent.dim[1])
        super(Text, self).__init__(position=pos_pct, dimension=actual_dim, name=name, parent=parent, uid=0, color=color)

    def update_colors(self, color=None, background_color=None):
        if color is None:
            self.color = self.parent.color
        else:
            self.color = color
        if background_color is None:
            self.background_color = self.parent.background_color
        else:
            self.background_color = background_color
        self.color_change(self.color)

    def delete(self):
        self.font = None
        self.txt_img = None

    def recompute_img(self):
        self.txt_img = self.font.render(self.name, True, self.color)

    def update(self, name, pos):
        if pos is not None:
            self.abs_pos = pos
        self.txt_img = self.font.render(name, True, self.color) # 2nd arg is antialiasing
        self.name = name
        while self.txt_img.get_width() > self.parent.dim[0]*self.max_dim_pct[0]:
            self.font = pygame.font.SysFont(None, max(10, self.f_sz))
            self.txt_img = self.font.render(name, True, self.color)
            self.f_sz -= 2
            if self.f_sz < 10:
                break
        if self.txt_img.get_width() > self.parent.dim[0]*self.max_dim_pct[0]: # not enough space: text becomes ".."
            name = ".."
            self.txt_img = self.font.render(name, True, self.color)


    def color_change(self, new_color):
        self.color = new_color
        self.txt_img = self.font.render(self.name, True, new_color)

    def get_abs_upper_left(self, anchor_id, abs_pos_at_anchor, abs_dim):
        if anchor_id == 1:
            return abs_pos_at_anchor
        elif anchor_id == 2:
            return (max(0, abs_pos_at_anchor[0]-0.5*abs_dim[0]), abs_pos_at_anchor[1])
        elif anchor_id == 3:
            return (max(0, abs_pos_at_anchor[0]-abs_dim[0]), abs_dim[1])
        elif anchor_id == 4:
            return (max(0, abs_pos_at_anchor[0]-abs_dim[0]), max(0, abs_pos_at_anchor[1]-0.5*abs_dim[1]))
        elif anchor_id == 5:
            return (max(0, abs_pos_at_anchor[0]-abs_dim[0]), max(0, abs_pos_at_anchor[1]-abs_dim[1]))
        elif anchor_id == 6:
            return (max(0, abs_pos_at_anchor[0]-0.5*abs_dim[0]), max(0, abs_pos_at_anchor[1]-abs_dim[1]))
        elif anchor_id == 7:
            return (abs_dim[0], max(0, abs_pos_at_anchor[1]-abs_dim[1]))
        elif anchor_id == 8:
            return (abs_dim[0], max(0, abs_pos_at_anchor[1]-0.5*abs_dim[1]))
        elif anchor_id == 9:
            return (max(0, abs_pos_at_anchor[0]-0.5*abs_dim[0]), max(0, abs_pos_at_anchor[1]-0.5*abs_dim[1]))
        else:
            print("wrong anchor point ID")
            return -1

    # no on_draw listener on text (nor any other listeners): the listeners for texts are done on the parent element
    def draw(self, screen): # return True if not a hidden element
        self.already_drawn = True
        if self.hidden:
            return False
        if self.draw_background:
            img_rect = self.txt_img.get_rect()
            img_rect.top  = self.abs_pos[1]
            img_rect.left = self.abs_pos[0]
            pygame.draw.rect(screen, self.background_color, img_rect, 0)
        screen.blit(self.txt_img, self.abs_pos)
        return True
