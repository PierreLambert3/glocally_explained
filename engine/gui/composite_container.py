from engine.container import Container
from engine.element   import Text
from engine.selector  import Button
from engine.event_ids import BUTTON_CLICKED


class Container_with_tabs(Container):
    def __init__(self, position, dimension, name, parent, uid_generator, show_border=False, filled=True, color=None, background_color=None, border_size=1,\
    	listening_hover=False, listening_Lmouse=False, listening_Rmouse=False, listening_scroll=False):
        super(Container_with_tabs, self).__init__(position=position, dimension=dimension, name=name, parent=parent, uid_generator=uid_generator,\
            color=color, background_color=background_color,listening_hover=listening_hover, listening_Lmouse=listening_Lmouse, listening_Rmouse=listening_Rmouse, listening_scroll=listening_scroll)
        self.contents = {} # tab_name: Container
        self.current_tab_name = None
        self.current_content  = None
        