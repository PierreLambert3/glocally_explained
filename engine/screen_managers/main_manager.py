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
import Multi_BIOT
import BIOT
from scipy.linalg import norm
import pandas as pd

@njit
def thoughtfull_name(Xld, colours, orig_colours, score, score_min, score_max, span):
    for pt in range(Xld.shape[0]):
        # colours[pt] = orig_colours[pt] + (np.array([220., 220., 220.])-orig_colours[pt])*(1 - (score[pt] - score_min)/span)
        colours[pt] = orig_colours[pt] * ((score[pt] - score_min)/span)

class Node():
    def __init__(self, idxs, is_leaf, split_axis, name):
        self.idxs = idxs
        self.is_leaf = is_leaf
        self.split_axis = split_axis
        self.name = name


def split_embedding(Xhd, Xld, threshold, min_support):
    N, M = Xhd.shape
    idxs = np.arange(N)
    root = Node(idxs, is_leaf=False, split_axis=0, name=0)

    nodes = [root]
    kept_explanations = []
    i = 0
    while(nodes):
        node = nodes.pop()
        tmp_explanation = Local_explanation_wrapper(node.idxs, Xld, Xhd, method = 'biot')
        expl_err = np.mean(tmp_explanation.compute_errors(Xhd[node.idxs], Xld[node.idxs]))
        print(expl_err, '(thresh=',threshold,')')
        if node.idxs.shape[0] <= min_support*2+1 or expl_err < threshold:
            node.is_leaf = True
            yield tmp_explanation
            # kept_explanations.append(tmp_explanation)
        else:
            node.is_leaf = False
            cut = np.median(Xld[node.idxs, node.split_axis], axis=0)

            idx1 = node.idxs[np.where(Xld[node.idxs, node.split_axis] < cut)[0]]
            idx2 = node.idxs[np.where(Xld[node.idxs, node.split_axis] >= cut)[0]]

            n1 = Node(idx1, is_leaf=False, split_axis = 1 - node.split_axis, name=i)
            n2 = Node(idx2, is_leaf=False, split_axis = 1 - node.split_axis, name=i)


            nodes.append(n1)
            nodes.append(n2)
            i+=1
    # return kept_explanations


class Main_manager(Manager):
    def __init__(self, config, main_window, theme, uid_generator):
        super(Main_manager, self).__init__("main", initial_state=True)
        # self.type_identifier = "main manger"
        self.active  = True
        self.deleted = False
        self.theme = theme
        self.method = 'biot'

        self.main_window   = main_window
        self.main_window.awaiting_key_press.append(self)
        self.uid_generator = uid_generator
        self.screen        = Main_screen(theme, main_window, self) # builds the screen
        self.lock = threading.Lock()
        self.main_view = self.main_window.containers[0]

        self.an_explanation_is_selected = False

        self.scatterplot_redraw_listener = Listener(REDRAW_THINGS, [self])
        self.scatterplot = self.main_view.leaves[0]
        self.ax1_explanation = self.main_view.containers[0].leaves[0]
        self.ax2_explanation = self.main_view.containers[0].leaves[1]
        self.ctrl_pressed = False
        self.Xhd, self.Xld, self.KDtree_ld, self.Y, self.Y_colours = None,None,None,None,None
        self.has_dataset = False
        self.K_select = 150
        self.draw_grid = np.zeros((150, 150), dtype = int)
        self.selected_feature = None

    def run_multi_biot(self):
        # hardcodé pour le dataset countries
        data = pd.read_csv("datasets/country_dataset_with_names.csv", index_col = 0)
        X = data.copy()
        Y = self.Xld

        max_lam = BIOT.calc_max_lam(X, Y)
        n_lam = 10
        lam_values = max_lam*(10**np.linspace(-1, 0, num=n_lam, endpoint=True, retstep=False, dtype=None))
        lam_list = lam_values.tolist()

        N = self.Xhd.shape[0]
        initial_clusters = np.ones(N, dtype=int)
        N_explanations = len(self.scatterplot.local_explanations)
        for i in range(N_explanations):
            expl = self.scatterplot.local_explanations[i]
            initial_clusters[expl.sample_idx] = i

        Yhat, W_list, w0_list, R_list, clusters = Multi_BIOT.CV_Multi_BIOT (X_train = X, X_test = X, Y_train = Y, lam_list = lam_list, K_list = None, num_folds=4, clusters = initial_clusters, rotation = True)

        while self.scatterplot.local_explanations:
            self.scatterplot.delete_explanation_number(0)

        clusters = np.array(clusters)
        N_expl = np.unique(clusters).shape[0]
        for new_expla_i in range(N_expl):
            indices = np.where(clusters == new_expla_i)[0]
            self.scatterplot.add_explanation(Local_explanation_wrapper(indices, self.Xld, self.Xhd, method='biot'))


    def receive_dataset(self, Xhd, Xld, Y, Y_colours, Y_colours_expl, feature_names=None):
        self.Xhd = Xhd
        self.Xld = Xld
        self.KDtree_ld = KDTree(Xld, leaf_size=2)
        self.Y = Y
        self.has_dataset = True
        feature_colours1 = np.tile(np.array([0., 149., 0]), Xhd.shape[1]).reshape((-1, 3))
        feature_colours2 = np.tile(np.array([149., 11., 14.]), Xhd.shape[1]).reshape((-1, 3))
        if feature_names is None:
            feature_names = ["variable "+str(i) for i in range(Xhd.shape[1])]
        self.scatterplot.set_points(Xld, Y_colours, Y_colours_expl, feature_names)
        self.ax1_explanation.receive_features(feature_names, feature_colours1)
        self.ax2_explanation.receive_features(feature_names, feature_colours2)
        self.feature_names = feature_names

    def new_explanation(self, neighbours_idx):
        explanation = Local_explanation_wrapper(neighbours_idx, self.Xld, self.Xhd, method=self.method)
        self.scatterplot.add_explanation(explanation)

    def explain_full_dataset_splitting(self, threshold, min_support):
        for explanation in split_embedding(self.Xhd, self.Xld, threshold, min_support):
            self.scatterplot.add_explanation(explanation)
            self.redraw_things()

    def explain_full_dataset_Kmeans(self, threshold, min_support, K):
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=K, init='random', n_init=500, random_state=0).fit(self.Xld)
        centers = model.cluster_centers_
        labels = model.labels_
        for i in range(centers.shape[0]):
            print(np.round(float(i/centers.shape[0]),2))
            idxs = np.where(labels == i)[0]
            Nidx = idxs.shape[0]
            if Nidx > min_support:
                explanation = Local_explanation_wrapper(idxs, self.Xld, self.Xhd, method = 'biot')
                self.scatterplot.add_explanation(explanation)

    def merge_try(self, mid, mid_HD, R_3, W_3, w0_3, sample3):
        new_explanation = Local_explanation_wrapper(sample3, self.Xld, self.Xhd, method = 'biot', fit=False)
        new_explanation.model.center_LD = mid
        new_explanation.model.center_HD = mid_HD
        new_explanation.model.axis2d = R_3
        new_explanation.model.features_coeffs_ax1 = W_3[:,0]
        new_explanation.model.features_coeffs_ax2 = W_3[:,1]
        new_explanation.model.W = W_3
        new_explanation.model.w0 = w0_3
        new_explanation.model.R = R_3
        new_explanation.sample_idx = sample3
        new_explanation.axis2d = new_explanation.model.axis2d


        centrd = self.Xhd[sample3] - new_explanation.model.center_HD
        Yhat_new = mid + ((np.tile(w0_3, (centrd.shape[0], 1)) + (centrd @ W_3)) @ R_3.T)
        err = np.mean(np.sqrt(np.mean((Yhat_new - self.Xld[sample3])**2, axis=1)))
        return new_explanation, err

    def merge_explanations(self, threshold):

        something_happened = True
        while(something_happened):
            something_happened = False

            merge1 = -1
            merge2 = -1
            new_expl = None

            explanations = self.scatterplot.local_explanations
            for i, expl_wrapper1 in enumerate(explanations):
                expl1 = expl_wrapper1.model
                center1 = expl1.center_LD
                center_HD1 = expl1.center_HD
                R_1  = expl1.R
                W_1  = expl1.W
                w0_1 = expl1.w0
                sample1 = expl_wrapper1.sample_idx

                for u in range(i+1, len(explanations)):

                    expl2 = explanations[u].model
                    center2 = expl2.center_LD
                    center_HD2 = expl2.center_HD
                    R_2  = expl2.R
                    W_2  = expl2.W
                    w0_2 = expl2.w0
                    sample2 = explanations[u].sample_idx

                    mid = 0.5 * (center1 + center2)
                    mid_HD = 0.5 * (center_HD1 + center_HD2)
                    R_3  = 0.5 * (R_1 + R_2)
                    W_3  = 0.5 * (W_1 + W_2)
                    w0_3  = 0.5 * (w0_1 + w0_2)
                    sample3 = np.hstack((sample1, sample2))

                    new_expl_tmp, new_error = self.merge_try(mid, mid_HD, R_3, W_3, w0_3, sample3)
                    print('err, threshold ', new_error, threshold)
                    if new_error < threshold:
                        something_happened = True
                        merge1 = i
                        merge2 = u
                        new_expl = new_expl_tmp
                        break
                if something_happened:
                    break

            if something_happened:
                if i == u:
                    1/0
                self.scatterplot.delete_explanation_number(max(i, u))
                self.scatterplot.delete_explanation_number(min(i, u))
                self.scatterplot.add_explanation(new_expl)
                self.redraw_things()
                print("\n MERGED \n")



    def last_biot(self):
        explanations = self.scatterplot.local_explanations
        for i, expl_wrapper in enumerate(explanations):
            expl_wrapper.model.fit(self.Xld[expl_wrapper.sample_idx], self.Xhd[expl_wrapper.sample_idx])
            expl_wrapper.axis2d = expl_wrapper.model.axis2d
        self.redraw_things()



    def explain_full_dataset(self, partition_method, threshold=10., min_support=10, Kmeans_K=10, multi_biot=True):
        if partition_method == "kmeans":
            self.explain_full_dataset_Kmeans(threshold=threshold, min_support=min_support, K=Kmeans_K)
        else:
            self.explain_full_dataset_splitting(threshold=threshold, min_support=min_support)
            self.merge_explanations(threshold=threshold)  # meh
        self.redraw_things()

        if multi_biot:
            print("running multi biot...")
            self.run_multi_biot()
            print("DONE")
            self.redraw_things()



    def select_explanation(self, explanation_idx):
        if explanation_idx == -1:
            return
        self.an_explanation_is_selected = True
        self.scatterplot.selected_explanation = explanation_idx
        explanation = self.scatterplot.local_explanations[explanation_idx]
        errors = explanation.compute_errors(self.Xhd, self.Xld)
        goodness = -errors
        score_min, score_max = np.min(goodness), np.max(goodness)
        span = score_max - score_min

        colours      = self.scatterplot.Y_colours
        orig_colours = self.scatterplot.orig_Y_colours
        thoughtfull_name(self.Xld, colours, orig_colours, goodness, score_min, score_max, span)

        self.ax1_explanation.clear()
        self.ax1_explanation.receive_explanation(explanation, 0)
        self.ax2_explanation.clear()
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
                pass
            else:
                self.scatterplot.delete_explanation(pos_in_LD)

    def wake_up(self, prev_manager):
        super(Main_manager, self).wake_up(prev_manager)

    def select_feature(self, feature_name, left=True):
        self.ax1_explanation.select_feature(feature_name, left_click=True)
        self.ax2_explanation.select_feature(feature_name, left_click=True)
        self.scatterplot.selected_feature = feature_name

    def unselect_feature(self, left=True):
        self.ax1_explanation.unselect_features(left_click=True)
        self.ax2_explanation.unselect_features(left_click=True)
        self.scatterplot.selected_feature = None

    def get_notified(self, event_class, id, value, to_redraw = []):
        with self.lock:
            if event_class == REDRAW_THINGS:
                self.redraw_things()
            elif event_class == CTRL_KEY_CHANGE:
                is_pressed, pos = value
                self.ctrl_pressed = is_pressed
                self.an_explanation_is_selected = False
                self.scatterplot.selected_explanation = -1
                self.scatterplot.Y_colours = self.scatterplot.orig_Y_colours.copy()
                self.redraw_things()

            elif event_class == SCATTERPLOT_LEFT_CLICK:
                self.unselect_feature()
                self.click(value[0], left_click=True)
                self.redraw_things()
            elif event_class == SCATTERPLOT_RIGHT_CLICK:
                self.unselect_feature()
                self.click(value[0], left_click=False)
                self.redraw_things()
            elif event_class == SCATTERPLOT_HOVER:
                return
            elif event_class == SCATTERPLOT_DRAWING_DONE:
                neighbours = value
                self.new_explanation(neighbours)
            elif event_class == AXIS_LEFT_CLICK:
                var_name = value[0]
                if var_name is None:
                    self.unselect_feature()
                    return
                print(var_name)
                self.select_feature(var_name, left = True)
                self.redraw_things()
            elif event_class == AXIS_RIGHT_CLICK:
                var_name = value[0]
                if var_name is None:
                    self.unselect_feature()
                    return
                print(var_name)
                self.select_feature(var_name, left = False)
                self.redraw_things()
            else:
                print("unrecognised event received by main_manager: ", event_class, "  (see correspondance in event_ids.py)")



    def on_awaited_key_press(self, to_redraw, pressed_keys, pressed_special_keys):
        if pressed_keys[0] == 'o':
            print("pressed o")
        return False

    def on_awaited_mouse_release(self, to_redraw, release_pos, released_buttons, pressed_special_keys):
        1/0
        return True


    def redraw_things(self):
        self.ask_redraw(self.main_view)
