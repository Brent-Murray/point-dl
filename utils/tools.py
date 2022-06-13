import glob
import os
from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch_geometric.data import Data, InMemoryDataset


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)

    # Get coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        # for las_field in las_fields[3:]:  # skip the X,Y,Z fields
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)

class PointCloudsInFiles(InMemoryDataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(
        self, root_dir, glob="*", column_name="", max_points=200_000, use_columns=None
    ):
        """
        Args:
            root_dir (string): Directory with the datasets
            glob (string): Glob string passed to pathlib.Path.glob
            column_name (string): Column name to use as target variable (e.g. "Classification")
            use_columns (list[string]): Column names to add as additional input
        """
        self.files = list(Path(root_dir).glob(glob))
        self.column_name = column_name
        self.max_points = max_points
        if use_columns is None:
            use_columns = []
        self.use_columns = use_columns
        super().__init__()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = str(self.files[idx])
        coords, attrs = read_las(filename, get_attributes=True)
        if coords.shape[0] >= self.max_points:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=False)
        else:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=True)
        if len(self.use_columns) > 0:
            x = np.empty((self.max_points, len(self.use_columns)), np.float32)
            for eix, entry in enumerate(self.use_columns):
                x[:, eix] = attrs[entry][use_idx]
        else:
            x = coords[use_idx, :]
        coords = coords - np.mean(coords, axis=0)  # centralize coordinates

        # impute target
        target = attrs[self.column_name]
        target[np.isnan(target)] = np.nanmean(target)

        sample = Data(
            x=torch.from_numpy(x).float(),
            y=torch.from_numpy(
                np.unique(np.array(target[use_idx][:, np.newaxis]))
            ).type(torch.LongTensor),
            pos=torch.from_numpy(coords[use_idx, :]).float(),
        )
        if coords.shape[0] < 100:
            return None
        return sample


class IOStream:
    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush

    def close(self):
        sefl.f.close()
        
def _init_(model_name):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + model_name):
        os.makedirs("checkpoints/" + model_name)
    if not os.path.exists("checkpoints/" + model_name + "/models"):
        os.makedirs("checkpoints/" + model_name + "/models")
    if not os.path.exists("checkpoints/" + model_name + "/confusion_matrix"):
        os.makedirs("checkpoints/" + model_name + "/confusion_matrix")
    if not os.path.exists("checkpoints/" + model_name + "/confusion_matrix/all"):
        os.makedirs("checkpoints/" + model_name + "/confusion_matrix/all")
    if not os.path.exists("checkpoints/" + model_name + "/confusion_matrix/best"):
        os.makedirs("checkpoints/" + model_name + "/confusion_matrix/best")
    if not os.path.exists("checkpoints/" + model_name + "/classification_report"):
        os.makedirs("checkpoints/" + model_name + "/classification_report")
    if not os.path.exists("checkpoints/" + model_name + "/classification_report/all"):
        os.makedirs("checkpoints/" + model_name + "/classification_report/all")
    if not os.path.exists("checkpoints/" + model_name + "/classification_report/best"):
        os.makedirs("checkpoints/" + model_name + "/classification_report/best")
        

def make_confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    
    count:         If True, show the raw number in the confusion matrix. Default is True.
    
    percent:     If True, show the proportions for each category. Default is True.
    
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
                   
    xyticks:       If True, show x and y ticks. Default is True.
    
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''
    
    
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
        

def delete_files(root_dir, glob="*"):
    files = list(Path(root_dir).glob(glob))
    for f in files:
        os.remove(f)