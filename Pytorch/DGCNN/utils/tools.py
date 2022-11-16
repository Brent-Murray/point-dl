import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
from itertools import cycle, islice

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from plyer import notification
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


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


def farthest_point_sampling(coords, k):
    # Adapted from https://minibatchai.com/sampling/2021/08/07/FPS.html

    # Get points into numpy array
    points = np.array(coords)

    # Get points index values
    idx = np.arange(len(coords))

    # Initialize use_idx
    use_idx = np.zeros(k, dtype="int")

    # Initialize dists
    dists = np.ones_like(idx) * float("inf")

    # Select a point from its index
    selected = 0
    use_idx[0] = idx[selected]

    # Delete Selected
    idx = np.delete(idx, selected)

    # Iteratively select points for a maximum of k samples
    for i in range(1, k):
        # Find distance to last added point and all others
        last_added = use_idx[i - 1]  # get last added point
        dist_to_last_added_point = ((points[last_added] - points[idx]) ** 2).sum(-1)

        # Update dists
        dists[idx] = np.minimum(dist_to_last_added_point, dists[idx])

        # Select point with largest distance
        selected = np.argmax(dists[idx])
        use_idx[i] = idx[selected]

        # Update idx
        idx = np.delete(idx, selected)
    return use_idx


class PointCloudsInPickle(Dataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(
        self,
        filepath,
        pickle,
        # column_name="",
        # max_points=200_000,
        # samp_meth=None,  # one of None, "fps", or "random"
        # use_columns=None,
    ):
        """
        Args:
            pickle (string): Path to pickle dataframe
            column_name (string): Column name to use as target variable (e.g. "Classification")
            use_columns (list[string]): Column names to add as additional input
        """
        self.filepath = filepath
        self.pickle = pd.read_pickle(pickle)
        # self.column_name = column_name
        # self.max_points = max_points
        # if use_columns is None:
        #     use_columns = []
        # self.use_columns = use_columns
        # self.samp_meth = samp_meth
        super().__init__()

    def __len__(self):
        return len(self.pickle)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file name
        pickle_idx = self.pickle.iloc[idx : idx + 1]
        filename = pickle_idx["FileName"].item()

        # Get file path
        file = os.path.join(self.filepath, filename)

        # Read las/laz file
        coords = read_las(file, get_attributes=False)

        # # Resample number of points to max_points
        # if self.samp_meth is None:
        #     use_idx = np.arange(len(coords))
        # else:
        #     if coords.shape[0] >= self.max_points:
        #         if self.samp_meth == "random":
        #             use_idx = np.random.choice(
        #                 coords.shape[0], self.max_points, replace=False
        #             )
        #         if self.samp_meth == "fps":
        #             use_idx = farthest_point_sampling(coords, self.max_points)
        #     else:
        #         use_idx = np.random.choice(
        #             coords.shape[0], self.max_points, replace=True
        #         )

        # # Get x values
        # if len(self.use_columns) > 0:
        #     x = np.empty((self.max_points, len(self.use_columns)), np.float32)
        #     for eix, entry in enumerate(self.use_columns):
        #         x[:, eix] = attrs[entry][use_idx]
        # else:
        #     # x = coords[use_idx, :]
        #     x = None
        coords = coords - np.mean(coords, axis=0)  # centralize coordinates

        # impute target
        target = pickle_idx["perc_specs"].item()
        target = [float(i) for i in target]  # convert items in target to float

        # if x is None:
        #     sample = Data(
        #         x=None,
        #         y=torch.from_numpy(np.array(target)).type(torch.FloatTensor),
        #         pos=torch.from_numpy(coords[use_idx, :]).float(),
        #     )
        # else:
        #     sample = Data(
        #         x=torch.from_numpy(x).float(),
        #         y=torch.from_numpy(np.array(target)).type(torch.FloatTensor),
        #         pos=torch.from_numpy(coords[use_idx, :]).float(),
        #     )
        coords = torch.from_numpy(coords).float()
        target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
        # target = torch.from_numpy(np.array(target)).half()
        if coords.shape[0] < 100:
            return None
        return coords, target


class IOStream:
    # Adapted from https://github.com/vinits5/learning3d/blob/master/examples/train_pointnet.py
    def __init__(self, path):
        # Open file in append
        self.f = open(path, "a")

    def cprint(self, text):
        # Print and write text to file
        print(text)  # print text
        self.f.write(text + "\n")  # write text and new line
        self.f.flush  # flush file

    def close(self):
        self.f.close()  # close file


def _init_(model_name):
    # Create folder structure
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + model_name):
        os.makedirs("checkpoints/" + model_name)
    if not os.path.exists("checkpoints/" + model_name + "/models"):
        os.makedirs("checkpoints/" + model_name + "/models")
    # if not os.path.exists("checkpoints/" + model_name + "/confusion_matrix"):
        # os.makedirs("checkpoints/" + model_name + "/confusion_matrix")
    # if not os.path.exists("checkpoints/" + model_name + "/confusion_matrix/all"):
    #     os.makedirs("checkpoints/" + model_name + "/confusion_matrix/all")
    # if not os.path.exists("checkpoints/" + model_name + "/confusion_matrix/best"):
    #     os.makedirs("checkpoints/" + model_name + "/confusion_matrix/best")
    # if not os.path.exists("checkpoints/" + model_name + "/classification_report"):
    #     os.makedirs("checkpoints/" + model_name + "/classification_report")
    # if not os.path.exists("checkpoints/" + model_name + "/classification_report/all"):
    #     os.makedirs("checkpoints/" + model_name + "/classification_report/all")
    # if not os.path.exists("checkpoints/" + model_name + "/classification_report/best"):
    #     os.makedirs("checkpoints/" + model_name + "/classification_report/best")
    if not os.path.exists("checkpoints/" + model_name + "/output"):
        os.makedirs("checkpoints/" + model_name + "/output")



def delete_files(root_dir, glob="*"):
    # List files in root_dir with glob
    files = list(Path(root_dir).glob(glob))

    # Delete files
    for f in files:
        os.remove(f)


def plot_3d(coords, save_plot=False, fig_path=None, dpi=300):
    # Plot parameters
    plt.rcParams["figure.figsize"] = [7.00, 7.00]  # figure size
    plt.rcParams["figure.autolayout"] = True  # auto layout

    # Create figure
    fig = plt.figure()  # initialize figure
    ax = fig.add_subplot(111, projection="3d")  # 3d projection
    x = coords[:, 0]  # x coordinates
    y = coords[:, 1]  # y coordinates
    z = coords[:, 2]  # z coordinates
    ax.scatter(x, y, z, c=z, alpha=1)  # create a scatter plot
    plt.show()  # show plot

    if save_plot is True:
        plt.savefig(fig_path, bbox_inches="tight", dpi=dpi)
    
    
def check_multi_gpu():
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        multi_gpu = True
        print("Using Multiple GPUs")
    else:
        multi_gpu = False
        
    return multi_gpu


def create_empty_df():
    # Create an empty dataframe with specific dtype
    df = pd.DataFrame(
        {
            "Model": pd.Series(dtype="str"),
            "Point Density": pd.Series(dtype="str"),
            "Overall Accuracy": pd.Series(dtype="float"),
            "F1": pd.Series(dtype="float"),
            "Augmentation": pd.Series(dtype="str"),
            "Sampling Method": pd.Series(dtype="str"),
        }
    )

    return df


def variable_df(variables, col_names):
    # Create a dataframe from list of variables
    df = pd.DataFrame([variables], columns=col_names)
    
    return df


def concat_df(df_list):
    # Concate multiple dataframes in list
    df = pd.concat(df_list, ignore_index=True)
    return df


def notifi(title, message):
    # Creates a pop-up notification
    notification.notify(title=title, message=message, timeout=10)
    
    
def create_comp_csv(y_true, y_pred, classes, filepath):
    # Create a CSV of the true and predicted species proportions
    classes = cycle(classes) # cycle classes
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}) # create dataframe
    df["SpeciesName"] = list(islice(classes, len(df))) # repeat list of classes
    species = df.pop("SpeciesName") # remove species name
    df.insert(0, "SpeciesName", species) # add species name
    df.to_csv(filepath, index=False) # save to csv
    
    
def get_stats(df):
    # Get stats
    r2 = r2_score(df["y_true"], df["y_pred"])  # r2
    rmse = np.sqrt(mean_squared_error(df["y_true"], df["y_pred"]))  # rmse
    df_max = np.max(df["Difference"])  # max difference
    df_min = np.min(df["Difference"])  # min difference
    df_std = np.std(df["Difference"])  # std difference
    df_var = np.var(df["Difference"])  # var difference
    df_count = np.count_nonzero(df["y_true"])  # count of none 0 y_true

    return pd.Series(
        dict(
            r2=r2,
            rmse=rmse,
            maximum=df_max,
            minimum=df_min,
            std=df_std,
            var=df_var,
            count=df_count,
        )
    )


def get_df_stats(csv, min_dif, max_dif):
    # Create dataframe
    df = pd.read_csv(csv)  # read in csv
    df["Difference"] = df["y_pred"] - df["y_true"]  # difference of y_pred and y_true
    df["Sum"] = df["y_pred"] + df["y_true"]  # sum of y_pred and y_true
    df["Between"] = df["Difference"].between(min_dif, max_dif)  # boolean of Difference

    # Print number of True and False values of min and max difference values
    print("Count")
    print(df.groupby("Between")["Difference"].count())
    print()

    # Calculate and print get_stats fpr df
    print("Stats")
    df_stats = df.groupby("SpeciesName").apply(get_stats)
    print(df_stats)
    print("#################################################")
    print()

    return df_stats


def scatter_plot(csv, point_density, root_dir):
    df = pd.read_csv(csv)
    species = df.SpeciesName.unique()
    for i in species:
        species_csv = df[df.SpeciesName == i]
        species_csv.plot.scatter(x="y_pred", y="y_true")
        plt.title(f"{i}: {point_density}")
        plt.savefig(os.path.join(root_dir, f"{i}_{point_density}.png"))
        plt.close()


def plot_stats(
    root_dir,
    point_densities,
    model,
    stats=["r2", "rmse"],
    save_csv=False,
    csv_name=None,
    save_fig=False,
):
    # Set Plot Parameters
    plt.rcParams["figure.figsize"] = [15.00, 7.00]  # figure size
    plt.rcParams["figure.autolayout"] = True  # auto layout
    plt.rcParams["figure.facecolor"] = "white"  # facecolor

    dfs_r2 = []
    dfs_rmse = []
    for x in point_densities:
        # Print point density
        print(f"Point Density: {str(x)}")

        # Get root directory
        model_output = os.path.join(root_dir, f"{model}_{x}\output")

        # Get CSVs
        csv = list(Path(model_output).glob("outputs*.csv"))

        # Iterate through CSVs
        for y in csv:
            # Create scatter plots
            scatter_plot(y, x, model_output)

            # Calculate stats
            csv_stats = get_df_stats(y, -0.05, 0.05)

            # Save csv
            if save_csv is True:
                csv_stats.to_csv(
                    os.path.join(model_output, f"{model}_{x}_{csv_name}"), index=False
                )

        for stat in stats:
            # Convert to dataframe
            csv_item = csv_stats[stat].to_frame()

            # Rename column to point denisty
            csv_item.rename({stat: x}, axis=1, inplace=True)

            # Append dfs list
            if stat == "r2":
                dfs_r2.append(csv_item)
            if stat == "rmse":
                dfs_rmse.append(csv_item)

    # Concatenate dataframes
    df_r2 = pd.concat(dfs_r2, axis=1)
    df_rmse = pd.concat(dfs_rmse, axis=1)

    # Create Bar Chart for r2
    df_r2.plot.bar(width=0.9, edgecolor="black")
    plt.ylabel("r2")
    plt.grid(color="grey", linestyle="--", linewidth=0.1)
    plt.legend(title="Point Density")
    plt.tight_layout()

    # Save Figure
    if save_fig is True:
        plt.savefig(os.path.join(root_dir, f"{model}_r2.png"))
    plt.close()

    # Create Bar Chart for rmse
    df_rmse.plot.bar(width=0.9, edgecolor="black")
    plt.ylabel("rmse")
    plt.grid(color="grey", linestyle="--", linewidth=0.1)
    plt.legend(title="Point Density")
    plt.tight_layout()

    # Save Figure
    if save_fig is True:
        plt.savefig(os.path.join(root_dir, f"{model}_rmse.png"))
    plt.close()
    
# Model arguments
def model_args():
    # Set Up Parser
    parser = argparse.ArgumetParser(
        description="Point Cloud Species Proportion Estimation"
    )

    # Parser Arguments
    parser.add_argument(  # experiment name
        "--exp_name",
        type=str,
        default="exp",
        metavar="N",
        help="Name of the experiment",
    )

    parser.add_argument(  # model name
        "--model", type=str, default="dgcnn", metavar="N", choices=["dgcnn"]
    )

    parser.add_argument(  # batch size
        "--batch_size",
        type=int,
        default=32,
        metavar="batch_size",
        help="Size of Batch (default: 32)",
    )

    parser.add_argument(  # epochs
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="Number of Training Epochs (default: 300)",
    )

    parser.add_argument(  # optimizer
        "--optimizer", type=str, default="adam", metavar="N", choices=["adam", "sgd"]
    )

    parser.add_argument(  # learning rate
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning Rate (default: 0.001)",
    )

    parser.add_argument(  # sgd momentum
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD Momentum (default: 0.9)",
    )

    parser.add_argument(  # number of points
        "--num_points", type=int, default=7168, help="Number of Points (default: 7168)"
    )

    parser.add_argument(  # dropout
        "--dropout", type=float, default=0.5, help="Dropout Rate (default: 0.5)"
    )

    parser.add_argument(  # dimension of embeddings
        "--emb_dims",
        type=int,
        default=1024,
        metavar="N",
        help="Dimension of Embeddings (default: 1024)",
    )

    parser.add_argument(  # k nearest points
        "--k",
        type=int,
        default=20,
        metavar="N",
        help="Number of Nearest Points to Use (default: 20)",
    )

    parser.add_argument(
        "--model_path", type=str, default="", metavar="N", help="Pretrained Model Path"
    )
    
    parser.add_argument("--classes", nargs="+", help="Classes")
    
    parser.add_argument(
        "--adaptive_lr", # use adpative learning rate
        type=bool, 
        default=True, 
        help="Use Adaptive Learning Rate (default: True)"
    )
    
    parser.add_argument( # patience
        "--patience",
        type=int,
        default=20,
        help="Patience for Adaptive Learning Rate (default: 20)"
    )
    
    parser.add_argument( # stepsize
        "--step_size",
        type=int,
        default=30,
        help="Step Size for Adaptive Learning Rate (default: 30)"
    )

    args = parser.parse_args()
    
    return args