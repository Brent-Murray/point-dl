import glob
import os
import random
import shutil
from pathlib import Path

import pandas as pd


def train_dirs(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(os.path.join(folder, "train")):
        os.mkdir(os.path.join(folder, "train"))
    if not os.path.exists(os.path.join(folder, "val")):
        os.mkdir(os.path.join(folder, "val"))
    if not os.path.exists(os.path.join(folder, "test")):
        os.mkdir(os.path.join(folder, "test"))
        
        
def data_prep(
    root_dir,
    in_csv,
    out_folder,
    out_filename,
    glob="*",
    group="PlotName",
    list_col="perc_spec",
):
    # Create List of Files in root_dir with glob
    files = list(Path(root_dir).glob(glob))

    # Create Random List of Training Samples
    numfiles = len(files)  # get length of files list
    n_train = round(numfiles * 0.7)  # get 70% of length of list
    train_set = random.sample(files, n_train)  # randomly sample 70% of files

    # Create Training Directorys
    train_dirs(out_folder)

    # Move Files
    for i in files:
        if i in train_set:  # see if file is in train_set
            shutil.copy(
                i, os.path.join(out_folder, "train")
            )  # move files to training folder
        else:
            shutil.copy(
                i, os.path.join(out_folder, "val")
            )  # move files to validation folder

    # Create Pandas Dataframe of in_csv
    df = (
        pd.read_csv(in_csv)  # read csv
        .groupby(group)[list_col]  # indexed group by
        .apply(list)  # create list
        .to_frame()  # convert to dataframe
        .reset_index(level=0)  # reset index
    )

    # Create csv and pickle of vectorized species composition for each file
    for x in ["train", "val"]:  # iterate through training and validation
        # Empty lists
        filepaths = []
        perc_specs = []

        # Get List of Files
        files = list(Path(root_dir).glob(glob))
        files = list(Path(os.path.join(out_folder, x)).glob(glob))
        # files = list(Path(os.path.join(out_folder, x).glob(glob)))

        # Iterate Through Files
        for file in files:
            filepaths.append(str(file))  # append filepaths
            filename = str(file).split("\\")[-1]  # get file name
            plot = filename.split("_")[3]  # get plot number
            df1 = df.loc[df[group] == int(plot)]  # find plot number in df
            try:
                ps = df1[list_col].item()  # get list_col value
            except:
                ps = None
            perc_specs.append(ps)  # append perc_specs
        # Create Data Frame
        out_df = pd.DataFrame(
            list(zip(filepaths, perc_specs)), columns=["FilePath", "perc_specs"]
        )

        # Save Data Frame
        out_df.to_csv(
            os.path.join(out_folder, x, out_filename + ".csv"), index=False
        )  # as csv
        out_df.to_pickle(
            os.path.join(out_folder, x, out_filename + ".pkl")
        )  # as pickle