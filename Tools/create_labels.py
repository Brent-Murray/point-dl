import glob
import os
from pathlib import Path

import pandas as pd


def create_labels_df(
    root_dir, in_csv, out_filename, glob="*", group="PlotName", list_col="perc_spec",
):
    # Create list of files in root_dir with glob
    files = list(Path(root_dir).glob(glob))

    # Create Pandas Dataframe of in_csv
    df = (
        pd.read_csv(in_csv)  # read csv
        .groupby(group)[list_col]  # indexed group by
        .apply(list)  # create list
        .to_frame()  # convert to dataframe
        .reset_index(level=0)  # reset index
    )

    # Empty lists
    filenames = []
    perc_specs = []
    for file in files:
        filename = str(file).split("\\")[-1]  # get filename
        filenames.append(str(filename))  # append filenames
        plot = filename.split("_")[3]  # get plot number
        df1 = df.loc[df[group] == int(plot)]  # find plot nuber in df
        try:
            ps = df1[list_col].item()  # get list_col value
        except:
            ps = None
        perc_specs.append(ps)  # append perc_specs

    # Create Data Frame
    out_df = pd.DataFrame(
        list(zip(filenames, perc_specs)), columns=["FileName", "perc_specs"]
    )

    # Save Data Frame
    out_df.to_csv(os.path.join(root_dir, out_filename + ".csv"), index=False) # as csv

    out_df.to_pickle(os.path.join(root_dir, out_filename + ".pkl")) # as pickle