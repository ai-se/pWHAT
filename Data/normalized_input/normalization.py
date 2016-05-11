import os
import pandas as pd
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
filenames = os.listdir("../input/")

def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        print col
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
        import pdb
        pdb.set_trace()
    return df

for filename in filenames:
    if filename == "__init__.py" or filename == ".DS_Store" or filename == "mo_TriMesh.csv" or filename == "mo_x264-DB.csv": continue
    print filename
    df = pd.read_csv("../input/" + filename)
    headers = [h for h in df.columns if '$<' not in h]
    dependent_name = [h for h in df.columns if '$<' in h]

    indep_df = df[headers]
    norm_indep_df = (indep_df - indep_df.min()) / (indep_df.max() - indep_df.min() + 0.00001)
    norm_indep_df[dependent_name[-1]] = df[dependent_name[-1]]

    norm_indep_df.to_csv("normalized_" + filename, header=list(df.columns.values), index=False)

