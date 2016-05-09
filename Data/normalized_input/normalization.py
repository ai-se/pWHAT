import os
import pandas as pd
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
filenames = os.listdir("../input/")

def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df

for filename in filenames:
    if filename == "__init__.py" or filename == ".DS_Store": continue
    print filename
    df = pd.read_csv("../input/" + filename)
    headers = [h for h in df.columns if '$<' not in h]

    scaled_df = scaleColumns(df,headers)


    scaled_df.to_csv("normalized_" + filename, header=list(df.columns.values), index=False)

