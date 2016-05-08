import os
import numpy as np
files = os.listdir("./input/")
for file in files:
    filename = "./input/" + file
    if filename == "./input/__init__.py": continue
    import pandas as pd
    df = pd.read_csv(filename)
    headers = [h for h in df.columns if '$<' not in h]
    data = df[headers]
    R, C = np.shape(data)  # No. of Rows and Col
    str_pr =  "\"" + file + "\","
    # str_pr =  "\"" + file + "\" :[" + str(C) + "," +  str(R) + "],"
    print str_pr,