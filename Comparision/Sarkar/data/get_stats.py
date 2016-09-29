import os
import numpy as np
files = os.listdir("./raw_input/")
for file in files:
    filename = "./raw_input/" + file
    if filename == "./raw_input/__init__.py": continue
    import pandas as pd
    df = pd.read_csv(filename)
    headers = [h for h in df.columns if '$<' not in h]
    data = df[headers]
    R, C = np.shape(data)  # No. of Rows and Col
    str_pr =  "\"" + file + "\","
    # str_pr =  "\"" + file + "\" :[" + str(C) + "," +  str(R) + "],"
    print str_pr,