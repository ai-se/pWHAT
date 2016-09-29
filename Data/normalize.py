from __future__ import division
import pandas as pd
import os

files = ["./raw_input/" + f for f in os.listdir("./raw_input/") if ".csv" in f]
for file in files:
    filename = file.split("/")[-1]
    content = pd.read_csv(file)
    content = (content - content.min())/(content.max() - content.min())
    content.fillna(0, inplace=True)
    content.to_csv('./input/'+filename, index=False, float_format='%.3f')

