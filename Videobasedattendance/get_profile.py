import pandas as pd
data = pd.read_csv("./labels.csv")


def getprofile(id):
    profile=None
    for ind,rows in data.iterrows():
        if (rows[0]==id):
            profile=rows[1]

    return profile


