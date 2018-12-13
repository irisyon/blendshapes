bl_info = {
    "name": "Move X Axis",
    "category": "Object",
}

import bpy
import numpy
import pandas as pd
import sys
from scipy.iterpolate import interp1d


df = pd.DataFrame()
df = pd.read_csv("data.csv", index_col = None, header = 0)

f = open("blendshape-test-H.anim", "w")
f.write("animVersion 1.1;\n")
f.write("mayaVersion 2018;\n")
f.write("timeUnit film;\n")
f.write("linearUnit cm;\n")
f.write("angularUnit deg;\n")
f.write("startTime 1;\n")
f.write("endTime " + str(df.shape[0]) + ";\n")

time_points = np.arange(0, df.shape[0] - 1, 2.5)
index_i = 0
for column in df:
    f.write("anim " + column + " " + column + " face 0 0 " + str(index_i) + ";\n")
    f.write("animData {\ninput time;\noutput unitless;\nweighted 0;\n")
    f.write("preInfinity constant;\npostInfinity constant;\n")
    f.write("keys {\n")
    #seconds = len(df[column] - 1) / 60
    
    interpolated = interp1d(np.arange(0, len(df[column])), df[column].values)
    
    j = 0
    for i in time_points:
        key_value = interpolated(time_points[j])
        f.write(str(j) + " " + '{:f}'.format(key_value) + " auto auto 1 0 0;\n")
        j += 1
    f.write("}\n")
    f.write("}\n")
    index_i += 1
