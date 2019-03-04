bl_info = {
    "name": "Face Anim Import",
    "category": "Object",
}

import bpy
import numpy
import pandas as pd
import sys
from scipy.iterpolate import interp1d

class FaceAnimImport(bpy.types.Operator):
	"""Imports iphone face animation from a csv"""
	bl_idname = "object.import_face_anim"
	bl_label = "Import Face Animation"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		df = pd.DataFrame()
		df = pd.read_csv("data.csv", index_col = None, header = 0)

		obj = bpy.context.object

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

		return {'FINISHED'}


def register():
	bpy.utils.register_class(FaceAnimImport)

def unregister():
	bpy.utils.unregister_class(FaceAnimImport)

if __name__ == "__main__":
	register()
