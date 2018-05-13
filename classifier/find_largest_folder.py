import sys
import os
import numpy as np

def find_largest_folder(folder_name):
	run_info = folder_name.split('_',1)[-1]
	list_populate = []
	for i in range(60):
		list_populate.append(np.load(folder_name + '/metrics_validation_'+ str(run_info) + '_' + str(i) + '_aps.txt.npy')[-1] )
	print list_populate
	print max(list_populate)
	print list_populate.index(max(list_populate))


if __name__=="__main__":
	find_largest_folder(sys.argv[1])
