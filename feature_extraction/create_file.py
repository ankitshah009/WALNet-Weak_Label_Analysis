import os
import subprocess

#This def will create a file list based on an array and dump the array contents in a file
def create_file(filename,array):
	if not os.path.exists(filename):
	        open(filename, 'w').close()
	with open(filename,'w') as file1:
		for item in array:
			file1.write("%s\n" % item)
	subprocess.call(['chmod','0777',filename])


