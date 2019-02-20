import numpy as np
import os
import sys
import argparse
from create_file import *
from numpy import newaxis

#Trim audio based on start time and duration of audio. 
def trim_audio(input_audio_file,output_audio_file,start_time,duration):
	#print input_audio_file
	#print output_audio_file
	#print input_audio_file
	#print output_audio_file
	#print start_time
	#print duration
	#ffmpeg -i Y--aaILOrkII.m4a -ss 10 -t 10 -c copy -y Y--aaILOrkII_trimmed.m4a
	#cmdstring = "sox %s %s trim %s %s" %(input_audio_file,output_audio_file,start_time,duration)
	cmdstring = "ffmpeg -loglevel panic -i %s -ss %s -t %s -c copy -y %s" %(input_audio_file,start_time,duration,output_audio_file)
	#print cmdstring
	os.system(cmdstring)

def create_trimmed_file(audio_path,duration_needed,audio_info_file,duration_list_file,output_path,output_file):
		output_file_list = []
		f = open(duration_list_file,'r')
		dict = {}
		for line in f:
			k,v = line.strip().split(' ')
			k1 = k.split('.')[0][1:]
			dict[k1.strip()] = v.strip()
		f.close()

		with open(audio_info_file,'r') as f1:
			info_lines = f1.readlines()
		f1.close()

		if not os.path.exists(output_path):
			os.makedirs(output_path)

		for line in info_lines:
			line_split = line.split(', ')
			#print line_split
			file_name = line_split[0]
			start_time = float(line_split[1])
			end_time = float(line_split[2])
			duration = float(dict[file_name])	
			mid_time = (start_time + end_time)/2
			target_end_time = mid_time + (duration_needed/2)
			target_start_time = mid_time - (duration_needed/2)
			if target_end_time > duration:
				diff = target_end_time - duration
				target_end_time = duration
				target_start_time -= diff
				if target_start_time < 0.0:
					target_start_time = 0
			if target_start_time < 0.0:
				diff = -target_start_time
				target_start_time = 0
				target_end_time +=diff
				if target_end_time > duration:
					target_end_time = duration
			new_duration = target_end_time - target_start_time
			print line
			print "Mid time = " + str(mid_time) + " Start time = " + str(start_time) + " End time = " + str(end_time) + " target_start_time = " + str(target_start_time) + " target_end_time = " + str(target_end_time) + " Old duration = " + str(duration) + " New duration = " + str(new_duration)
			output_audio_trimmmed_filename = 'Y' + file_name + '_' + str(start_time) + '_' + str(end_time) + '_' + str(target_start_time) + '_' + str(target_end_time) + '.m4a ' + str(new_duration)
			output_audio_absolute_trimmmed_filename = output_path + '/Y' + file_name + '_' + str(start_time) + '_' + str(end_time) + '_' + str(target_start_time) + '_' + str(target_end_time) + '.m4a'
			input_audio_file = audio_path + '/Y' + file_name + '.m4a'		 
			trim_audio(input_audio_file,output_audio_absolute_trimmmed_filename,target_start_time,new_duration)
			output_file_list.append(output_audio_trimmmed_filename)
		create_file(output_file,output_file_list)






if __name__ == "__main__":
	if len(sys.argv) < 7:
		print "python create_files_X_seconds arg1 = audio_path arg2 = value of X arg3 = audio_information file (modified_output_downloaded) arg4 = duration list file arg5 = output path arg6 = output path file"
	else:
		parser = argparse.ArgumentParser()
		parser.add_argument('--input_audio_path', type=str, default='audio path', metavar='N',
        	            help='input_audio_path (default: audio path)')
		parser.add_argument('--duration_required', type=int, default='30', metavar='N',
        	            help='duration_required (default: 30)')
		parser.add_argument('--audio_info_file', type=str, default='balanced_modified_output.list', metavar='N',
        	            help='audio_info_file (default: balanced_modified_output.list)')
		parser.add_argument('--duration_list_file', type=str, default='duration_output.list', metavar='N',
        	            help='duration_list_file (default: duration_output.list)')
		parser.add_argument('--output_audio_path', type=str, default='output_audio_path', metavar='N',
        	            help='output_audio_path (default: output_audio_path)')
		parser.add_argument('--output_file_list', type=str, default='output_file_list', metavar='N',
        	            help='output_file_list (default: output_file_list)')
		args = parser.parse_args()
		duration_needed = args.duration_required
		create_trimmed_file(args.input_audio_path,duration_needed,args.audio_info_file,args.duration_list_file,args.output_audio_path,args.output_file_list)

#Sample command
#print "python create_files_X_seconds arg1 = audio_path arg2 = value of X arg3 = audio_information file (modified_output_downloaded) arg4 = duration list file arg5 = output path arg6 = output path file"
