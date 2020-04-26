import numpy as np
import scipy.signal
import sys
import re
import os
import subprocess
import librosa
from create_file import *
from tqdm import tqdm
import argparse

sampling_freq = 44100
window_length = 1024
hop_length = 512
power = 2
num_mels = 128

# A list of segmented spectrograms features file is dumped in a list at the end of processList function
def processList(inlist):
    subprocess.Popen(['ffmpeg'])
    #File containing list of input wav files to process and extract features    
    inls = open(inlist)

    infls = inls.readlines()
    out_file_list = []
    for i,infl in tqdm(enumerate(infls)):
        infl_new = audio_root + '/' + infl.replace("\ "," ")
        if not os.path.isfile(os.path.expanduser(infl_new.strip())):
                print(os.path.expanduser(infl_new.strip()))
                print (infl + ' Not Found')
        else:
            #Fetch the audio samples at the sampling rate.
            y,sr = librosa.load(infl_new.strip(),sr=None)
            if len(y.shape) > 1:
                       print ('Mono Conversion')
                       y = librosa.to_mono(y)
            if sr != sampling_freq:
                       print ('Resampling {}'.format(sr))
                       y = librosa.resample(y,sr,sampling_freq) 
            #mel-spectogram
            spec = librosa.feature.melspectrogram(y, sr=sampling_freq, n_fft=window_length, hop_length=hop_length, n_mels=num_mels)
            #Log scaling
            spec = librosa.power_to_db(spec,ref=1.0)
            infl_list = infl_new.strip().split("/")
            file_name = infl_list[-1].strip()
            out_dir = output_root + "/" + infl_list[-2]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            specfile = str(out_dir) + '/' + str(file_name) + '.orig.spec.npy'
            print (specfile)
            np.save(specfile, spec, allow_pickle=False)
            if infl_list[-2].strip() == "":
                output_mel_file = str(file_name) + '.orig.spec.npy'
            else:
                output_mel_file = infl_list[-2].strip() + '/' + str(file_name) + '.orig.spec.npy'
            print (output_mel_file)
            out_file_list.append(output_mel_file.strip())
    create_file(output_file,out_file_list)
            


if __name__ == "__main__":
    print (len(sys.argv))
    if len(sys.argv) != 9:
        print ('Takes arg1 = input-file list - formatted audio wav file list, arg2 = audio path, arg3 = output feature path, arg4 = list root path ')
    else:
        #Parameters hard coded as our paths remains fixed and less hassle changing parameters 
        parser = argparse.ArgumentParser()
        #print parser
        parser.add_argument('--input_file_list', type=str, default='wavfile.list', metavar='N',
                            help='input file list (default: wavfile.list)')
        parser.add_argument('--audiopath', type=str, default='audiopath', metavar='N',
                            help='audiopath - default:audiopath')
        parser.add_argument('--outputpath', type=str, default='outputpath', metavar='N',
                            help='outputpath - default:outputpath')
        parser.add_argument('--list_root', type=str, default='audio/lists/melspectrogram', metavar='N',
                            help='list_root - default:audio/lists/melspectrogram')
        args = parser.parse_args()      
        sys_arg_list = ((args.input_file_list.split('/')[-1]).split('.list')[0])
        list_root = args.list_root      
        if not os.path.exists(list_root):
                os.makedirs(list_root)
        input_list_file = args.input_file_list
        audio_root = args.audiopath
        output_root = args.outputpath
        output_file= list_root + sys_arg_list + "_melspectrogram.list"
        print (input_list_file)
        print (audio_root)
        print (output_root)
        print (output_file)
        processList(input_list_file)

#Sample command 
#python compute_melspectrograms.py --input_file_list ../../audio/lists/original/clean_data.list --audiopath ../../audio/original/clean_data/ --outputpath ../../features/melspectrogram/original/clean_data --list_root ../../lists/melspectrogram/

