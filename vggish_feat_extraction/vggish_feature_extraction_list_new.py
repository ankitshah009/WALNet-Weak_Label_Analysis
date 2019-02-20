# Copyrigh 2017 The tensorflow authors all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
# ==============================================================================

"""a smoke test for vggish.

this is a simple smoke test of a local install of vggish and its associated
downloaded files. We create a synthetic sound, extract log mel spectrogram
features, run them through VGGish, post-process the embedding ouputs, and
check some simple statistics of the results, allowing for variations that
might occur due to platform/version differences in the libraries we use.

Usage:
- Download the VGGish checkpoint and PCA parameters into the same directory as
  the VGGish source code. If you keep them elsewhere, update the checkpoint_path
  and pca_params_path variables below.
- Run:
  $ python vggish_smoke_test.py
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import numpy as np
import librosa
import os
import sys
import argparse
from create_file import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print('\nTesting your install of VGGish\n')

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'

# Relative tolerance of errors in mean and standard deviation of embeddings.
rel_error = 0.1  # Up to 10%

def processList(filelist):
    # Generate a 1 kHz sine wave at 44.1 kHz (we use a high sampling rate
    # to test resampling to 16 kHz during feature extraction).
    out_file_list = []
    input_list=open(filelist)
    input_lists=input_list.readlines()
    for i,infl in tqdm(enumerate(input_lists)):
        infl_new = audio_root + '/' + infl.replace("\ "," ") 
        y,sr = librosa.load(infl_new.strip(),sr=None)
        if len(y) < sr:
            y1=np.pad(y,(0,sr-len(y)),'wrap')
            y=y1
        # Produce a batch of log mel spectrogram examples.

        input_batch = vggish_input.waveform_to_examples(y, sr)
        #print('Log Mel Spectrogram example: ', input_batch[0])
        # Define VGGish, load the checkpoint, and run the batch through the model to
        # produce embeddings.
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
          vggish_slim.define_vggish_slim()
          vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

          features_tensor = sess.graph.get_tensor_by_name(
              vggish_params.INPUT_TENSOR_NAME)
          embedding_tensor = sess.graph.get_tensor_by_name(
              vggish_params.OUTPUT_TENSOR_NAME)
          [embedding_batch] = sess.run([embedding_tensor],
                                       feed_dict={features_tensor: input_batch})
          print('VGGish embedding: done ',i)

        # Postprocess the results to produce whitened quantized embeddings.
        #print(embedding_batch)
        pproc = vggish_postprocess.Postprocessor(pca_params_path)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        #print(postprocessed_batch)

        infl_list = infl_new.strip().split("/")
        file_name = infl_list[-1].strip()
        out_dir = output_root + "/" + infl_list[-2]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        featfile = str(out_dir) + '/' + str(file_name) + '.txt'
        out_file_list.append(featfile.strip())
        np.savetxt(featfile,postprocessed_batch.astype(int),fmt='%i',delimiter=",")
    create_file(output_file,out_file_list)

print('\nLooks Good To Me!\n')

if __name__=="__main__":
    print(len(sys.argv))
    if len(sys.argv) != 9:
        print('Takes arg1 = input-file list - formatted audio wav file list, arg2 = audio path, arg3 = output feature path, arg4 = list root path ')
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
	output_file= list_root + sys_arg_list + "_vggish.list"
	print (input_list_file)
	print (audio_root)
	print (output_root)
	print (output_file)
	processList(input_list_file)

#Sample command 
#python vggish_feature_extraction_list.py --input_file_list <list> --audiopath <rel_path> --outputpath <rel path> --list_root <rel list path>

