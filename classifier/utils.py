import torch
import os
import math
import pickle
import numpy as np

def load_model(netx,modpath,strict_val=True):
    print (strict_val)
    model_dict=netx.state_dict()
    pretrained_dict = torch.load(modpath)
    print (model_dict)
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    print (pretrained_dict)
    model_dict.update(pretrained_dict)
    #netx.load_state_dict(torch.load(modpath),strict=strict_val)
    netx.load_state_dict(model_dict)

def load_model_cpu(netx,modpath,strict_val=True):
    state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    netx.load_state_dict(new_state_dict,strict=strict_val)

def dump_pickle_dict(filename,dict_obj):
	if not os.path.isdir(os.path.dirname(filename)):
		os.makedirs(os.path.dirname(filename))
	output_file=open(filename,'wb')
	pickle.dump(filename,dict_obj)

def numpy_save(filename,obj):
	if not os.path.isdir(os.path.dirname(filename)):
		os.makedirs(os.path.dirname(filename))
	np.save(filename,obj)

