import scipy.io
import numpy as np
import pandas as pd 
import os
import argparse
import sys
from pathlib import Path

output_folder = "./output"

def process(folder,path):
    if folder:
        input = scipy.io.loadmat(os.path.join(folder,path))
    else:
        input = scipy.io.loadmat(path)
    mat = input
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label 
    y = y[:, 0]
    y=y[:,None]
    X=np.append(X,y,axis=1)
    n_samples, n_features = X.shape
    print("Samples , Features : ",n_samples," ",n_features)
    he=np.arange(n_features)
    he+=1
    a_list = list(range(1, n_features))
    va=[]
    for val in a_list:
        va.append("Att"+str(val))
    va.append('Class')
    df=pd.DataFrame(X)
    if folder:
        df.to_csv(os.path.join(output_folder,Path(path).stem+".csv"),index=False,header=va)
    else:
        df.to_csv(Path(path).stem+".csv",index=False,header=va)

def main(args):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if args.input_folder:
        input_pths = os.listdir(args.input_folder)
        count = 1
        for input_pth in input_pths:
            print("File ",count,":", input_pth)
            count+=1
            process(args.input_folder,input_pth)
    else:
        process(None,args.input)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--input', type=str,
		help='input input',default = None)
	parser.add_argument('--input_folder', type=str,
		help='input input',default = None)
	parser.add_argument('--gpu_fraction', type=float,
		help='how much gpu is needed, usually 4G is enough',default = 1.0)
	return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))