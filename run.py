from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
import pickle as pk
from PIL import Image
import torch
import argparse
from get_features import Get_Features
from get_others import get_others
import shutil


def move(directory:str, out_directory: str, pattern_im_name: str, distance: float):
    #gets a dictionary with features, selects extra images and moves them to a 
    #separate folder along with the dictionary

    #run get_featers.py
    gf=Get_Features()
    gf.run(directory, out_directory)

    dict_path=f"{out_directory}/{directory.split('/')[-1]}.pkl"

    #run get_others.py in the modification of returning paths to images
    result_list=get_others(pattern_im_name, dict_path, out_directory, distance, False)

    #creating a directory and moving unnecessary files there
    others_directory=f"{out_directory}/others"
    try:
        os.mkdir(others_directory)
    except:
        None
    
    for im_path in result_list:        
        path = shutil.move(im_path, others_directory)
    path = shutil.move(dict_path, others_directory)


def arg_parse():    
    
    parser = argparse.ArgumentParser(description='Get featers from images')
   
    parser.add_argument("--directory", dest = 'directory', help = 
                        "Directory containing images to get feutes: /directory/sub_directory",
                        default = None, type = str) 
    
    parser.add_argument("--image", dest = 'image', help = 
                        "Pattern image path: /directory/sub_directory/Image_1.jpg",
                        default = "", type = str)    
    
    parser.add_argument("--distance", dest = 'distance', help = 
                        "Distance in the distribution field - float value from 0 to 2, for example 0.5",
                        default = 0.5, type = float) 

    parser.add_argument("--out", dest = 'out', help = 
                        "Output directory: /directory/sub_directory",
                        default = None, type = str) 
    
    return parser.parse_args()


if __name__ == '__main__':

    args=arg_parse()
   
    if args.out==None:
        args.out=args.directory
    
    move(args.directory, args.out, args.image, args.distance)
    



