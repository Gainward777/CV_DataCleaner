from sklearn.neighbors import NearestNeighbors
from os import listdir
import numpy as np
import pickle as pk
from PIL import Image
import torch
import argparse
from get_features import Get_Features

def get_others(pattern_im_name: str, file_path: str, output_path: str, distance: float, save_list: bool=True):

    #getting the features of the image to be compared with, 
    #and removing it from the total selection, 
    #for separate submission to the model
                #features_dict=pk.load(open(f"{directory}/{directory.split('/')[-1]}.pkl", "rb"))
    features_dict=pk.load(open(f"{file_path}", "rb"))
    if not pattern_im_name in features_dict:
        gf=Get_Features()
        device="cuda" if torch.cuda.is_available() else "cpu"
        try:
            image = Image.open(pattern_im_name)
        except Exception as ex:
                print(ex)
        pattern_image=gf.get_features(image, device)
    else:
        pattern_image=features_dict[pattern_im_name]
        del features_dict[pattern_im_name]
    
    #getting an array with images features and
    #converting its to the appropriate format
    all_im_features=list(features_dict.values())
    all_im_features=np.array(all_im_features)
    all_im_features=np.squeeze(all_im_features)    

    #getting an array with file addresses from the rest of the selection
    file_names=list(features_dict.keys())

    #just fit and predict
    knn = NearestNeighbors(n_neighbors=len(file_names),algorithm='brute',metric='euclidean')
    knn.fit(all_im_features)

    dist, indices = knn.kneighbors(pattern_image, return_distance=True)

    #get dict with only others
    others_dict={}
    for i in indices[0][:len(dist[0][dist[0]>distance])]: 
        key=file_names[i]
        others_dict[key]=features_dict[key] 
    
    if save_list:
        if output_path==file_path:
            pk.dump(others_dict, open(f"{file_path.split('/')[-1].split('.')[0]}.pkl","wb"))
        else:
            pk.dump(others_dict, open(f"{output_path}/{file_path.split('/')[-1].split('.')[0]}.pkl","wb"))
    else:
        return list(others_dict.keys()) 


def arg_parse():
        
    parser = argparse.ArgumentParser(description='Get same files list')
   
    parser.add_argument("--image", dest = 'image', help = 
                        "Pattern image path: /directory/sub_directory/Image_1.jpg",
                        default = "", type = str)
    
    parser.add_argument("--dict", dest = 'dict', help = 
                        "Feutures file path: /directory/sub_directory/file_name.pkl",
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
        args.out=args.dict

    get_others(args.image, args.dict, args.out, args.distance)

