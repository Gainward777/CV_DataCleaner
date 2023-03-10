import argparse
import torch
import clip
import os
from PIL import Image
import pickle as pk
from tqdm import tqdm

class Get_Features():

    def __init__(self):        
       #load the model and and the function that brings an image in line 
       #with the sample on which the model was trained 
        self.clip, self.preprocess=clip.load("ViT-B/32")      

        
    
    def get_features(self, image, device: str):       
        #transforms the image, submits it to the model and returns the features           
        transformed_im =  self.preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            im_features = self.clip.encode_image(transformed_im)
            im_features /= im_features.norm(dim=-1, keepdim=True)
        return im_features.cpu().numpy()
        

    def is_in_dict(self, im_name: str, features_dict: dict):
        #checks if a file exists in a directory
        if im_name in list(features_dict.keys()):                
                return True
        return False


    def is_in_folder(self, im_name: str, im_names: list):
        #checks if a key exists in a dirtionary
        if im_name in im_names:
                return True
        return False


    def syncrinize(self, im_names: list, features_dict: dict):                                     
        #checks if all the images specified in the dictionary are still available in the directory, 
        #if not, removes the excess from the dictionary
        for im_name in list(features_dict.keys()):                    
            if not self.is_in_folder(im_name, im_names):                
                try:
                    del features_dict[im_name]
                except:
                    continue   
                
                
    def get_path(self, im_directory: str):
        #get the paths of all files in a directory and subdirectories 
        all_files=[]
        for directory, _subdirs, files in os.walk(im_directory):
            for name in files:
                all_files.append(os.path.join(directory, name))
        return all_files


    def run(self, im_directory: str, output_path: str):
        #processing images in a folder with synchronization. 
        #if the image has already been processed and the data is already in the dictionary, 
        #its processing is skipped
        device="cuda" if torch.cuda.is_available() else "cpu"

        features_dict={}

        im_path_list=self.get_path(im_directory)
        #checks if there is already a dictionary file in the directory, 
        #if not, it will be created at the end of the execution
        
        try:
            features_dict=pk.load(open(f"{output_path}/{im_directory.split('/')[-1]}.pkl", "rb"))
        except (OSError, IOError) as e:
            print("file_not_found")
            

        self.syncrinize(im_path_list, features_dict)
        for im_path in tqdm(im_path_list):
                       
            if self.is_in_dict(im_path, features_dict):
                continue

            try:
                image = Image.open(im_path)
            except:
                continue

            image_features=self.get_features(image, device)
            
            features_dict[im_path]=image_features            
            
        pk.dump(features_dict, open(f"{output_path}/{im_directory.split('/')[-1]}.pkl","wb"))  


def arg_parse():    
    
    parser = argparse.ArgumentParser(description='Get featers from images')
   
    parser.add_argument("--directory", dest = 'directory', help = 
                        "Directory containing images to get feutes: /directory/sub_directory",
                        default = None, type = str) 
    
    parser.add_argument("--out", dest = 'out', help = 
                        "Output directory path: /directory/sub_directory",
                        default = None, type = str)    
    
    return parser.parse_args()


if __name__ == '__main__':

    args=arg_parse()    

    gf=Get_Features() 
    
   
    if args.out==None:
        args.out=args.directory
    
    gf.run(args.directory, args.out)