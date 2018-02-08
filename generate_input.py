from __future__ import print_function

import h5py
from PIL import Image
import numpy as np
import io
import cv2
from numpy import array



def mnozi_to_bre(prev_kernel, bias, kernel, index, first):
    retval = np.zeros((len(kernel),len(kernel[0])))
    for i in range(0, len(bias)):
        
        multip = bias[i]* 10000
        if(first):
            multip = multip* prev_kernel[i][index]
        else:
            for j in range(0, len(prev_kernel[i])):
                multip = multip* prev_kernel[i][j]* 100
                
        
        
        for j in range(0, len(kernel)):
            retval[j][i] = kernel[j][i]*multip
           
    return retval

def generate_inputs(index, weight_file_path):
    """
    Generates optimal inputs for a selected class, pased on h5 file

    Args:
      weight_file_path (str) : Path to the file to analyze
	  weight_file_path (int) : class to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 	
        prev_kernel = None
        first = True
        glob = 0
        svejedno = True
        for layer, g in reversed(f.items()):
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
			
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                if("bias" in param.keys() and "kernel" in param.keys()):
                    bias = param.get("bias")
                    kernel = param.get("kernel")
                    if(svejedno == True):
                        svejedno = False
                        prev_kernel = np.zeros((len(kernel),len(kernel[0])))
                        print(kernel)
                        for red in range(0, len(kernel)):
                            prev_kernel[red][index] = kernel[red][index]* bias[index]
                    else:
                        print("herkules")
                        prev_kernel = mnozi_to_bre(prev_kernel, bias, kernel, index, first)
                        first = False
            save_results(prev_kernel , glob)
            glob = glob + 1
            
        img = []
        img2 = []
    
        for red in prev_kernel:
            temp = sum(red)
            img2.append(temp)
            if temp > 0:
                img.append(255)
            else:
                img.append(0)
    
        write_as_img(img)
    finally:
        f.close()

 
def save_results(prev_kernel, glob):
    img = []
    img2 = []
    
    for red in prev_kernel:
        temp = sum(red)
        img2.append(temp)
        if temp > 0:
            img.append(255)
        else:
            img.append(0)
      
    with open('zero_or_max' + str(glob) + '.txt', 'w') as the_file:
        for number_in_img in img:
            the_file.write(str(number_in_img) + ", ")
                
    with open('raw_values' + str(glob) + '.txt', 'w') as the_file:
        for number_in_img in img2:
            the_file.write(str(number_in_img) + ", ")
    
    
    #write_as_img(img)
    glob = glob + 1
    


def write_as_img(pixel_values):
    blank_image = np.zeros((32,32,3), np.uint8)
    i = 0
    for red in blank_image:
        for kolona in red:
            for index in range(0, 3):
                kolona[index] = pixel_values[i]
                i = i+1
    cv2.imwrite("output.png", blank_image)
    
      
      
if __name__ == "__main__":

    generate_inputs(0, "end_result.h5")