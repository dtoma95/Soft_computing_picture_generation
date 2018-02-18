from __future__ import print_function

import h5py
from PIL import Image
import numpy as np
import io
import cv2
from numpy import array
import math


def mnozi_to_bre(prev_kernel, kernel, index, first, prev_minus):
    retval = np.zeros((len(kernel),len(kernel[0])))
    minus = np.zeros((len(kernel),len(kernel[0])))
    for i in range(0, len(kernel[0])):
        
        multip = 0
        minus_value = 0
        if(first):
            multip = prev_kernel[i][index]
            minus_value = prev_minus[i][index]
        else:
            for j in range(0, len(prev_kernel[i])):
                multip += prev_kernel[i][j]
                minus_value += prev_minus[i][j]
        
        
        for j in range(0, len(kernel)):
            if(kernel[j][i] > 0):
                retval[j][i] = math.log(kernel[j][i]) + multip
                minus[j][i] = minus_value
            else:
                retval[j][i] = math.log(kernel[j][i]*(-1)) + multip
                minus[j][i] = minus_value + 1
            
    return retval, minus

def generate_inputs(index, weight_file_path):
    """
    Generates optimal inputs for a selected class, based on h5 file

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
        prev_minus = None
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
                if("kernel" in param.keys()):
                    kernel = param.get("kernel")
                    if(svejedno == True):
                        svejedno = False
                        prev_kernel = np.zeros((len(kernel),len(kernel[0])))
                        prev_minus = np.zeros((len(kernel),len(kernel[0])))
                        print(kernel)
                        for red in range(0, len(kernel)):
                            if(kernel[red][index] > 0):
                                prev_kernel[red][index] = math.log(kernel[red][index])
                                prev_minus[red][index] = 0
                            else:
                                prev_kernel[red][index] = math.log(kernel[red][index]*(-1))
                                prev_minus[red][index] = 1
                    else:
                        print("herkules")
                        prev_kernel, prev_minus = mnozi_to_bre(prev_kernel, kernel, index, first, prev_minus)
                        first = False
            save_results(prev_kernel , glob)
            glob = glob + 1
            
        
        img = []
        img2 = []
        print("keut")
        for i in range(0, len(prev_kernel)):
            temp = 0
            for j in range(0, len(prev_kernel[i])):
                if(prev_kernel[i][j]%2 == 0):
                    temp += math.exp(prev_kernel[i][j]+2000000) #Mnozi sve sa e^2000000
                else:
                    temp += (-1)*math.exp(prev_kernel[i][j])
            img2.append(temp)
            if temp > 0:
                img.append(255)
            else:
                img.append(0)
        
        with open('zero_or_max.txt', 'w') as the_file:
            for number_in_img in img:
                the_file.write(str(number_in_img) + ", ")
                
        with open('raw_values.txt', 'w') as the_file:
            for number_in_img in img2:
                the_file.write(str(number_in_img) + ", ")
                
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
    blank_image = np.zeros((28,28,3), np.uint8)
    i = 0
    for red in blank_image:
        for kolona in red:
            for index in range(0, 3):
                kolona[index] = pixel_values[i]
            i = i+1
    cv2.imwrite("output.png", blank_image)
    
      
      
if __name__ == "__main__":

    generate_inputs(0, "digits_NN/end_result_new.h5")