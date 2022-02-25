#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import sys
import timeit
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageOps


"""
Function: inject(image,answer_dict)
This function takes in the image and answer dictionary as arguments and
returns the final array after injection.
"""
def inject(image,answer_dict):
    values=list(answer_dict.values())
    im_new=np.array(image)
    print(f"im_new shape: {im_new.shape}")
    
    #Extracting a block of pixels for injection
    part=im_new[100:1600,20:120]
    
    #Initializing i and j
    i=10
    j=10
    
    #Injecting a reference point
    part[i:i+10,j:j+10]=[0]
   
    #Reinitializing i and j
    i=30
    j=10
    
    #Injection procedure
    for k in values:
        # for questions with multiple answers
        if len(k)>1:
            for s in k:
                #print(i,k,s)
                if s=='A':
                    part[i:i+10,j:j+10]=[0]
                elif s=='B':
                    part[i:i+10,j+15:j+25]=[0]
                elif s=='C':
                    part[i:i+10,j+30:j+40]=[0]
                elif s=='D':
                    part[i:i+10,j+45:j+55]=[0]
                elif s=='E':
                    part[i:i+10,j+60:j+70]=[0]
            i+=10
        # for questions with single answer
        else:
            if k=='A':
                part[i:i+10,j:j+10]=[0]
            elif k=='B':
                part[i:i+10,j+15:j+25]=[0]
            elif k=='C':
                part[i:i+10,j+30:j+40]=[0]
            elif k=='D':
                part[i:i+10,j+45:j+55]=[0]
            elif k=='E':
                part[i:i+10,j+60:j+70]=[0]
            
            i+=10
    new_img = Image.fromarray(im_new)
    
    return new_img

"""
Function: find_skew(image)
This function takes in image as the argument and returns the rho and theta of maximum value from the hough space.
This theta is used to rerotate the image to make it straight if it is skewed by some angle.
"""
def find_skew(image):
    
    rows, columns = image.shape

    """
    Sanika was struggling to implement hough transform.
    She has taken some ideas on how to implement hough transform. She has reffered it from
    #https://www.uio.no/studier/emner/matnat/ifi/INF4300/h09/undervisningsmateriale/hough09.pdf
    #https: // alyssaq.github.io / 2014 / understanding - hough - transform /
    """

    theta = np.linspace(-90, 90, 181)

    diagonal = np.sqrt((rows - 1) ** 2 + (columns - 1) ** 2)
    rho = np.linspace(int(-diagonal),int(+diagonal), int(2*np.ceil(diagonal)+1))

    houghMatrix = np.zeros((len(rho), len(theta)))
    print("HoughMatrix shape",houghMatrix.shape)
    cos_t = np.cos(theta * np.pi / 180.0)
    sin_t = np.sin(theta * np.pi / 180.0)

    result = np.where(image == 255)
    Coordinates = list(zip(result[0], result[1]))

    for row, col in Coordinates:
    # for row in range(rows):
    #     for col in range(columns):
            # if image[row, col] > 0:  # image has values 0 or 255
                # print(row,col)
        for index in range(len(theta)):
            rhoVal = col * cos_t[index] + row * sin_t[index]
            houghMatrix[int(rhoVal+diagonal), index] += 1
    # print(houghMatrix)
    # print(np.amax(houghMatrix), np.amin(houghMatrix))

    """ ----- End of reffered code ------- """
    
    argmaxs = np.argmax(houghMatrix, axis=0)
    maxIndex = tuple()
    max=0
    for i in range(len(houghMatrix[0])):
        if max < houghMatrix[argmaxs[i]][i]:
            max = houghMatrix[argmaxs[i]][i]
            maxIndex = (argmaxs[i],i)
    print("maxIndex", maxIndex)

    img = Image.fromarray(houghMatrix)
    if img.mode != 'L':
        img = img.convert('L')
    img.save('hough_matrix.png')
    return maxIndex


if __name__ == '__main__':
    
    # Load an image
    image = Image.open(sys.argv[1])
    # image = image.rotate(0.5)
    
    if image.mode != 'L':
        image = image.convert('L')
    
    answer_dict={}
    answers=open(sys.argv[2])
    
    for i in answers:
        key,val=i.split()
        answer_dict[key]=val
    print(answer_dict)
    
    edged = np.asarray(image)
    #Cropping image to get a smaller size image for hough transform
    image_arr = edged[600:-100,100:-100]
    part1 = image_arr[500:-500, 0:475]
    # start = timeit.default_timer()
    maxIndex = find_skew(np.asarray(ImageOps.invert(Image.fromarray(part1))))
    print("Rho",maxIndex[0])
    print("theta",maxIndex[1])
    # end = timeit.default_timer()
    # print("time",end-start)
    rotated = Image.fromarray(edged).rotate(maxIndex[1]-90)

    injected = inject(rotated,answer_dict)
    injected.save(sys.argv[3])
    