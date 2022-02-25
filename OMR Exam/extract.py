import sys

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import sys
import random
#import cv2

"""
Function: find_skew(image)
This function takes in image as the argument and returns the rho and theta of maximum value from the hough space.
This theta is used to re-rotate the image to make it straight if it is skewed by some angle.
"""
def find_skew(image):
    #https://www.uio.no/studier/emner/matnat/ifi/INF4300/h09/undervisningsmateriale/hough09.pdf
    rows, columns = image.shape

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



"""
Function: find_reference(image)
Very similar to the find_reference in grade.py, this function finds the starting point of
where our answers are injected in the image. We injected an extra 'A' answer at the beginning
of the injection just so our reference point would be found at the same point every time. I
iterate through the rows and columns until I find a black pixel, and from there I skip down an
extra 20 to compensate for the injected reference point and that's where I begin extracting answers
"""
def find_reference(image):
    im = np.asarray(image)
    rows, cols = im.shape
    threshold = 25

    break_out_flag = False
    for x in range(cols-1):
        for y in range(rows-1):
            if im[y,x] < threshold:
                init_reference_point = (x,y)
                break_out_flag = True
                break

        if break_out_flag:
            break


    final_reference_point = init_reference_point[0],init_reference_point[1]+20

    return final_reference_point


"""
Function: extract_answers(image)
This function finds the initial pixel where answers were injected in the image, and scans
in the injected answers until there's none left. It takes 10 rows at a time from the image cropped 
around where the injection took place, and scans 5 ranges of columns in that row that each correspond
to an A,B,C,D, or E and records the results.
"""
def extract_answers(image):
    im_new = np.asarray(image)
    cropped = im_new[0:1700, 0:200]

    ref_point = find_reference(cropped)
    x,y = ref_point
    threshold = 25
    results = []

    while True:
        curr = cropped[y:y+10, x:x+75]

        temp = ''
        a = np.average(curr[0:10, 0:10])
        b = np.average(curr[0:10, 15:25])
        c = np.average(curr[0:10, 30:40])
        d = np.average(curr[0:10, 45:55])
        e = np.average(curr[0:10, 60:70])


        if a < threshold:
            temp += 'A'
        if b < threshold:
            temp += 'B'
        if c < threshold:
            temp += 'C'
        if d < threshold:
            temp += 'D'
        if e < threshold:
            temp += 'E'
        if all(i > threshold for i in [a,b,c,d,e]):
            break
        
        if temp:
            results.append(temp)
        y+=10

    return results




if __name__ == '__main__':
    image = Image.open(sys.argv[1])
    output = sys.argv[2]

    if image.mode != 'L':
        image = image.convert('L')


    np_array = np.asarray(image)
    crop1 = np_array[600:-100, 100:-100]
    crop2 = crop1[500:-500, 0:475]

    angles = find_skew(np.asarray(ImageOps.invert(Image.fromarray(crop2)))) 

    rotated = Image.fromarray(np_array).rotate(angles[1]-90)

    results = extract_answers(rotated)

    output_file = open(output, 'w')
    for i, ans in enumerate(results):
        output_file.write(f"{i+1} {ans}\n")

    output_file.close()
