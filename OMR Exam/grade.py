import sys
import timeit
import numpy as np
from PIL import Image, ImageOps

"""
Function: find_skew(image)
This function takes in image as the argument and returns the rho and theta of maximum value from the hough space.
This theta is used to re-rotate the image to make it straight if it is skewed by some angle.
"""
def find_skew(image):

    rows, columns = image.shape

    """
    I was struggling to implement hough transform.
    I have taken some ideas on how to implement hough transform. I have reffered it from
    #https://www.uio.no/studier/emner/matnat/ifi/INF4300/h09/undervisningsmateriale/hough09.pdf
    #https: // alyssaq.github.io / 2014 / understanding - hough - transform /
    """
    # Here we have set the resolution to 1 degree. So skew with minimum 1 degree can be detected.
    theta = np.linspace(-90, 90, 181) # -90 and 90 inclusive make 181 thetas

    #maximum distance in the image is calculated as the diagonal.
    diagonal = np.sqrt((rows - 1) ** 2 + (columns - 1) ** 2)
    rho = np.linspace(int(-diagonal),int(+diagonal), int(2*np.ceil(diagonal)+1))

    houghMatrix = np.zeros((len(rho), len(theta)))
    print("HoughMatrix shape",houghMatrix.shape)
    # Computed the cost thetas and sine thetas beforehand to aviod fucntion calls everytime.
    cos_t = np.cos(theta * np.pi / 180.0)
    sin_t = np.sin(theta * np.pi / 180.0)

    # Finding only those pixels which have value 255(white) to reduce the execution of the for loop
    # and reducing the time complexity of the execution.
    result = np.where(image == 255)
    Coordinates = list(zip(result[0], result[1]))
    # These coordinates with value as 255 are then used to iterate and then fill the hough matrix.
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

    #Finding the index values for the maximum votes in the hough space.
    argmaxs = np.argmax(houghMatrix, axis=0)
    maxIndex = tuple()
    max=0
    for i in range(len(houghMatrix[0])):
        if max < houghMatrix[argmaxs[i]][i]:
            max = houghMatrix[argmaxs[i]][i]
            maxIndex = (argmaxs[i],i)
    print("maxIndex", maxIndex)

    # img = Image.fromarray(houghMatrix)
    # if img.mode != 'L':
    #     img = img.convert('L')
    # img.save('hough_matrix.png')
    return maxIndex

"""
Function: detect_answers(image)
This function divides the image into 3 parts vertically. 
Reference points are then detected on each image which are the left upper corner of the first Option box in that image.
These reference points are used to segment the image for each question and its 5 option boxes. 
The answers are detected using thresholding.
"""
def detect_answers(image):
    part1 = image[:, 0:475]
    part2 = image[:, 475:910]
    part3 = image[:, 910:]
    Image.fromarray(part1).save('part1.png')
    Image.fromarray(part2).save('part2.png')
    Image.fromarray(part3).save('part3.png')
    threshold = 272

    #Finding the reference point for 3 parts of the image.
    row1,col1 = find_reference(part1)
    row2,col2 = find_reference(part2)
    row3,col3 = find_reference(part3)

    answers = []

    # interating over every part to segment and then detect answers.
    i = row1-6
    print("starting for part1",row1,col1)
    while i < len(part1):
        temp = ""
        extra_part = part1[i:i + 48, 10:col1 - 50]
        count_extra = np.count_nonzero(extra_part == 255)
        Image.fromarray(extra_part).save('extra_part.png')
        if count_extra > 50:
            add = "x"
        else:
            add = ""
        j = col1-12

        mcq1 = part1[i:i + 48,j:j + 61]
        # Image.fromarray(mcq1).save('mcq1.png')
        count1 = np.count_nonzero(mcq1 == 255)
        if count1>threshold:
            temp = temp+"A"
        j = j +61
        mcq2 = part1[i:i + 48,j:j + 61]
        # Image.fromarray(mcq2).save('mcq2.png')
        count2 = np.count_nonzero(mcq2 == 255)
        if count2>threshold:
            temp = temp+"B"
        j = j + 61
        mcq3 = part1[i:i + 48,j:j + 61]
        # Image.fromarray(mcq3).save('mcq3.png')
        count3 = np.count_nonzero(mcq3 == 255)
        if count3>threshold:
            temp = temp+"C"
        j = j + 61
        mcq4 = part1[i:i + 48,j:j + 61]
        # Image.fromarray(mcq4).save('mcq4.png')
        count4 = np.count_nonzero(mcq4 == 255)
        if count4>threshold:
            temp = temp+"D"
        j = j + 61
        mcq5 = part1[i:i + 48,j:j + 61]
        # Image.fromarray(mcq5).save('mcq5.png')
        count5 = np.count_nonzero(mcq5 == 255)
        if count5>threshold:
            temp = temp+"E"
        temp = temp+add
        answers.append(temp)
        # print("i",i)
        i = i+47
        if len(answers)==29:
            break
        # print("Here i ",i)
    # print(answers)

    print("starting for part2", row2, col2)
    i = row2-6
    while i < len(part2):
        temp = ""
        extra_part = part2[i:i + 48, 10:col3 - 50]
        count_extra = np.count_nonzero(extra_part == 255)
        Image.fromarray(extra_part).save('extra_part.png')
        if count_extra > 50:
            add = "x"
        else:
            add = ""
        j = col2 - 12

        mcq1 = part2[i:i + 48, j:j + 61]
        # Image.fromarray(mcq1).save('mcq1.png')
        count1 = np.count_nonzero(mcq1 == 255)
        if count1 > threshold:
            temp = temp + "A"
        j = j + 61
        mcq2 = part2[i:i + 48, j:j + 61]
        # Image.fromarray(mcq2).save('mcq2.png')
        count2 = np.count_nonzero(mcq2 == 255)
        if count2 > threshold:
            temp = temp + "B"
        j = j + 61
        mcq3 = part2[i:i + 48, j:j + 61]
        # Image.fromarray(mcq3).save('mcq3.png')
        count3 = np.count_nonzero(mcq3 == 255)
        if count3 > threshold:
            temp = temp + "C"
        j = j + 61
        mcq4 = part2[i:i + 48, j:j + 61]
        # Image.fromarray(mcq4).save('mcq4.png')
        count4 = np.count_nonzero(mcq4 == 255)
        if count4 > threshold:
            temp = temp + "D"
        j = j + 61
        mcq5 = part2[i:i + 48, j:j + 61]
        # Image.fromarray(mcq5).save('mcq5.png')
        count5 = np.count_nonzero(mcq5 == 255)
        if count5 > threshold:
            temp = temp + "E"
        temp = temp + add
        answers.append(temp)
        i = i+47
        if len(answers) == 58:
            break
        # print("Here i ",i)
    # print(answers)

    print("starting for part3", row3, col3)
    i = row3 - 6
    while i < len(part3):
        temp = ""
        extra_part = part3[i:i + 48, 10:col3 - 50]
        count_extra = np.count_nonzero(extra_part == 255)
        Image.fromarray(extra_part).save('extra_part.png')
        if count_extra > 50:
            add = "x"
        else:
            add = ""
        j = col3 - 12

        mcq1 = part3[i:i + 48, j:j + 61]
        # Image.fromarray(mcq1).save('mcq1.png')
        count1 = np.count_nonzero(mcq1 == 255)
        if count1 > threshold:
            temp = temp + "A"
        j = j + 61
        mcq2 = part3[i:i + 48, j:j + 61]
        # Image.fromarray(mcq2).save('mcq2.png')
        count2 = np.count_nonzero(mcq2 == 255)
        if count2 > threshold:
            temp = temp + "B"
        j = j + 61
        mcq3 = part3[i:i + 48, j:j + 61]
        # Image.fromarray(mcq3).save('mcq3.png')
        count3 = np.count_nonzero(mcq3 == 255)
        if count3 > threshold:
            temp = temp + "C"
        j = j + 61
        mcq4 = part3[i:i + 48, j:j + 61]
        # Image.fromarray(mcq4).save('mcq4.png')
        count4 = np.count_nonzero(mcq4 == 255)
        if count4 > threshold:
            temp = temp + "D"
        j = j + 61
        mcq5 = part3[i:i + 48, j:j + 61]
        # Image.fromarray(mcq5).save('mcq5.png')
        count5 = np.count_nonzero(mcq5 == 255)
        if count5 > threshold:
            temp = temp + "E"
        temp = temp + add
        answers.append(temp)
        # print("i",i)
        i = i + 47
        # print("Here i ",i)
        if len(answers)==85:
            break
    print(answers)
    return answers

"""
Function: find_reference(image)
This image takes in one part of the image as argument to find the reference.
I have counted the number of white pixels for each row and column and then used first few indices of the highest values.
I have sorted the indices which is used to find the left upper corner of first option box for the first question in that image.
"""
def find_reference(image):
    horizontals  = np.count_nonzero(image > 240,axis =1)
    verticals = np.count_nonzero(image >240, axis =0)

    Max_verticals = np.argpartition(verticals,-70)[-70:]
    Max_horizontals = np.argpartition(horizontals, -100)[-100:]

    Max_horizontals.sort()
    Max_verticals.sort()
    row_number=0
    i=0
    while row_number <70:
        row_number = Max_horizontals[i]
        print(row_number)
        i=i+1
    col_number =0
    j=0
    while col_number <70:
        col_number = Max_verticals[j]
        j=j+1
    return row_number,col_number

"""
Function: write_to_file(answers,file_name)
This function writes the answers detected and question numbers to a test file with name which was taken through command line.
"""
def write_to_file(answers,file_name):
    f = open(file_name, "w+")
    i = 0
    while i <len(answers):
        f.write(str(i+1) + " " + answers[i]+"\n")
        i = i+1
    f.close()


if __name__ == '__main__':

    #Store File name
    file_name = sys.argv[2]

    # Load an image
    image = Image.open(sys.argv[1])
    if image.mode != 'L':
        image = image.convert('L')
    # image = image.rotate(-5)
    image.save('Skewed_image.png')

    image = ImageOps.invert(image)
    image.save('invert.jpg')

    # edged = image.filter(ImageFilter.FIND_EDGES)
    # edged = np.asarray(edged)
    # print(edged.shape)
    edged = np.asarray(image)
    #Cropping image to get a smaller size image for hough transform
    image_arr = edged[600:-100,100:-100]
    part1 = image_arr[500:-500, 0:475]
    print("shape",part1.shape)
    croppped_image = Image.fromarray(image_arr)
    # croppped_image.save('Croppped_image.png')
    # print("Cropped Image shape",image_arr.shape)

    # start = timeit.default_timer()
    maxIndex = find_skew(part1)
    print("Rho",maxIndex[0])
    print("theta",maxIndex[1])
    # end = timeit.default_timer()
    # print("time",end-start)
    rotated = Image.fromarray(image_arr).rotate(maxIndex[1]-90)
    # rotated.save('After_Rotation.png')

    #using the cropped image to find the answers
    answers = detect_answers(np.asarray(rotated))

    #writing the answers to text file
    write_to_file(answers,file_name)






