import numpy as np
from PIL import Image
from numpy.linalg import inv
import sys
from copy import deepcopy as deepcopy
import cv2
from sklearn.cluster import AgglomerativeClustering

#Code for ORB matches from:https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
#From Brute-Force Matching with SIFT Descriptors and Ratio Test(using ORB instead of SIFT in this case)

def image_matching_and_clustering(images,arr,k):

    #matrix of matches to feed the clustering function
    number_of_matches_matrix = np.zeros(shape=(len(images), len(images)))
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            #create orb object for image 1 and find keypoints and descriptors
            orb_1 = cv2.ORB_create()
            kp1, des1 = orb_1.detectAndCompute(list(images.values())[i],None)

            #create orb object for image 2 and find keypoints and descriptors
            orb_2 = cv2.ORB_create()
            kp2, des2 = orb_2.detectAndCompute(list(images.values())[j],None)

            #use brute force matching and knnMatch to find matching points.
            #We take k=2 to get the 2 closest matching points
            bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
            matches = bf.knnMatch(des1,des2,k=2)
            good = []
            for first_closest,second_closest in matches:
                #apply thresholding to get better matches
                #if the closest distance is much smaller than the second closest distance,
                #then that's a good match

                #logic for this used from Lowe's ratio test
                #link to paper https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94
                if first_closest.distance/second_closest.distance < 0.70 :
                    good.append(first_closest)
            number_of_matches_matrix[i][j] = len(good)

            #Switch the order and carry out the same steps as above
            orb_3 = cv2.ORB_create()
            kp3, des3 = orb_3.detectAndCompute(list(images.values())[j],None)
            orb_4 = cv2.ORB_create()
            kp4, des4 = orb_3.detectAndCompute(list(images.values())[i],None)
            bf1 = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
            matches1 = bf1.knnMatch(des3,des4,k=2)
            good1 = []
            for first_closest,second_closest in matches1:
                if first_closest.distance/second_closest.distance < 0.70 :
                    good1.append(first_closest)
            number_of_matches_matrix[j][i] = len(good1)
            

    #AgglomerativeClustering Function from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    clustering = AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='complete').fit(number_of_matches_matrix).labels_

    z= zip(arr,clustering)        
    new_list=list(z)
    res = sorted(new_list, key = lambda x: x[1])
    print("Number of images:")
    print(len(res))

    dictionary_list_1={}
    dictionary_list_2={}
    for i in range(k):
        dictionary_list_1[i]=[]

    for i in range(0,len(res)):
        dictionary_list_1[res[i][1]].append(res[i][0])

    for i in range(0,len(res)):
        dictionary_list_2[res[i][0]]=res[i][1]


    total_pairs=len(clustering)*(len(clustering)-1)
    print(total_pairs)
    true_positives=0
    true_negatives=0

    

    #find pairs of images that should be in the same cluster and are in the same cluster(True positive)
    #also find pairs of images that shouldn't be in the same cluster and aren't(True negative)
    for i in range(0,len(res)):
        for j in range(i+1, len(res)):
            
            i1=res[i][0]
            i2=res[j][0]
            i_1 = i1.replace("_", "")
            i_2 = i2.replace("_", "")
            im1 = ''.join([i for i in i_1 if not i.isdigit()])
            im2 = ''.join([i for i in i_2 if not i.isdigit()])

            #True positive
            if im1==im2 and dictionary_list_2[i1]==dictionary_list_2[i2]:
                true_positives+=1
            elif im1!=im2 and dictionary_list_2[i1]!=dictionary_list_2[i2]:
                true_negatives+=1

            i3=res[j][0]
            i4=res[i][0]
            i_3 = i3.replace("_", "")
            i_4 = i4.replace("_", "")
            im3 = ''.join([i for i in i_3 if not i.isdigit()])
            im4 = ''.join([i for i in i_4 if not i.isdigit()])
            if im3==im4 and dictionary_list_2[i3]==dictionary_list_2[i4]:
                true_positives+=1
            elif im3!=im4 and dictionary_list_2[i3]!=dictionary_list_2[i4]:
                true_negatives+=1
    print("Number of true positives: ", true_positives)
    print("Number of true negatives: ",true_negatives)
    #Calculate the accuracy
    accuracy=(true_positives+true_negatives)/total_pairs
    print("The Pairwise Clustering Accuracy is: ")
    print(accuracy*100)

    filename=sys.argv[-1]
    list_of_cluster_indexes=list(dictionary_list_1.keys())
    print("Writing clusters to file "+filename)
    #Writing the clustering results in file
    for i in range(0,len(list_of_cluster_indexes)):
        line_here=dictionary_list_1[list_of_cluster_indexes[i]]
        with open(filename, 'a') as f:
            for j in range(len(line_here)):
                if "/" in line_here[j]:
                    line_here[j]=line_here[j].split("/")[1]
                f.write(line_here[j]+" ")
            f.write("\n")

"""
Function to find the matching points between two images using ORB Descriptors. 
"""
def orb_descriptor(image1,image2):
    #https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    #Code for ORB matches from:https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    #From Brute-Force Matching with SIFT Descriptors and Ratio Test(using ORB instead of SIFT in this case)

    orb_1 = cv2.ORB_create()

    #Detect the keypoints required in both images
    kp1, des1 = orb_1.detectAndCompute(image1,None)
    kp2, des2 = orb_1.detectAndCompute(image2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    #Find the keypoint matches in both images. Using knnMatch with k=2 to get the 2 closest matches
    matches = bf.knnMatch(des1,des2,k=2)
    print("---------------------------------------------",len(matches))
    good = []
    for first_closest,second_closest in matches:
        #logic for this used from Lowe's ratio test
        #link to paper https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94
        if first_closest.distance/second_closest.distance < 0.9 :
            good.append(first_closest)
    img3 = cv2.drawMatches(image1,kp1,image2,kp2,good,None)
    cv2.imwrite('image.jpg',img3)

    #Store the matching keypoint coordinates for image 1 and image 2 and return 2 lists.
    image1_points = []
    image2_points = []
    for match in good:
        query_index=match.queryIdx
        train_index=match.trainIdx
        image1_points.append(kp1[query_index].pt)
        image2_points.append(kp2[train_index].pt)
    return image1_points, image2_points

"""
Function to warp the image using bilinear interpolation and inverse warping.
"""
def warp(image_array, inverse_tm):
    #Reffered the video https://youtu.be/UhGEtSdBwIQ for bilinear interpolation
    dest_image = np.zeros(image_array.shape)
    #iterate over all pixels of the image
    for col in range(len(image_array[0])):
        for row in range(len(image_array)):
            source_pixels = np.dot(inverse_tm, [col, row, 1])
            # find the location in the original image where the new pixels values map.
            if (source_pixels[0] > 0 and source_pixels[1] > 0 and source_pixels[2] > 0) or (
                    source_pixels[0] < 0 and source_pixels[1] < 0 and source_pixels[2] < 0):
                source_x = source_pixels[1] / source_pixels[2]
                source_y = source_pixels[0] / source_pixels[2]
                # find the distances from the neighbouring 4 locations.
                x_ = int(np.floor(source_x))
                y_ = int(np.floor(source_y))
                # print(x_,y_)
                a = source_x - x_
                b = source_y - y_

                #using the below loop to find all the R G B values.
                rgb = [0, 0, 0]
                for i in range(3):
                    # print(i)
                    # perform bilinear interpolation
                    if x_ + 1 < len(image_array) and y_ + 1 < len(image_array[0]):
                        value = (1 - a) * (1 - b) * image_array[x_][y_][i] + (1 - a) * (b) * image_array[x_][y_ + 1][
                            i] + (a) * (1 - b) * image_array[x_ + 1][y_][i] + (a) * (b) * image_array[x_ + 1][y_ + 1][i]
                        rgb[i] = int(value)

                dest_image[row][col] = rgb

    dest_image = dest_image.astype(np.uint8)
    return dest_image

"""
Function to find the transformation matrix between a source and destination image.
"""
def transform(n, dest_img_p1, dest_img_p2, dest_img_p3, dest_img_p4, src_img_p1, src_img_p2, src_img_p3, src_img_p4):

    if n==1:
        # n = 1 --- Translation
        matrix = np.array([[1, 0, abs(dest_img_p1[0] - src_img_p1[0])], [0, 1, abs(dest_img_p1[1] - src_img_p1[1])], [0, 0, 1]])
        return matrix

    if n == 2:
        #n=2 --- Euclidean
        # j = [img2_p1, img2_p2]
        # k = np.array(img1_p1)
        # # print(j)
        # # print(k)
        # res1 = np.linalg.solve(j, k)
        #
        # j = [img2_p1, img2_p2]
        # k = np.array(img1_p2)
        # res2 = np.linalg.solve(j, k)
        matrix = np.array([
            [src_img_p1[0], -src_img_p1[1], 1, 0],
            [src_img_p1[1], src_img_p1[0], 0, 1],
            [src_img_p2[0], -src_img_p2[1], 1, 0],
            [src_img_p2[1], src_img_p2[0], 0, 1]
        ])

        mat_b = np.array(
            [[dest_img_p1[0]], [dest_img_p1[1]], [dest_img_p2[0]], [dest_img_p2[1]]])
        x = np.linalg.solve(matrix, mat_b)
        print("x",x)
        # x = np.append(x, [0, 0, 1])
        matrix = np.array([
            [x[0],-x[1],x[2]],
            [x[1],x[0],x[3]],
            [0,0,1]
        ],dtype=float)

        return matrix

    if n == 3:
        #n=3 --- Affine
        matrix = np.array([
            [src_img_p1[0], src_img_p1[1], 1, 0, 0, 0],
            [0, 0, 0, src_img_p1[0], src_img_p1[1], 1],
            [src_img_p2[0], src_img_p2[1], 1, 0, 0, 0],
            [0, 0, 0, src_img_p2[0], src_img_p2[1], 1],
            [src_img_p3[0], src_img_p3[1], 1, 0, 0, 0],
            [0, 0, 0, src_img_p3[0], src_img_p3[1], 1]])

        mat_b = np.array(
            [[dest_img_p1[0]], [dest_img_p1[1]], [dest_img_p2[0]], [dest_img_p2[1]], [dest_img_p3[0]], [dest_img_p3[1]]])
        x = np.linalg.solve(matrix, mat_b)
        x = np.append(x, [0,0,1])
        # print(x)
        # print(x.reshape((3, 3)))
        matrix = x.reshape((3, 3))
        # matrix = np.array([res1, res2, [0, 0, 1]])

        return matrix

    if n == 4:
        #n=4 --- Projective
        matrix = np.array([
            [src_img_p1[0], src_img_p1[1], 1, 0, 0, 0, -dest_img_p1[0] * src_img_p1[0], -dest_img_p1[0] * src_img_p1[1]],
            [0, 0, 0, src_img_p1[0], src_img_p1[1], 1, -dest_img_p1[1] * src_img_p1[0], -dest_img_p1[1] * src_img_p1[1]],
            [src_img_p2[0], src_img_p2[1], 1, 0, 0, 0, -dest_img_p2[0] * src_img_p2[0], -dest_img_p2[0] * src_img_p2[1]],
            [0, 0, 0, src_img_p2[0], src_img_p2[1], 1, -dest_img_p2[1] * src_img_p2[0], -dest_img_p2[1] * src_img_p2[1]],
            [src_img_p3[0], src_img_p3[1], 1, 0, 0, 0, -dest_img_p3[0] * src_img_p3[0], -dest_img_p3[0] * src_img_p3[1]],
            [0, 0, 0, src_img_p3[0], src_img_p3[1], 1, -dest_img_p3[1] * src_img_p3[0], -dest_img_p3[1] * src_img_p3[1]],
            [src_img_p4[0], src_img_p4[1], 1, 0, 0, 0, -dest_img_p4[0] * src_img_p4[0], -dest_img_p4[0] * src_img_p4[1]],
            [0, 0, 0, src_img_p4[0], src_img_p4[1], 1, -dest_img_p4[1] * src_img_p4[0], -dest_img_p4[1] * src_img_p4[1]]])

        mat_b = np.array(
            [[dest_img_p1[0]], [dest_img_p1[1]], [dest_img_p2[0]], [dest_img_p2[1]], [dest_img_p3[0]], [dest_img_p3[1]], [dest_img_p4[0]],
             [dest_img_p4[1]]])

        x = np.linalg.solve(matrix, mat_b)
        x = np.append(x, [1])
        # print(x.reshape((3, 3)))
        matrix = x.reshape((3, 3))

        return matrix

def letsStitch(image1, image2, bestTransMat, invTransMat):
    #Given below is an honest attempt of writing our own homegrown version of the cv2.getRectSubPix method
    #Unfortunately it did no work out due to some unknown reasons and hence we decided to go ahead with the cv2.getRectSubPix method
    """
    x_ = int(pt_[1])
    y_ = int(pt_[0])
    # print(x_,y_)
    a = pt_[1] - x_
    b = pt_[0] - y_

    rgb = [0, 0, 0]
    for i in range(3):
        # print(i)
        if x_ + 1 < len(image2) and y_ + 1 < len(image2[0]):
            value = (1 - a) * (1 - b) * image2[x_][y_][i] + (1 - a) * (b) * image2[x_][y_ + 1][i] + (a) * (1 - b) * image2[x_ + 1][y_][i] + (a) * (b) * image2[x_ + 1][y_ + 1][i]
            rgb[i] = int(value)
    # print(rgb)
    if stitched_image[y,x,0] != 0:
        print((rgb + stitched_image[y + offset_y, x + offset_x,:])/2)
        stitched_image_[y + offset_y, x + offset_x,:] = (rgb + stitched_image[y + offset_y, x + offset_x,:])/2
    else:
        stitched_image_[y + offset_y, x + offset_x,:] = rgb
    """
    """
    We were having some problems while stitching the images so we have refrenced the following code from
    https://github.com/melanie-t/ransac-panorama-stitch/blob/master/src/PanoramaStitching.py
    """
    #############################
    #Referenced code starts here#
    #############################
    image2_h, image2_w, image2_ch = image2.shape
    print(image2_h, image2_w, image2_ch)

    #Finding corners of images after bringing them in the same coordinate system.

    # Top-Left Corner
    pt1 = np.array([0, 0, 1])
    pt1_ = np.dot(invTransMat, pt1)
    pt1_ = pt1_ / pt1_[2]

    # Top-Right Corner
    pt2 = np.array([0, image2_h - 1, 1])
    pt2_ = np.dot(invTransMat, pt2)
    pt2_ = pt2_ / pt2_[2]

    # Bottom-Left Corner
    pt3 = np.array([image2_w - 1, 0, 1])
    pt3_ = np.dot(invTransMat, pt3)
    pt3_ = pt3_ / pt3_[2]

    # Bottom-Right Corner
    pt4 = np.array([image2_w - 1, image2_h - 1, 1])
    pt4_ = np.dot(invTransMat, pt4)
    pt4_ = pt4_ / pt4_[2]

    
    #Add the values from image2's corners that are outside of image1's width and height to get the new size of the stitched image.
    image1_h, image1_w, image1_ch = image1.shape

    # When finding the minimum values, we added 0 since padding is 0 if the corners of image2 are inside the size of image1.
    Xmin = np.round(min(0, pt1_[0], pt2_[0], pt3_[0], pt4_[0])).astype(int)  # The offset for x
    Ymin = np.round(min(0, pt1_[1], pt2_[1], pt3_[1], pt4_[1])).astype(int)  # The offset for y
    offset_x = np.abs(Xmin)  # Padding added to the right of image1's width (offset of x)
    offset_y = np.abs(Ymin)  # Padding added to the top of image1's height (offset of y)

    # We incorporated image1's height and width when establishing the maximum value of image2's projected corners, 
    # thus the padding will be 0 if image2's corners fall inside image1's height and breadth.
    Xmax = np.round(max(image1_w, pt1_[0], pt2_[0], pt3_[0], pt4_[0])).astype(int)
    Ymax = np.round(max(image1_h, pt1_[1], pt2_[1], pt3_[1], pt4_[1])).astype(int)

    # We only need the offset to pad image1, so we ignore the maximum x and y values produced by projecting image2's corners with image1's height and width.
    new_width = offset_x + image1_w + np.abs(image1_w - Xmax)
    new_height = offset_y + image1_h + np.abs(image1_h - Ymax)

    #Creating a canvas for our images 
    stitched_image = np.zeros((new_height, new_width, 3))

    #Putting image1 onto the canvas
    for y in range(image1_h):
        for x in range(image1_w):
            stitched_image[y + offset_y, x + offset_x] = image1[y, x]

    #Getting shape of the new canvas
    stitched_image_h, stitched_image_w, c = stitched_image.shape

    #print("\tTransferring pixels from the second image to the stitched image")
    for y in range(Ymin, stitched_image_h): 
        for x in range(Xmin, stitched_image_w):
            if y + offset_y < stitched_image_h and x + offset_x < stitched_image_w:

                # using a transformation to align the 2 images

                pt = np.array([x, y, 1])
                pt_ = np.dot(bestTransMat, pt)
                pt_ = pt_ / pt_[2]

                # cv2.getRectSubPix works like a bilinear interpolation method

                if 0 < pt_[0] < image2_w and 0 < pt_[1] < image2_h:
                    pixel = cv2.getRectSubPix(image2, (1, 1), (pt_[0], pt_[1]))
                    stitched_image[y + offset_y, x + offset_x] = pixel 

    return stitched_image

    ###########################
    #Referenced code ends here#
    ###########################



if __name__=="__main__":
    if len(sys.argv) < 4:
        print("Insufficient arguments given! Exiting code")
        sys.exit(0)
    part_number = sys.argv[1]
    if part_number == "part1":
        print("starting Part 1:")
        images ={}
        arr=[]
        images_index=0
        for i in range(3,len(sys.argv)-1):
            images[images_index]=cv2.imread(sys.argv[i])
            arr.append(sys.argv[i])
            images_index+=1

        k = int(sys.argv[2])
        image_matching_and_clustering(images,arr,k)



    if part_number == "part2":

        # below are the commmands used for some tests.
        # python a2.py part2 4 scene2.jpg scene1.jpg scene_output_projective2.jpg 476,243 220,328 449,246 192,332 700,73 442,158 671,363 414,449
        # python a2.py part2 4 scene1.jpg scene2.jpg scene_output_projective3.jpg 220,328 476,243 192,332 449,246 442,158 700,73 414,449 671,363
        #part2 4 part2-images\book2.jpg part2-images\book1.jpg try.jpg 141,131 318,256 480,159 534,372 493,630 316,670 64,601 73,473

        print("Starting Part 2:")
        n = int(sys.argv[2])
        src = sys.argv[3]
        dest = sys.argv[4]

        dest_image = Image.open(dest)
        dest_image = np.asarray(dest_image)

        src_image = Image.open(src)
        src_image = np.asarray(src_image)

        output_image = sys.argv[5]
        src_img_p1 = np.array([ int(i) for i in sys.argv[6].split(",")])
        dest_img_p1 = np.array([int(i) for i in sys.argv[7].split(",")])

        src_img_p2 = np.array([int(i) for i in sys.argv[8].split(",")])
        dest_img_p2 = np.array([ int(i) for i in sys.argv[9].split(",")])

        src_img_p3 = np.array([ int(i) for i in sys.argv[10].split(",")])
        dest_img_p3 = np.array([ int(i) for i in sys.argv[11].split(",")])

        src_img_p4 = np.array([ int(i) for i in sys.argv[12].split(",")])
        dest_img_p4 = np.array([ int(i) for i in sys.argv[13].split(",")])

        matrix = transform(n, dest_img_p1, dest_img_p2, dest_img_p3, dest_img_p4, src_img_p1, src_img_p2, src_img_p3, src_img_p4)
        #printing the transformation matrix obtained.
        print("Transformation matrix\n",matrix)

        inverse_tm = inv(matrix)
        transformed_image = Image.fromarray(warp(src_image, inverse_tm))
        transformed_image.save(output_image)


    if part_number == "part3":
        """
        
        Commands for different images pairs

        1. python a2.py part3 part3-test/scene1.jpg part3-test/scene2.jpg op.jpg
        2. python a2.py part3 part3-test/building2.jpg part3-test/building1.jpg op.jpg
        3. python a2.py part3 part3-test/parking1.jpg part3-test/parking2.jpg op.jpg
        4. python a2.py part3 part3-test/coffee2.jpg part3-test/coffee1.jpg op.jpg
        5. python a2.py part3 part3-test/chips_ahoy2.jpg part3-test/chips_ahoy1.jpg op.jpg
        6. python a2.py part3 part3-test/house1.jpg part3-test/house2.jpg op.jpg
        7. python a2.py part3 part3-test/lap1.jpg part3-test/lap2.jpg op.jpg
        
        
        """
        print("Starting Part 3:")

        # part3 scene1.jpg scene2.jpg
        image_name1 = sys.argv[2]
        image_name2 = sys.argv[3]

        image1 = Image.open(image_name1)
        image1 = np.asarray(image1)

        image2 = Image.open(image_name2)
        image2 = np.asarray(image2)
        
        # call the orb descriptor function to get the orb descriptors and point correspodences.
        points_image1, points_image2 = orb_descriptor(image1,image2)
        #print("final_points",points_image1,points_image2)
        print("length of final_points",len(points_image2))
        
        """
        RANSAC:
        Following links were reffered to write the RANSAC for Image homography calculation.
        https://stackoverflow.com/questions/71282441/im-working-on-image-stitching-i-cant-figure-this-error-out-in-homography-usin
        https://file.scirp.org/Html/40004.html
        https://cs.stackexchange.com/questions/70434/how-is-ransac-used-to-estimate-a-homography-given-the-descriptors-of-both-images
        """
        s = 4 # number of samples or point correspondences for homography.
        max_inliers = 0
        best_points = []
        best_transformation = None
        for i in range(10000):
            inliers = 0
            sample_indices = list(np.random.randint(0, len(points_image1), s))
            # print(sample_indices)
            # print(points_image1[sample_indices[0]])
            img1_p1, img1_p2, img1_p3, img1_p4 = [points_image1[i] for i in sample_indices]
            img2_p1, img2_p2, img2_p3, img2_p4 = [points_image2[i] for i in sample_indices]

            try:
                # find the projective trasformation
                matrix = transform(4, img2_p1, img2_p2, img2_p3, img2_p4, img1_p1, img1_p2, img1_p3, img1_p4)
            except:
                print("singular matrix. \n------ skipping-------\n")
                continue

            # finding the number of inliers for the above projective transformation
            for i in range(len(points_image1)):
                pt1 = np.array([points_image1[i][0], points_image1[i][1], 1])
                pt_ = np.dot(matrix, pt1)
                pt_ = pt_ / pt_[2]
                # print(pt_)

                pt2 = np.array([points_image2[i][0], points_image2[i][1], 1])
                # find euclidean distance between the transformed point and the actual point from the descriptor
                # print("dist",np.sqrt(np.sum(np.square(pt_-pt2))))
                if np.sqrt(np.sum(np.square(pt_ - pt2))) < 0.75:
                    inliers += 1
            print(inliers)
            if inliers > max_inliers:
                print("------here------!!!!!!!!!!!!!!!")
                best_transformation = deepcopy(matrix)
                max_inliers = inliers
        """End of reffered code"""
        print(max_inliers)
        print(best_transformation)

        inverse_tm = inv(matrix)
        transformed_image = Image.fromarray(warp(image1, inverse_tm))
        #transformed_image.save("lets-see.jpg")

        #We need the best transformation matrix and the inv transformation matrix.
        stitched_img=letsStitch(image1,image2,best_transformation,inverse_tm)
        cv2.imwrite(sys.argv[4], stitched_img)
