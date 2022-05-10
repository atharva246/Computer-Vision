# Part 1: Image matching and clustering


## Problem Definition
In this problem, we had to find matches between the pairs of images given and use a clustering algorithm to cluster the images that have the most matches among themselves and to find the accuracy of the clustering algorithm. 

## Function Definition
For this part, we have written a function:image_matching_and_clustering() which takes 3 arguments: 

1. images, which is an array consisting of numpy arrays of each image
2. arr, which is the list of images present
3. k, number of clusters given 

This function will take these parameters and cluster the images given according to a clustering algorithm(Agglomerative Clustering in this case) and give us clusters of images.
There are 2 parts to this function: image matching and clustering. 

## Image Matching
Image matching is a technique where we try and find similar "features" of 2 images and match them.
There can be multiple images of the same object taken but they might have different orientations, different lighting, and even different scale.  Ultimately, they are the images of the same object and should be categorized as such.

1. Initially, we created a matrix of dimensions number of images by the number of images. This matrix will store the number of matches for each pair of images. 
   
2. We then used https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html as a reference to do Brute-Force Matching with ORB descriptors. We create the orb object for both images to detect the key points and descriptors of each image. Then we will use the BFMatcher function with Hamming Distance as the distance metric to find the matches between the 2 images. We used the bf.knnMatch function to find the closest k matches or keypoints, in this case, we set k=2, so that we can use the Lowe ratio test to use only the good matches. (https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94).

According to the Lowe test, if the ratio of the distance between the closest and second closest match is below a certain threshold, we can consider that to be a good match, otherwise, we eliminate those key points.

## Image Clustering

1. After finding the good key point matches for each pair of images, we performed clustering on the matrix obtained. We used the AgglomerativeClustering library function to perform this operation. (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html). We tried various affinity parameter values and linkages, and cosine affinity and complete linkage Agglomerative clustering gave us the best results. We obtain the image and the cluster number it belongs to in the result for each image. 
   
2. Once we obtain the results of the clustering, we checked the Pairwise Clustering Accuracy. To do this, we stored the results of clustering in 2 dictionaries. The first dictionary contained the cluster number as the key and the images that belonged to those clusters as the list of values for that key, and the second dictionary contained the image name as the key and the corresponding cluster as the value. 
   
3. To check the Pairwise Clustering Accuracy, we checked if the images of the same monument or object are in the same cluster or not. To do this, we took each pair of images and checked if they are of the same monument. If the images are of the same monument and belong in the same cluster, then that's a true positive case. If the images are of a different monument and belong in different clusters, then that's a true negative case. We carried out the calculation by adding the true positive and true negative cases and dividing them by the total number of possible pairs. We then printed this accuracy. 
   
4. To store the clusters in a text file, we just used the first dictionary which had the cluster numbers and images corresponding to that cluster. We wrote each cluster into the file line by line. 

## Sample Cases

For all the images in the given dataset, if we set the number of clusters to 18, we get the following result.


![Successful Case with k=18](./part1-special-cases/good_case_code.png)

We get a Pairwise Clustering accuracy of 85.36, which is the highest accuracy that we achieved by setting the number of clusters to twice the number of monuments used for the dataset. 


The output file that we obtain is: 

![Successful Case with k=18](./part1-special-cases/good_case.png)


For all the images in the given dataset, if we set the number of clusters to 10, which is the number of monuments used for the dataset, we get the following result.


![Successful Case with k=10](./part1-special-cases/good_case_code_2.png)


The output file that we obtain is: 

![Successful Case with k=10](./part1-special-cases/good_case_2.png)

We get a Pairwise Clustering accuracy of 79.03225806451613 for this case. 


## Limitations


### 1. Failure Cases
During the calculation of the clusters, for certain cases, there were failures. For example, for this case: 

![Failure Case 1](./part1-special-cases/failure_case_code.png)

the accuracy came out to be only 47.61%. There were 0 true positives and 20 true negatives. 


![Failure Case 1](./part1-special-cases/failure_case_1.png)

As you can see, despite the bigben_8.jpg, bigben_7.jpg, and bigben_2.jpg being images of the same monument, they are in different clusters. This failure may occur because of the python Agglomerative Clustering function being used with different parameters. It may also be possible that the images, despite being of the same monument, might not have many matching points as expected due to their different orientations, lighting, and angles. 

Here are some other cases where the accuracy is much lower than expected and the clusters formed are different from what is expected. 


![Failure Case 2](./part1-special-cases/failure_case_code_2.png)


![Failure Case 2](./part1-special-cases/failure_case_2.png)


![Failure Case 3](./part1-special-cases/failure_case_code_3.png)


![Failure Case 3](./part1-special-cases/failure_case_3.png)


![Failure Case 4](./part1-special-cases/failure_case_code_4.png)


![Failure Case 4](./part1-special-cases/failure_case_4.png)

In all these cases, the images that should belong in the same cluster are not actually in the same cluster. It may be possible that the key points that were matched using Agglomerative Clustering might not have been the right ones or the images might have different angles or orientations, making it difficult to match the features. 


# Part 2: Image transformations

### Image Warping:
Image warping means that the image is transformed using a 3x3 Transformation matrix.
We have written a function warp() which takes an image which is to be transformed and the transformation matrix as its arguments.
We have used bilinear interpolation along with inverse warping as specified in the assignemnt. Using inverse warping aviods holes in the warped image and bilinear transformation helps smoothen the image.
To write the code for bilinear transformation, I watched the following video https://youtu.be/UhGEtSdBwIQ  and followed its steps.
To test this function we tested it on the given "lincoln.jpg" image to confirm if warp() worked correctly.
![Alt text](https://media.github.iu.edu/user/18152/files/98ca624d-98af-4485-96c9-cce2380899dd)

<br> 
### Finding the transformation matrix from given correspondenses:

The results for transformation on the book image from the assignemnt are:<br>

**1) Translation n=1:** <br>
Translation uses only one pair of point correspondence between 2 images.<br>
![Alt text](https://media.github.iu.edu/user/18152/files/ad69aaa2-04ff-4ad2-b01c-614485789c21)
<br>


**2) Euclidean n=2:**<br>
Euclidean transformation consists of Translation as well as Rotation.
The transformation matrix here will be of the following format<br>
![Alt text](https://media.github.iu.edu/user/18152/files/a4d4c7b4-c1c5-44c9-805d-8cad0be177ee)

Here a and b are the cosins(theta) and sine(theta) terms where theta is the angle of rotation. c and d are the translations.
I used above matrix along with point correspondenses to find linear equations and solved the linear system of equations to find the 4 unknowns.<br>
![Alt text](https://media.github.iu.edu/user/18152/files/a2dc1e72-ccb2-4f40-b8c0-78a64123c766)
<br>


**3) Affine n=3:**<br>
There are 6 unknowns to find in the transformation matrix for affine transformation.<br>
![Alt text](https://media.github.iu.edu/user/18152/files/7b34e7cc-418c-4e75-9835-6cf9cd7e1430)<br>
I used above matrix along with point correspondenses to find linear equations and solved the linear system of equations to find the 6 unknowns.<br>
![Alt text](https://media.github.iu.edu/user/18152/files/dd09c93b-79ab-458e-84f5-a94946e1c17d)

**4) Projective n=4:**<br>
There are 9 unknowns in the transformation matri for projective transformation. <br>
I used the method shown by the professor in one of the ppts to solve the projective transformation.<br>
I used aove matrix along with point correspondenses to find linear equations and solved the linear system of equations to find the 4 unknowns.<br>
![Alt text](https://media.github.iu.edu/user/18152/files/9088ca4c-7bf0-4d6e-82fb-bbc046a0bfcf)

<br>
We also tested the transformations on the building image which was in the assignment named as scene1 and scene2. 

![Alt text](https://media.github.iu.edu/user/18152/files/6bcf1296-ab52-449e-8797-2d02efe0c373)


The results from those images are as follows.<br>
We tested both types of transformations which are as follows:
The corresponding point matches of the above images were manually found out using Paint.<br>
1) Transforming scene2 as per scene1:<br>
Following command was run to test this.<br>
python a2.py part2 4 scene2.jpg scene1.jpg scene_output1.jpg 476,243 220,328 449,246 192,332 700,73 442,158 671,363 414,449<br>
![Alt text](https://media.github.iu.edu/user/18152/files/6ef2fa5c-f2a5-4ef8-bc2f-6f1da09bf6cd)


2) Transforming scene1 as per scene2:<br>
Following command was run to test this.<br>
python a2.py part2 4 scene1.jpg scene2.jpg scene_output2.jpg 220,328 476,243 192,332 449,246 442,158 700,73 414,449 671,363<br>
![Alt text](https://media.github.iu.edu/user/18152/files/18806cc7-9be0-481c-9456-04d59b7c523a)




## Limitations:
The code fails to tranform the image correctly if the corresponding points have errors.
We tried to find corresponding points manually on a high resolution image but there are some manual errors which are introduced. This results in a poorly transformed image.
The following is one such example.
python a2.py part2 4 src.jpg dest.jpg bhutan.jpg 167,801 617,1133 723,1693 1057,1930 725,2075 1019,2391 335,141 843,437<br>
![Alt text](https://media.github.iu.edu/user/18152/files/cb85d8e9-6060-4452-8901-d6d2b07aead4)


## Challenges: <br>
Struggled to figure out how to calculate matrix for n=2, n=3 and n=4.<br>
Had to manually solve and find linear equation system for n=2 and n=3<br>

# Part 3: Automatic image matching and transformations

## Problem Description: 
This part consists of 4 parts:
1) Extract interest points from each image
2) Figure the relative transformation between the images by implementing RANSAC
3) Transform the images into a common coordinate system
4) Blend the images together

## Step 1: Extract interest points from each image
This step uses the ORB detector. ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. According to 'ORB: An efficient alternative to SIFT or SURF', a paper written by Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary Bradski (https://ieeexplore.ieee.org/document/6126544), ORB works better and faster than SIFT and that is why we chose to use the ORB descriptor in our approach. 
Our ORB function gives us the matching points in the 2 images. An example of this matching is given below.

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/image.jpg)

## Step 2: Figure the relative transformation between the images by implementing RANSAC
Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates. Therefore, it also can be interpreted as an outlier detection method.
An advantage of RANSAC is its ability to do robust estimation[3] of the model parameters, i.e., it can estimate the parameters with a high degree of accuracy even when a significant number of outliers are present in the data set. A disadvantage of RANSAC is that there is no upper bound on the time it takes to compute these parameters (except exhaustion).
### Algorithm RANSAC: 
1) Select randomly the minimum number of points required to determine the model parameters.
2) Solve for the parameters of the model.
3) Determine how many points from the set of all points fit with a predefined tolerance.
4) If the fraction of the number of inliers over the total number points in the set exceeds a predefined threshold t, re-estimate the model parameters using all the identified inliers and terminate.
5) Otherwise, repeat steps 1 through 4 (maximum of N times).

All of the above steps return us a relative transformation matrix which will be useful in the next step.

## Step 3: Transform the images into a common coordinate system
Using the transformation matrix given by RANSAC, we can now bring the both the images in the same coordinate plane. This step is crucial as the transformation will help us sticth both the images together and if they are in different coordinate planes, common image pixels will never overlap. 
An example of this procedure is given below.

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/transformation%20image.jpg)

## Step 4: Blend the images together
This step involves blending the destination imagage with the Transformed Source Image. 
Some successfull blends are shown below

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/scene_stitching.jpg)

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/parking_sticth.jpg)

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/laptop_stitch.jpg)

SOURCE AND DESTINATION IMAGE CREDITS: Arpita Welling (aawellin.iu.edu)
![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/vada_stitch.jpg)

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/chips_stitch.jpg)

## Limitations
We have some failures which are shown below:

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/house_fail.png)

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/building_fail.jpg)

The code fails in such situations because of 2 main reasons(according to us):
1) The RANSAC algorithm doesn't pick the optimal transformation matrix, as we randomly pick the starting points, which can be used for stitching.
2) Inability of the ORB descriptor to get enough inliers which is the primary requiremnt for our ransac algorithm.

We also found some images which did the stitching right but had some duplications in the background. for example:

![Alt text](https://github.iu.edu/cs-b657-sp2022/sdabhane-sparanjp-athakulk-a2/blob/main/report_images/coffee.png)

Again, we think that this could be becasue of the reasons mentioned above mentioned. 

## Problems Faced

Alot of problems. <br>
We did manage to write the RANSAC algorith which takes in the orb descriptors and returns a transformation matrix but tuning the distance(ORB descriptor) and inlier distance(RANSAC) thresholds was a tedious research process. we kept getting wrong transformation matrices so we decoded every step of the RANSAC to refactor in ways that our transform function(from part 2) got the right parameters.<br>
We went through multiple iterations for the stitching code but nothing was quite fruitful. So we referred a stitching methodology from https://github.com/melanie-t/ransac-panorama-stitch/blob/master/src/PanoramaStitching.py and refactored it according to our needs. This methodology used a function called 'cv2.getRectSubPix' which is used for bilinear interpolation. We weren't sure if we were allowed to use this so really tried hard to write our own homegrown method (which we have included in our code for your reference) but it failed so we had to resort on using 'cv2.getRectSubPix'.


## Contribution of Authors:<br>
Part 1: Atharva Kulkarni and Shardul Dabhane<br>
Part 2: Sanika Paranjpe<br>
Part 3:<br>
   RANSAC : Sanika Paranjpe<br>
   Image Stitching: Atharva Kulkarni and Shardul Dabhane
