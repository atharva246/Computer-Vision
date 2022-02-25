
### Grading: <br>
The implementation of grade.py is explained below.<br>     
**Approach:**<br>
The overall apporach uses Hough Transform and other image manipulation techniques for finding the answers marked by the students. We perform all following operations on an inverted image.<br>
Hough Transform is used to find the angle by which the images is skewed or tilted. The images is then straightened by re-rotating it in the opposite direction. We have cropped the image to just retain the part where the questions are to be marked and then divided into 3 parts vertically (columns of questions).<br>
On every part of the image we find a reference point from where the option boxes start for option "A". After aquiring the reference point we can then segment the option boxes for each of the questions. <br>
After segmenting the five option boxes for a specific question, we check the number of white pixels in all five boxes. The boxes which have the count above some threshold are bubbled and are then sent to the output. We also check the extra spaces by checking the area before the reference point excluding the part where question number is mentioned. The number of pixels excluded for the qiestion number was tuned using trial and error.<br> We have considered that the box is around 37 * 37 pixels. This number was found using trial and error and by testing on all images.<br>
There are many hyperparamters which are used in this code, which are hard coded, were tuned by trial and error and checking the number of pixels in Paint application. This includes the size of the boxes, the horizontal distance between the boxes, the vertical distance between the boxes, the distance between the columns of questions and the blank part where students may write thier answers.

Below image shows how we have segmented the image to get option boxes.<br>
![Alt text](https://media.github.iu.edu/user/18152/files/7cddd496-cf72-4183-85cd-6c63719d214d)

Edge Detection: At first we were thinking of using edge detection. But edge detection is producing double lines at the edges of the boxes which was giving difficulty while segmenting the option boxes. As images that are given in test images folder are already gray scale, we decided not to use edge detection at all.

**Discarded Approach:** <br>
We had decided to use Hough transform's output which would be lines detected on the image, and use those lines in segmentation of the boxes. After discussing this approach with Xizi she suggested that this approach would not be robust and we should try something which will segment the image into boxes. Thats how we came up with our current approach.

**Testing:**<br>
1) To test the grade.py, we ran the code on all given images as they were and checked the output.
2) We rotated the images by positive or negetive angle and then ran the code to test of our code is robust toward skewed images.

**Challenges:**<br>
1) Understanding and Implementing Hough Transform
2) Trying to draw the lines detected by hough transform to further detect the answers. Ended up discarding this approach. 
3) Handling the extra line above the questions and boxes part in some images while segemntation. (images b-13 and a-48)

**Report on results:**<br>
Overall we are getting pretty correct outputs in the text file for grade.py.<br>
For some images, which require re-rotation, for a few questions, we are getting false positives for the "x" which is found out by checking if students have written anything before the question number. So sometimes, when in reality students ahve no written anything in that part, yet we are getting the  "X" along with detected answers. Have observed this behavior for following images a-3, c-33.<br> 
For image a-30, I purposefully rotated the image by 5 degrees and then ran my code. The code detected the 5 degree roattion and straightened the image. But I got many false positives with "x" in output. But the options marked were detected correctly.
For images b-13 and a-48, the box above the question and boxes section is creating a problem when there is a rotated image as input. We have handled that while getting the reference point for some cases.

### Injection :
**Approach:**<br>
As the forms will be printed, filled by the students and then scanned again, we came up with a robust solution which will keep the injection consistant through the entire cycle. Scanning the 'filled' forms could potentially lead to some tilted images. We used Hough Transform here as well, to find the skew or tilt of the scanned images and then rotated them in the opposite direction to get straightened images. (same as executed in grade.py). 
<br><br>
Our code takes in a '.txt' containing all the answers. We are storing these answers in a python dictionary where question numbers are the keys and the answers are the values to those respective keys.<br> Now coming to the injection part, we are using a piece of the image from the left hand top corner to inject the answers. Keeping the relative position constant for all options (A,B,C,D,E), we iterate through the entire block (100 x 1500 pixels). Initial pixel value for all pixels inside this block is 255 as it just a white box. For every extracted answer from the dictionary, we changed the pixel values in a small 10x10 pixel box to 0. This gave us a black 10x10 box for every question. For questions having multiple answers, we changed the pixel values of multiple 10x10 boxes to 0.
<br>We also added a small 10x10 pixel box of 0 values,before question 1, to create a reference point during the extraction process. 
<br><br>Below is an example of this injection process:
<br>![Alt text](https://github.iu.edu/cs-b657-sp2022/athakulk-sparanjp-bmcshane-a1/blob/main/injection_output/injected_box.jpg)


**Discarded Approaches:**<br>
While looking for relevant approaches to inject data into an image, we came across the 'Least Significant Bit' (LSB) encryption. It looked like an amazing way to go forward with, so we implemented it. "Works fine guys, yay! Let's save it in .jpg! We've cracked this right?". WRONG. It did **not** work fine. All the encryption disappeared as .jpg is a compressed file format. LSB only worked well with .png files. 
<br>
**Testing:**<br>
1) To test the inject.py, we ran the code on all given images as they were and checked the output.
2) We rotated the images by positive or negetive angle and then ran the code to test of our code is robust toward skewed images.

**Challenges:**<br>
1) Implementing a robust injection technique.
2) Figuring out complexities of saving and reloading .jpg images.

**Report on Results:**<br>
We successfully injected the answers on the given form and the output was pretty cryptic.
Saving the images as .jpg still caused some problems on the extraction end, but were handled efficiently.
To sum up, we created a robust injection technique which can handled rotated images as well. 


### Extraction :
Extraction was essentially just a matter of reverse engineering the injection process. Injection wasn't perfect, a few of the pixels weren't 0 where they should have been. We're thinking it has something to do with the saving/loading the .jpg file process. To play it safe, we had to take the average pixel value over the range of injected pixel values as they corresponded to each possible A/B/C/D/E, and check to make sure the average was below a chosen threshold value before recording the answer. Like with injection, we first had to rotate the image as dictated by our Hough transformations to make sure our injected answers had the correct orientation regardless of how they were scanned in. From there, we simply cropped away the part of the image that didn't have the answers injected on it, scanned to find the reference point at the top of the injection, and then iterated over the injection and recorded the answers.
