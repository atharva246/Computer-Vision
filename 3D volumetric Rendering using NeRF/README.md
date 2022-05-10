# NeRF Implementation

Atharva Kulkarni, Jaydeep Chauhan 

## 1. Overview
This is a pytorch-lighning implementation of paper [NeRF](https://www.matthewtancik.com/nerf). We experimented on the synthetic blenders dataset and try to reproduce the results for all 8 scene objects by training the model for 125K iterations and 200K iterations(only on lego and hotdog), we calculated the PSNR(Peak signal to noise ratio) as a metric to evaluate the generate 3D view quality and compare it with other contemporary methods. For training we used IU's carbonate gpu cluster.

## 2. Steps for training
#### Training :-
- First download the synthetic blender dataset from here : [Blender Dataset](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) and extract the dataset.     
- Now to train the model first replace the ```dataset_path``` with your local path on your machine.
- Set the ```IMG_DIMENSION``` to (400, 400).
- All other parameters are adapted from the original paper, but you can adjust the parameter according to your preference.
- Train your model using  ```python train.py``` on local machine.
- During training, you can view the training loss and generated 3D view using tensorflow logger on [localhost](localhost:6006)
- To train the model, you might need to run it on ```carbonate``` machine using ``sbatch job.sh``.

#### Testing :-
- After training you can see the folder names logs inside this NeRF folder.
- Inside that folder there would be checkpoints folder and tfevent file.
- Set the ```root_dir``` path of the dataset scene object and ```checkpoint path``` in the ```eval.py```.
- After evaluate your model using ```python eval.py```
- It will generate a recursive folders in this fashion ```results/{dataset_name}/{scene_name}``` to store the final visualizations.

## 3. Results

![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/chair.gif) ![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/chair_depth.gif)

![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/drums.gif) ![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/drums_depth.gif)

![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/ficus.gif) ![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/ficus_depth.gif)

![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/hotdog.gif) ![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/hotdog_depth.gif)

![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/lego.gif) ![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/lego_depth.gif)

![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/mic.gif) ![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/mic_depth.gif)

![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/ship.gif) ![Alt Text](https://github.com/Jd8111997/NeRF/blob/main/Results/ship_depth.gif)


## 4. References
- https://www.matthewtancik.com/nerf
- https://github.com/bmild/nerf
- https://github.com/kwea123/nerf_pl
- https://github.com/krrish94/nerf-pytorch
- https://github.com/yenchenlin/nerf-pytorch/
- https://pyimagesearch.com/2021/11/17/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-2/
- https://dellaert.github.io/NeRF/
