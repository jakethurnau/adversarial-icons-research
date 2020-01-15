# adversarial-icons-research

# Installing Tensorflow-GPU

The majority of this tutorial will be referencing a YouTube tutorial on installing Tensorflow GPU for windows 10: https://www.youtube.com/watch?v=KZFn0dvPZUQ

I would like to first mention that it is entirely possible to run Tensorflow primarily using a CPU. However, the processing speed will be greatly reduced compared to utilizing a GPU, so I will only be talking about the Tensorflow-GPU installation.

That being said, Tensorflow-GPU requires a CUDA compatible graphics card in order to run, this means if you do not have a compatible graphics card you will not be able to run Tensorflow otherwise.

GPU compatibility with CUDA
If you do not have a Nvidia GPU or you are unsure if your graphics card is compatible this site will tell you if your graphics card is compatible or not: https://developer.nvidia.com/cuda-gpus

Visual Studio
Once you've determined that your graphics card is compatible with CUDA, you need to install the newest Visual Studio using this link: https://visualstudio.microsoft.com/downloads/

I installed Visual Studio 2019 using the community package

Installing CUDA
Before installing CUDA, we have to make sure we're using the most up to date version with Tensorflow using this link: https://www.Tensorflow.org/install/gpu

At the time of this tutorial, CUDA version 10.0 is the recommended version to install: https://developer.nvidia.com/cuda-10.0-download-archive

# installing cuDNN

Once the correct version of CUDA is installed we have to install cuDNN using this link: https://developer.nvidia.com/rdp/cudnn-download

The site will initially ask you for to login, this means you will have to make a Nvidia account if you do not already have one. Once the account is made, you can login and access the downloads section.

Select (Download cuDNN v7.6.2 (July 22, 2019), for CUDA 10.0) Including (cuDNN Library for Windows 10) under the drop down menu.

Once you have the package downloaded, you're going to need to unzip the .rar file and be ready to copy the contents over to your current CUDA directory

For me, CUDA was found at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0

Once you're in the v10.0 folder, you're going to want to copy the bin, include, and lib folder from your CUDA folder (extracted from the zip) into the v10.0 folder and overwrite the files present.

# Setting up Environment Variables

The next step is to setup the necessary paths in environment variables so Tensorflow knows where to look for all the CUDA dependencies.

In the windows search bar you can type “envi” and a prompt to "Edit the system environment variables" should show up, click that.

In System Properties, go down to the bottom and click Environment Variables.

Under User Variables, you're going to want to click Path and edit it. We're going to be adding two new paths for the purpose of this tutorial.

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp
You will also need to add a new path called PYTHONPATH 

When you've added these paths, click Ok on all corresponding windows

# Prerequisites and Dependencies

At this point it is assumed you already have Python installed and will not be covered in this tutorial. To keep up with consistency I recommend you install version 3.7 as this is the version I will be using. I also highly recommending that if you do not already to have the to your current python version added to the Path section in Environment Variables like we previously did with CUDA.

C:\Users\Jake Thurnau\AppData\Local\Programs\Python\Python37
C:\Users\Jake Thurnau\AppData\Local\Programs\Python\Python37\Scripts
(example of my python directories in Path)

This allows you to be able to run any python script in the cmd window in any directory.

Going forward now, all prerequisites will be installed using pip, so it is highly recommended to have the newest version of pip to be able to run these cmds in your cmd prompt window.

All below pip installs are required both to run Tensorflow as well as run the scripts in this project

pip install --ignore-installed --upgrade Tensorflow-gpu

pip install keras

pip install pandas

pip install numpy

pip install pillow

pip install lxml

pip install Cython

pip install contextlib2

pip install jupyter

pip install matplotlib

pip install opencv-python

pip install pywin32


# Running the programs

To use this program, an image is first fed into a script to perturb the image. To create a perturbed image using image processing techniques such as gaussian or poisson noise, the noisy.py script is used. This script tells the user to choose a background image, a web browser icon to perturb, makes the perturbations, and then pastes the perturbed icon back into the background image. 

To create adversarial icons using the FGSM method, an input image is fed into the adversarial1.py script. After this, the noisy.py script is run and the user chooses a background image, and then chooses any browser icon, followed by the option for an adversarial image. This choice pastes the FGSM adversarial icon into the background image using the image created from the adversarial1.py script. 

After creation of the perturbed images, the image is run thru the Multi_Object_Detect_Demo.py script to start the object detection model using the image with the perturbed icon in it. The model will then attempt to predict the icon and show a confidence level of recognition. 
