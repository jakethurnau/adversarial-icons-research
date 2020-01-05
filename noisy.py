#Python code to add noise to an image
# Author: Jake Thurnau
# Date: 9/14/19
# Description:
# This program takes an image and adds different types of noise to it
#Parameters
#----------
#image : ndarray
#    Input image data. Will be converted to float.
#mode : str
#    One of the following strings, selecting the type of noise to add:

#    'gauss'     Gaussian-distributed additive noise.
#    'poisson'   Poisson-distributed noise generated from the data.
#    's&p'       Replaces random pixels with 0 or 1.
#    'speckle'   Multiplicative noise using out = image + n*image,where
#                n is uniform noise with specified mean & variance.


import numpy as np
import os
import cv2
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt



image_string = (input("Please choose an image to add noise to: \n"))
image = cv2.imread(image_string)
icon = input(
    "Will you be adding noise to Chrome, Edge, Opera, or Firefox? " +
    "Type 1 for Chrome, 2 for Edge, 3 for Opera, 4 for Firefox:\n")

if icon == "1":
	icon = cv2.imread('google-chrome-icon-small.png')
elif icon == "2":
	icon = cv2.imread('microsoft_edge_icon.png')
elif icon == "3":
	icon = cv2.imread('opera_icon.png')
elif icon == "4":
	icon = cv2.imread('firefox_icon.png')

#image = Image.open(image_string)
#chrome = Image.open('google-chrome-icon-small.png')   

#image.show()
img_arr = np.array(icon)
#print (img_arr)
#print (img_arr.shape)
noise_type = input(
    "Will you be adding Gaussian noise, salt-and-pepper noise, poisson noise, speckle noise, or random noise " +
    "speckle noise? Type 1 for Gaussian, 2 for salt-and-pepper, 3 for poisson, 4 for speckle, 5 for random, 6 for adversarial or 7 for none:\n")

toggle = True

count = 1
fileName = 'noisyImage.jpg'

if noise_type == "1":	#Gaussian Noise
	row,col,ch = img_arr.shape
	mean = 0
	var = 4	#edit this to change amt of noise
	sigma = var#**0.5
	#sigma = 4
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	#gauss = Image.fromarray(gauss, 'RGB')
	#gauss.show()
	#gauss = gauss.reshape(row,col,ch)
	#noisy = img_arr + gauss*0.007
	#blur = cv2.GaussianBlur(chrome,(5,5),0)
	#noisy = np.clip(noisy, 0, 1)
	#print(noisy)
	#print(noisy.shape)
	#noisy_img = Image.fromarray(noisy, 'RGB')
	noisy_img = icon + gauss
	cv2.imwrite('noisyIcon.jpg', noisy_img)
	x_offset=y_offset=50
	image[y_offset:y_offset+noisy_img.shape[0], x_offset:x_offset+noisy_img.shape[1]] = noisy_img
	cv2.imwrite(fileName, image)
	#image.paste(noisy_img, (200,50))
	#image.save(fileName, "JPEG")
	
elif noise_type == "2":	#Salt and Pepper Noise
	#image = Image.open(image_string)
	#chrome = Image.open('google-chrome-icon-small.png')
	img_arr = np.array(icon)
	row,col,ch = img_arr.shape
	s_vs_p = 0.5
	amount = 0.1	#edit this to change amt of noise
	out = np.copy(img_arr)
	# Salt mode
	num_salt = np.ceil(amount * img_arr.size * s_vs_p)
	coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in img_arr.shape]
	out[coords] = 1
	# Pepper mode
	num_pepper = np.ceil(amount* img_arr.size * (1. - s_vs_p))
	coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in img_arr.shape]
	out[coords] = 0
	#noisy_img = Image.fromarray(out, 'RGB')
	noisy_img = out
	x_offset=y_offset=50
	image[y_offset:y_offset+noisy_img.shape[0], x_offset:x_offset+noisy_img.shape[1]] = noisy_img
	cv2.imwrite(fileName, image)
	cv2.imwrite('noisyIcon.jpg', noisy_img)
	#image.paste(noisy_img, (200,50))
	#image.save(fileName, "JPEG")
	#noisy_img.save(fileName, "JPEG")
elif noise_type == "3":	#Poisson Noise
	vals = len(np.unique(img_arr))
	vals = 2 ** np.ceil(np.log2(vals))
	#noisy = np.random.poisson(img_arr * vals) / float(vals)
	noisy = np.random.poisson(100, icon.shape) #edit this to change noise level
	noisy_img = noisy + icon
	x_offset=y_offset=50
	image[y_offset:y_offset+noisy_img.shape[0], x_offset:x_offset+noisy_img.shape[1]] = noisy_img
	cv2.imwrite(fileName, image)
	cv2.imwrite('noisyIcon.jpg', noisy_img)
	#noisy_img = Image.fromarray(noisy, 'RGB')
	#noisy_img.save(fileName, "JPEG")
	#image.paste(noisy_img, (200,50))
	#image.save(fileName, "JPEG")
elif noise_type =="4":	#Speckle Noise
	#image = Image.open(image_string)
	#chrome = Image.open('google-chrome-icon-small.png')
	#img_arr = np.array(chrome)
	row,col,ch = img_arr.shape
	noise = np.random.randn(row,col,ch)
	#noise = noise.reshape(row,col,ch)        
	#noisy = img_arr + img_arr * noise
	noisy_img = icon + icon*noise
	x_offset=y_offset=50
	image[y_offset:y_offset+noisy_img.shape[0], x_offset:x_offset+noisy_img.shape[1]] = noisy_img
	cv2.imwrite(fileName, image)
	cv2.imwrite('noisyIcon.jpg', noisy_img)
	#noisy_img = Image.fromarray(noisy, 'RGB')
	#noisy_img.save(fileName, "JPEG")
	#image.paste(noisy_img, (200,50))
	#image.save(fileName, "JPEG")
elif noise_type == "5":	#Random distributed noise
	noise = np.random.randint(50, size = img_arr.shape, dtype = 'uint8')
	noisy_img = icon + noise
	x_offset=y_offset=50
	image[y_offset:y_offset+noisy_img.shape[0], x_offset:x_offset+noisy_img.shape[1]] = noisy_img
	cv2.imwrite(fileName, image)
	#print(noisy)
	#print(noisy.shape)
	#noisy_img = Image.fromarray(noisy, 'RGB')
	#noisy_img.save(fileName, "JPEG")
	#image.paste(noisy_img, (200,50))
	#image.save(fileName, "JPEG")
elif noise_type == "6":
	image = Image.open(image_string)
	noisy_img = Image.open('operaResize.jpg')
	#noisy_img = np.asarray(noisy_img)
	#np.resize(noisy_img,(48,48,3))
	#x_offset=y_offset=50
	#image[y_offset:y_offset+noisy_img.shape[0], x_offset:x_offset+noisy_img.shape[1]] = noisy_img
	image.paste(noisy_img, (0,0))
	image.save('adversarialImageTest.jpg')
elif noise_type == "7":
	image = Image.open(image_string)
	chrome = Image.open('google-chrome-icon-small.png')
	edge = Image.open('microsoft_edge_icon.png')
	firefox = Image.open('firefox_icon.png')
	opera = Image.open('opera_icon.png')
	#noisy_img = np.asarray(noisy_img)
	#np.resize(noisy_img,(48,48,3))
	#x_offset=y_offset=50
	#image[y_offset:y_offset+noisy_img.shape[0], x_offset:x_offset+noisy_img.shape[1]] = noisy_img
	image.paste(chrome, (0,0))
	image.paste(firefox, (0,310))
	image.paste(edge, (310,0))
	image.paste(opera, (310,310))
	image.save('unperturbedImageTest.jpg')
count = count + 1

