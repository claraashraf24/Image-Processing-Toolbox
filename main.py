from ast import Break
from cProfile import label
from configparser import Interpolation
from importlib.resources import path
import sys
import math
import matplotlib.pylab as plt
import cv2
from turtle import width
from skimage import color
from skimage import io
import pathlib
import numpy as npy
import numpy as np
import os
import magic
import PIL
import scipy.fftpack

from PyQt5.QtWidgets import QLabel, QFileDialog, QApplication, QWidget, QGraphicsView
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.image as img
import pydicom as dicom
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import pydicom.data

from PyQt5.QtWidgets import QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui
import random
import skimage as skit
from skimage import transform, data
#from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


from PyQt5 import QtWidgets, uic




class MainWindow(QtWidgets.QMainWindow):


 def __init__(self, *args, **kwargs):
        #Load the UI Page
        super(MainWindow, self).__init__(*args, **kwargs)
       
        uic.loadUi('mainwindow.ui', self)
        self.open_button.triggered.connect(self.browse_files)
        self.label = self.findChild(QLabel, "label")
        self.size = self.findChild(QLabel, "size_label")
        self.zoom_button.clicked.connect(self.nearest_neighbour)
        self.linear_interpolation.clicked.connect(self.linear_zoom)
        self.rotation_button.clicked.connect(self.rotate_image)
        self.generate_button.clicked.connect(self.generate_t)
        self.shear_button.clicked.connect(self.shear_image)
        self.histogram_browse_button.clicked.connect(self.draw_normal_histogram)
        self.filter_browse_button.clicked.connect(self.browse_for_filtering)
        self.unsharp_button.clicked.connect(self.unsharp_filter)
        self.median_button.clicked.connect(self.median_filter)
        self.salt_button.clicked.connect(self.salt_pepper_filter)
        self.browse_fourier.clicked.connect(self.browse_for_fourier)
        self.apply_fourier_button.clicked.connect(self.apply_fourier)
        self.open_for_fourier.clicked.connect(self.open_fourier)
        self.transform_button.clicked.connect(self.transform_difference)
        self.open_for_pattern.clicked.connect(self.browse_for_pattern)
        self.apply_pattern_button.clicked.connect(self.apply_pattern)
        self.display_image_button.clicked.connect(self.display_intensity_image)
        self.add_noise_button.clicked.connect(self.add_noise)
        self.histogram_noise_button.clicked.connect(self.histogram_noise)
        self.schepp_logan_button.clicked.connect(self.display_schepp_logan_phantom)
        self.apply_on_phantom_button.clicked.connect(self.back_projection)
        self.display_binary_button.clicked.connect(self.display_binary_image)
        self.apply_on_binary_button.clicked.connect(self.apply_on_binary)

        self.error_msg = QtWidgets.QErrorMessage()
        self.gray_img = [] #the original image in every tab
        self.generated_t = []
        self.salt_pepper_image = [] #new image for salt and pepper
        self.image_fft_array = [] #array of fourier
        self.image_fft_mag_array = [] #array for magnitude only
        self.image_fft_phase_array = [] #array for phase only
        self.phantom_array = [] #array of the phantom
        self.ROI_part = [] #array of the phantom after adding noise to it

       
        

       
        


        




 def browse_files(self, form): #to open files
        files_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open DICOM/JPG/PMB ', os.getenv('HOME'))
        path = files_name[0]
       

        magic.from_file(path) #opens the image from its path
        magic.from_file(path) == 'DICOM medical imaging data'

        if (pathlib.Path(path).suffix == ".dcm"): #if the path was a dicom picture
          ds = dicom.dcmread(path) #reads the image
          print(ds) #this was me trying to see if it works

      
          self.imagewidget.canvas.axes.imshow(ds.pixel_array, cmap=plt.cm.bone) #plots the image
       
         
          self.rows_label.setText(f'{ds.Rows}') #width
          self.column_label.setText(f' {ds.Columns}') #height
          self.size_label.setText(f' {ds.Rows * ds.Columns * ds.BitsStored }') #size
          self.modality_label.setText(ds.Modality) #modality
          self.name_label.setText(f' {ds.PatientName}') #patient name
          self.part_label.setText(f' {ds.StudyDescription}') #body part
          self.age_label.setText(f' {ds.PatientAge}') #patient age
          self.bit_label.setText(f'{ds.BitsStored}') #bit depth
          self.color_label.setText(f'{ds.PhotometricInterpretation}') #color of image
          
        
        else: #if it's PMB or JPG
              

        
         image = Image.open(path).convert('LA') #converts any type of image to grayscale 
         self.imagewidget.canvas.axes.imshow(image) #plots the image
         width, height = image.size #gets the height and width
         self.rows_label.setText(f'{width}') #prints the width
         self.column_label.setText(f'{height}') #prints the height
         
         


         self.color_label.setText(f'{image.mode}') #gets the color (RGB, grayscale,binary...)
       
         print(image.mode)
         
         if (image.mode == 'LA'): #only accepts grayscale images 
      
            grayimg = image
            self.imagewidget.canvas.axes.imshow(grayimg)
            self.imagewidget.canvas.draw()
            self.color_label.setText('Grayscale')
            print (type(grayimg))
              

            self.gray_img =  np.array(grayimg)[:,:,0] #makes array of the grayscale image to be used later in interpolation



        
 def linear_zoom(self): #bilinear interpolation 
       old_dim= self.gray_img.shape #gets the dimension of old grayscale image
       scale =  float(self.scale_input.text()) #takes the scale from the user
       if  scale <=0: #error handling for scale equals or less than 0
            
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Scale is equal or less than 0!')
                msg.setWindowTitle("Error!!")
                msg.exec_()

       scaled_dim= [old_dim[0]*scale, old_dim[1]*scale] #the new dimension is the old one multiplied by scale input

       scaled_array = np.random.randint(1, size=(int(scaled_dim[0]),int(scaled_dim[1]))) #making array of the new dimensions
       wScale = (old_dim[1] ) / (scaled_dim[1] ) if scaled_dim[0] != 0 else 0 #comparing the new scale to the old image to get exact scale of both h and w
       hScale = (old_dim[0] ) / (scaled_dim[0] ) if scaled_dim[1] != 0 else 0

      #the loop below to find the stars near the values of pixels and calculate the interpolation of each one and then the value compared 
      #to these stars by the same rule

       for i in range (int(scaled_dim[0])): 
              for j in range (int(scaled_dim[1])):
                     x=i*hScale
                     y=j*wScale
                     x_floor=math.floor(x)
                     x_ceil=min(old_dim[0]-1, math.ceil(x))

                     y_floor=math.floor(y)
                     y_ceil=min(old_dim[1]-1, math.ceil(y))
                     # print(scaled_dim[0])
                     if (x_ceil==x_floor) and (y_ceil==y_floor):
                            star=self.gray_img[int(x),int(y)]
                            

                     elif (x_ceil == x_floor):
                            # print(y_floor)
                            star1=self.gray_img[int(x),int(y_floor)]
                            star2=self.gray_img[int(x),int(y_ceil)]
                            star=(star1*(y_ceil-y))+(star2*(y-y_floor))

                     elif (y_ceil == y_floor):
                            star1=self.gray_img[int(x_floor),int(y)]
                            star2=self.gray_img[int(x_ceil),int(y)]
                            star=(star1*(x_ceil-x))+(star2*(x-x_floor))
                     else:
                            pt1=self.gray_img[x_floor,y_floor]
                            pt2=self.gray_img[x_ceil,y_floor]
                            pt3=self.gray_img[x_floor,y_ceil]
                            pt4=self.gray_img[x_ceil,y_ceil]

                            star1=pt1*(x_ceil-x)+pt2*(x-x_floor)
                            star2=pt3*(x_ceil-x)+pt4*(x-x_floor)
                            star=star1*(y_ceil-y)+star2*(y-y_floor)
                            #print(star)

                    
                     scaled_array[i,j]=star
       newImage = PIL.Image.fromarray(scaled_array) #making an image out of the array 
       self.zoomed_image_widget.canvas.axes.cla() #to clear the plot 
       self.zoomed_image_widget.canvas.axes.imshow(newImage ) #drawing the new interpolated image
       print(newImage.size)
       self.zoomed_image_widget.canvas.draw()








 def nearest_neighbour(self):
      
        w, h = self.gray_img.shape[:2] #getting the width and height of the original image
         
        xScale = float(self.scale_input.text()) #taking the scale from the user 
        yScale = float(self.scale_input.text())
        if(float(self.scale_input.text()) <= 0): #error handling for scale less or equals to zero
            
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Scale is equal or less than 0!')
                msg.setWindowTitle("Error!!")
                msg.exec_()
        

        xNew = int(w * float(xScale)) #to get the new positions we multiply the w and h by the scale input
        yNew = int(h * float(yScale))

              


        newImage = npy.zeros([xNew, yNew]) #make empty array for the new dimensions 
        print(self.gray_img.shape) #just to check smth
        #the next for loop is to divide each position by the scale and compare it to the old image to find where was it 
        #applying the nearest neighbour techniques
        for i in range(xNew): 
          for j in range(yNew):
             newImage[i, j] = self.gray_img[(int(i / float(xScale))), (int(j / float(yScale)))]

        #image_array = np.array(newImage, dtype=np.int32)

        self.zoomed_image_widget.canvas.axes.cla()
        self.zoomed_image_widget.canvas.axes.imshow(newImage, cmap = plt.cm.gray)
        #print(newImage.shape)
        self.zoomed_image_widget.canvas.draw()
        
 def generate_t(self):
       t_image = np.zeros((128,128))
       t_image[29:49,29:99] = 255
       t_image[49:99,54:74] = 255
       self.rotation_widget.canvas.axes.imshow(t_image, cmap = plt.cm.binary, interpolation='nearest')
       self.rotation_widget.canvas.draw()
       
       #self.generated_t = t_image


       

 def rotate_image(self):
       
       #w, h = self.normal_img.shape[:2]
       t_image = np.zeros((128,128))
       t_image[29:49,29:99] = 255
       t_image[49:99,54:74] = 255
       w, h = t_image.shape[:2]
       





       



       
       angle = int(self.angle_input.text())
       if(angle >= 0):
              self.status_label.setText("Rotated " + str(angle) + " Anti-Clockwise")
       else:
           self.status_label.setText("Rotated " + str(-angle) + " Clockwise")          

       angle=math.radians(angle)                               #converting degrees to radians
       cosine=math.cos(angle)
       sine=math.sin(angle)
       height=t_image.shape[0]                                   #define the height of the image
       width=t_image.shape[1]                                    #define the width of the image

       # Define the height and width of the new image that is to be formed
       new_height  = height
       new_width  = width

       # define another image variable of dimensions of new_height and new _column filled with zeros
       output=np.zeros((height,width))

       # Find the centre of the image about which we have to rotate the image
       original_centre_height   = int(((t_image.shape[0])/2))    
       original_centre_width    = int(((t_image.shape[1])/2))    

       

       if (self.nearest_rotate.isChecked()):
          for i in range(height):
              for j in range(width):
                    
                     y=i-(original_centre_height)                   
                     x=j-(original_centre_width)  
                     new_x=(x*sine+y*cosine)
                     new_y=(-x*cosine+y*sine)                   

                     
                     if x> (height-1) or y > (width-1):
                            new_x=int(new_x)
                            new_y=int(new_y)
                     else:    
                            new_x=round(new_x)
                            new_y=round(new_y)

                    
                     new_y=new_y + original_centre_height
                     new_x=new_x + original_centre_width

                    
                     if 0 <= new_x < width and 0 <= new_y < height:
                            output[i,j]=t_image[new_x,new_y]                          


                     
                    
       if (self.bilinear_rotate.isChecked()): 
              for i in range (height): 
                     for j in range (width):
                          

                            
                            
                            x= (i-original_centre_width)*cosine +(j-original_centre_height)*sine
                            y= -(i-original_centre_width)*sine +(j-original_centre_height)*cosine
                            x=x+original_centre_width
                            y=y+original_centre_height

                            x_floor=math.floor(x)
                            x_ceil=min(height-1, math.ceil(x))
                            y_floor=math.floor(y)
                            y_ceil=min(width-1, math.ceil(y))

                           

                            # print(scaled_dim[0])
                            if  0 <= x_ceil < width and 0 <= x_floor < width and 0 <=y_ceil < height and 0 <=y_floor < height:
                                   if (x_ceil==x_floor) and (y_ceil==y_floor):
                                          output[i,j]=t_image[int(x),int(y)]
                                          
                                   elif (x_ceil == x_floor):
                                          # print(y_floor)
                                          star1=t_image[int(x),int(y_floor)]
                                          star2=t_image[int(x),int(y_ceil)]
                                          star=(star1*(y_ceil-y))+(star2*(y-y_floor))
                                          output[i,j]=star

                                   elif (y_ceil == y_floor):
                                          star1=t_image[int(x_floor),int(y)]
                                          star2=t_image[int(x_ceil),int(y)]
                                          star=(star1*(x_ceil-x))+(star2*(x-x_floor))
                                          output[i,j]=star
                                   else:
                                          pt1=t_image[x_floor,y_floor]
                                          pt2=t_image[x_ceil,y_floor]
                                          pt3=t_image[x_floor,y_ceil]
                                          pt4=t_image[x_ceil,y_ceil]

                                          star1=pt1*(x_ceil-x)+pt2*(x-x_floor)
                                          star2=pt3*(x_ceil-x)+pt4*(x-x_floor)
                                          star=star1*(y_ceil-y)+star2*(y-y_floor)
                                          output[i,j]=star


              
              
                                   

                     
                            
       newImage = np.array(output, dtype=np.int32)  
       self.rotation_widget.canvas.axes.imshow(newImage, cmap = plt.cm.binary)
       self.rotation_widget.canvas.draw()


 def shear_image(self):
       t_image = np.zeros((128,128))
       t_image[29:49,29:99] = 255
       t_image[49:99,54:74] = 255
       height=t_image.shape[0]                               
       width=t_image.shape[1]
                           
       output=np.zeros((height,width))
       original_centre_height   = int(((t_image.shape[0])/2))  
       original_centre_width    = int(((t_image.shape[1])/2))
       for i in range(height):
           for j in range(width):
               
                     
                     y=i-(original_centre_height)                   
                     x=j-(original_centre_width)
                     new_x=x
                     new_y= y-x                    

                 
                     new_y=new_y + original_centre_height
                     new_x=new_x + original_centre_width

                     if 0 <= new_x < width and 0 <= new_y < height:
                            output[j,i]=t_image[new_x,new_y]
       newImage = np.array(output, dtype=np.int32)  
       self.rotation_widget.canvas.axes.imshow(newImage, cmap = plt.cm.binary)
       self.rotation_widget.canvas.draw()
       

 def draw_normal_histogram(self):
       files_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open JPG/PMB ', os.getenv('HOME'))
       path = files_name[0]
       image = Image.open(path).convert('L') #converts any type of image to grayscale 
       self.before_equalize_img.canvas.axes.imshow(image, cmap='gray') #plots the image
       self.before_equalize_img.canvas.draw()
      # new_image =np.array(image)[:,:]
       
       if (image.mode == 'L'): #only accepts grayscale images 
      
               grayimg = image
               self.gray_img =  np.array(grayimg)[:,:]

       

       a = np.zeros((256,),dtype=np.float16)
      
      
       height = len(self.gray_img)
       width = len(self.gray_img[0])


       # #finding histogram
       for i in range(width):
          for j in range(height):
               g = self.gray_img[j,i]
               a[g] = a[g]+1 #count number of intensities
       newImage = np.array(a, dtype=np.int32)  
       #HISTOGRAM
       first_histogram = np.array(newImage/(self.gray_img.shape[0]*self.gray_img.shape[1])) #normalizing divide size
       self.normal_histogram.canvas.axes.bar(np.arange(256),height=first_histogram)
       self.normal_histogram.canvas.draw()

#-------------------------------------------------------equalization-------------------------------------------------------------------------

       b = np.zeros((256,),dtype=np.float16)
       tmp = 1.0/(height*width) #normalize
       
# el Sk
       for i in range(256):
          for j in range(i+1):
               b[i] += a[j] * tmp #CDf
          b[i] = round(b[i] * 255)

       # b now contains the SK
 
       equalized = np.array(b, dtype=np.int32)  
       
#------------equalized image
       for i in range(width):
          for j in range(height):
              g = self.gray_img[j,i]
              self.gray_img[j,i]= equalized[g]

       


       newImage = np.array(self.gray_img, dtype=np.int32)
       print(newImage.shape)
        
       self.after_equalize_img.canvas.axes.imshow(newImage, cmap = 'gray')
       self.after_equalize_img.canvas.draw()



       a = np.zeros((256,),dtype=np.float16)
      
       height = len(self.gray_img)
       width = len(self.gray_img[0])


       # #finding equalized histogram 
       for i in range(width):
          for j in range(height):
               g = self.gray_img[j,i]
               a[g] = a[g]+1
       newImage = np.array(a, dtype=np.int32)  
       new_histogram = np.array(newImage/(self.gray_img.shape[0]*self.gray_img.shape[1]))
       print(np.amax(new_histogram))
       self.equalized_histogram.canvas.axes.bar(np.arange(256),height=new_histogram)
       self.equalized_histogram.canvas.draw()



 def browse_for_filtering(self):
       files_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open JPG/PMB ', os.getenv('HOME'))
       path = files_name[0]
       image = Image.open(path).convert('L') #converts any type of image to grayscale 
       self.before_filtering_image.canvas.axes.imshow(image, cmap='gray') #plots the image
       self.before_filtering_image.canvas.draw()
      # new_image =np.array(image)[:,:]
       
       if (image.mode == 'L'): #only accepts grayscale images 
      
               grayimg = image
               self.gray_img =  np.array(grayimg)[:,:]
       return self.gray_img        

 def browse_for_fourier(self):
       files_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open JPG/PMB ', os.getenv('HOME'))
       path = files_name[0]
       image = Image.open(path).convert('L') #converts any type of image to grayscale 
       self.before_fourier_img.canvas.axes.imshow(image, cmap='gray') #plots the image
       self.before_fourier_img.canvas.draw()
      # new_image =np.array(image)[:,:]
       
       if (image.mode == 'L'): #only accepts grayscale images 
      
               grayimg = image
               self.gray_img =  np.array(grayimg)[:,:]
       return self.gray_img
 def browse_for_pattern(self):
       files_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open JPG/PMB ', os.getenv('HOME'))
       path = files_name[0]
       image = Image.open(path).convert('L') #converts any type of image to grayscale 
       self.before_pattern_img.canvas.axes.imshow(image, cmap='gray') #plots the image
       self.before_pattern_img.canvas.draw()
      # new_image =np.array(image)[:,:]
       
       if (image.mode == 'L'): #only accepts grayscale images 
      
               grayimg = image
               self.gray_img =  np.array(grayimg)[:,:]
       return self.gray_img

 def matrix_convolution(self, kernel_input, mask_input): #takes kernel box and mask as parameters for convolution 
       product = 0 #product initialized
       kernel_box = kernel_input.flatten() #turns the kernel parameter into 1D array

       mask = mask_input.flatten() #turns the mask parameter into 1D array
       #loops over the kernel box values to multiplicate them with the mask values 
       for k in range (len(kernel_box)):
              product += (kernel_box[k]*mask[k])  
       return product #returns the product to then be used in filters
 

 

        
 def unsharp_filter(self):
       scale = int(self.factor_label.text()) #takes the factor from the user 
       kernel_size = int(self.kernel_size_label.text()) #takes the kernel size from the user 
       kernel_box = np.ones(shape= (kernel_size, kernel_size)) #forms columns and rows of kernel size 
       
       pad = int(kernel_size/2) #calculates the number of rows and columns lengths needed for padding 
       new_arr_width = self.gray_img.shape[1] + kernel_size - 1
       new_arr_height = self.gray_img.shape[0] + kernel_size - 1
       array_of_pads = np.zeros(shape=(new_arr_height, new_arr_width), dtype=np.uint)
       array_of_pads[pad:pad+self.gray_img.shape[0], pad:pad+self.gray_img.shape[1]] = self.gray_img
       #array_of_pads = np.pad(self.gray_img, [(pad, pad), (pad, pad)], mode='constant', constant_values=0) #completes the image with this padding
       sharp_array = np.zeros(self.gray_img.shape) #new array of zeros to be used as potential output of the filter
       offset = int(kernel_size /2) #the offset rows and columns lengths 
       rows_limit = offset + self.gray_img.shape[0] #end of offset rows 
       columns_limit = offset + self.gray_img.shape[1] #end of offset columns
       for i in range (offset, rows_limit): #from offset to end of rows (the padded part)
             for j in range (offset, columns_limit): #from offset to end of columns (the padded part)
                   mask = array_of_pads[i-offset: i+offset+1,j-offset:j+offset+1] #gets the representation of image that's under the mask
                   sharp_array[i-offset][j-offset] = round(self.matrix_convolution(kernel_box,mask)/(kernel_size **2)) #convolution divided by kernel input then rounded
                 
       contrast = self.gray_img - sharp_array #calculates the difference between the original image and the image after filtering
       sharped_image = self.gray_img + (scale * contrast) #the output would be the original + the factor input multiplied by the contrast we calculated from filter output 
       sharped_image[sharped_image<0] = 0 #finally the scaling back to be from 0 to 255 not less not more
       sharped_image[sharped_image>255] =255
       self.filtered_image.canvas.axes.imshow(sharped_image, cmap='gray') #plots the image
       self.filtered_image.canvas.draw()

 def sort(self,a, b, left, right):

    
    #right can't be smaller than left 
    if right <= left:
        return

    #calculates the midpoint of left and right 
    mid = (left + right)//2

    #recursively do this again but with each subarray and merge again by calling the merge function
    self.sort(a, b, left, mid)
    self.sort(a, b, mid + 1, right)
    self.merge(a, b, left, mid, right)


 def merge(self,a, b, left, mid, right):
    i = left     #start of left subarray
    j = mid + 1  # start of right subarray
    k = left
 
    while i <= mid and j <= right:    #if start is in bounds
    
        # compare and increment
        if a[i] <= a[j]:
            b[k] = a[i]
            i += 1
        else:
            
            b[k] = a[j]
            j += 1
 
        k += 1

    #add them to a temporary array
    while i <= mid:
        b[k] = a[i]
        k += 1
        i += 1

    while j <= right:
        b[k] = a[j]
        k += 1
        j += 1
 
    #add the sorted to the original to be used after calling
    for i in range(left, right + 1):
        a[i] = b[i]    
 def salt_pepper_filter(self):
       self.salt_pepper_image = np.zeros(shape= self.gray_img.shape, dtype=np.uint) #array of zeros to be used as output array later
       for i in range(self.gray_img.shape[0]): #loops over the original image rows
            for j in range(self.gray_img.shape[1]): #loops over the original image columns

                noise_option = random.randint(0,1)     #randomly choose to use the salt and pepper filter or not (on each pixel)

                if noise_option == 1: #if yes
                    noise = random.randint(0,1)       # randomly choose to use salt or pepper on each pixel
                    self.salt_pepper_image[i][j] = noise * 255 #the new image will be the salt OR pepper added multiplied by 255
                else: #no salt and pepper filter on pixel
                    self.salt_pepper_image[i][j] = self.gray_img[i][j] #leave the pixel as it was in the original image

       self.salt_pepper_widget.canvas.axes.imshow(self.salt_pepper_image, cmap='gray') #plots the image
       self.salt_pepper_widget.canvas.draw() 

 def median_filter(self):
        
        median_array = np.zeros(shape=self.salt_pepper_image.shape, dtype=np.uint) #array of zeros to be used as output of median filter later on
        kernel_size = int(self.kernel_size_label.text()) #takes kernel size from the user

        
        pad = int(kernel_size/2) ##calculates the number of rows and columns lengths needed for padding 
        array_of_pads = np.pad(self.gray_img, [(pad, pad), (pad, pad)], mode='constant', constant_values=0) #completes the image with this padding

        
        offset = int(kernel_size/ 2) #the offset rows and columns lengths 
        rows_limit = offset + self.salt_pepper_image.shape[0] #end of offset rows
        columns_limit = offset + self.salt_pepper_image.shape[1] #end of offset columns

       
        for i in range(offset, rows_limit): #from offset to end of rows (the padded part)
            for j in range(offset, columns_limit): #from offset to end of columns (the padded part)
                mask = array_of_pads[i-offset:i+offset+1, j-offset:j+offset+1] #gets the representation of image that's under the mask
                b = mask.flatten() #turns the mask into 1D array
                self.sort(mask.flatten(), b, 0, (len(mask.flatten()) - 1)) #calls the function merge sort (check above)
                median_array[i-offset][j-offset] = b[len(b) // 2] #replace each sorted pixel with median and add it to new array of zeros
        self.median_image_widget.canvas.axes.imshow(median_array, cmap='gray') #plots the image
        self.median_image_widget.canvas.draw()

     



 def apply_fourier(self):
       self.image_fourier_array = np.fft.fft2(self.gray_img)
       self.image_fourier_array = np.fft.fftshift(self.image_fourier_array) #applies fft
        #takes square root of real and imaginary squared to calculate the magnitude:
       self.mag_array = np.sqrt(self.image_fourier_array.real ** 2 + self.image_fourier_array.imag ** 2) 
        #shift tan of imaginary and real to calculate the phase
       self.phase_array = np.arctan2(self.image_fourier_array.imag, self.image_fourier_array.real) 
       #plot the magnitude
       self.mag_fourier_img.canvas.axes.imshow(self.mag_array, cmap='gray') 
       self.mag_fourier_img.canvas.draw() 
       #plot the phase
       self.after_fourier_img.canvas.axes.imshow(self.phase_array, cmap='gray') 
       self.after_fourier_img.canvas.draw()
       #the magnitude after log is just the log of it
       self.mag_after_log = np.log(self.mag_array + 1) 
       #plot the magnitude after log
       self.log_mag_img.canvas.axes.imshow(self.mag_after_log, cmap='gray') 
       self.log_mag_img.canvas.draw()    
       #the phase after log is log of the phase plus of the pi 
       self.phase_after_log = np.log(self.phase_array + np.pi) 
       #plot the phase after log  
       self.log_phase_img.canvas.axes.imshow(self.phase_after_log, interpolation = "None", cmap='gray', vmin = 0) 
       self.log_phase_img.canvas.draw()   

 def open_fourier(self):      
       files_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open JPG/PMB ', os.getenv('HOME'))
       path = files_name[0]
       image = Image.open(path).convert('L') #converts any type of image to grayscale 
       self.original_image_fourier.canvas.axes.imshow(image, cmap='gray') #plots the image
       self.original_image_fourier.canvas.draw()
      # new_image =np.array(image)[:,:]
       
       if (image.mode == 'L'): #only accepts grayscale images 
      
               grayimg = image
               self.gray_img =  np.array(grayimg)[:,:]
       return self.gray_img
 
 def transform_difference(self):
       k_size = int(self.kernel_size_fourier_label.text()) #takes the factor from the user 
       kernel_box = np.ones(shape= (k_size, k_size)) #forms columns and rows of kernel size 
       
       pad = int(k_size/2) #calculates the number of rows and columns lengths needed for padding 
       new_arr_width = self.gray_img.shape[1] + k_size - 1
       new_arr_height = self.gray_img.shape[0] + k_size - 1
       array_of_pads = np.zeros(shape=(new_arr_height, new_arr_width), dtype=np.uint)
       array_of_pads[pad:pad+self.gray_img.shape[0], pad:pad+self.gray_img.shape[1]] = self.gray_img
       #array_of_pads = np.pad(self.gray_img, [(pad, pad), (pad, pad)], mode='constant', constant_values=0) #completes the image with this padding
       blurred = np.zeros(self.gray_img.shape) #new array of zeros to be used as potential output of the filter
       offset = int(k_size /2) #the offset rows and columns lengths 
       rows_limit = offset + self.gray_img.shape[0] #end of offset rows 
       columns_limit = offset + self.gray_img.shape[1] #end of offset columns
       for i in range (offset, rows_limit): #from offset to end of rows (the padded part)
             for j in range (offset, columns_limit): #from offset to end of columns (the padded part)
                   mask = array_of_pads[i-offset: i+offset+1,j-offset:j+offset+1] #gets the representation of image that's under the mask
                   blurred[i-offset][j-offset] = round(self.matrix_convolution(kernel_box,mask)/(k_size **2)) #convolution divided by kernel input then rounded
                 
       self.blurred_img.canvas.axes.imshow(blurred, cmap='gray') #plots the  blurred image
       self.blurred_img.canvas.draw()
       kernel=np.ones((k_size,k_size),dtype=np.float32)/(k_size**2) #makes a mask
       kernel_value = np.zeros((self.gray_img.shape[0],self.gray_img.shape[1]),dtype=np.float32) #kernel values as zero
       x=(self.gray_img.shape[0]-k_size+1)//2 #looping over the image minus the kernel size
       y=(self.gray_img.shape[1]-k_size+1)//2
       for i in range (x,x+k_size): #from image minus kernel size to adding the kernel size 
              for j in range(y,y+k_size): 
                     kernel_value[i,j]=kernel[i-x,j-y] #filling it with the input of kernel (which is 1)
       fft2_image = np.fft.fft2(self.gray_img) #getting fourier 2 of image and kernel 
       fft2_kernel = np.fft.fft2(kernel_value)
       product=fft2_image*fft2_kernel #multiply the 2 values 
       ifft2_product = np.fft.ifft2(product) #inverse fourier for the result 
       shift_product = np.fft.ifftshift(ifft2_product) #shift the inverse fourier of the result
       fourier_final=np.abs(shift_product) #convert to real number (abs can be used as well)
       
       
       self.fft_img.canvas.axes.imshow(fourier_final, cmap='gray') #plots the image
       self.fft_img.canvas.draw()
       difference= fourier_final-blurred #the difference between fourier and spacial after blurring
       difference[difference<0] = 0  #to make the difference clear 
       difference[difference>255] =255
       self.difference_img.canvas.axes.imshow(difference, cmap='gray') #plots the image
       self.difference_img.canvas.draw()
       print(difference)
 def apply_pattern(self):
       img_height = self.gray_img.shape[0] 
       img_width = self.gray_img.shape[1]
       filter_arr = np.full((img_height, img_width), 1) 
       filter_arr[475:495,375:395] = 0
       filter_arr[555:575,375:395] = 0
       filter_arr[475:495,325: 340] = 0
       filter_arr[555:575,325: 340] = 0

       filter_arr[435:455,375:395] = 0
       filter_arr[435:455,325: 335] = 0
       filter_arr[605:625,325: 340] = 0
       filter_arr[605:625,380:395] = 0 #ranges in coordinates of the array that will be equal to 0 



       fft_image = np.fft.fft2(self.gray_img) #apply fourier and shift
       fftshift_image = np.fft.fftshift( fft_image)
       real_part =  fftshift_image.real #get the real and imaginary of the last
       imag_part =  fftshift_image.imag

       magnitude = np.sqrt(( real_part ** 2) + ( imag_part ** 2)) #get the magnitude from the real and imaginary (sqrt)
       log_of_magnitude = np.log( magnitude + 1) #get the log of the magnitude


       after_filtering_img = fftshift_image * filter_arr #the image after fourier multiplied by the array after zeros
       restored_img_array = ((np.fft.ifft2(np.fft.ifftshift(after_filtering_img)))) #new image will be shifting the last one then fourier 2
       real_part = restored_img_array.real #real and imaginary of the restored image
       imag_part = restored_img_array.imag
       restored_img = np.sqrt((real_part ** 2) + (imag_part ** 2)) #the image magnitude which is the final image
       freqs = log_of_magnitude * filter_arr
       self.pattern_img.canvas.axes.imshow(log_of_magnitude, interpolation = "None", cmap='gray', vmin = 0) 
       self.pattern_img.canvas.draw() 
       self.restored_img_widget.canvas.axes.imshow(restored_img, cmap='gray') 
       self.restored_img_widget.canvas.draw() 

 def display_intensity_image(self):
       x = np.linspace(-10, 10, 256)
       y = np.linspace(-10, 10, 256)
       x, y = np.meshgrid(x, y)
       x_0 = 0
       y_0 = 0
       circle = np.sqrt((x-x_0)**2+(y-y_0)**2) #equation of circle

       r = 5 #radius (constant)
       for x in range(256): 
              for y in range(256):
                if circle[x,y] < r: #inside the circle color
                     circle[x,y] = 80
                elif circle[x,y] >= r: #outside the circle color
                     circle[x,y] = 0



       #the following part is for the squares (outside the circle)
       squares = np.full((256, 256), 50) #256*256 phantom with value 50
       for i in range(35, 221):
          for j in range(35, 221):
              squares[i,j] = 120 #loop over x and y and give it 120 (the smaller square), outside this loop its 50

       phantom = squares + circle #combine the 2 parts to form the phantom 
       self.phantom_array = phantom #put it in an array to be used later 
       self.displayed_image.canvas.axes.imshow(phantom, cmap = 'gray') #shows the phantom 
       self.displayed_image.canvas.draw()
 def add_noise(self):
    a_word = "A"
    b_word = "B" 
    if self.gaussian_radio.isChecked(): 
        
        noise = np.random.normal(int(self.mean_label.text()), int(self.std_label.text()), self.phantom_array.shape) #performs gaussian noise on the image
    elif self.uniform_radio.isChecked():
            self.mean_a_word.setText(f'{a_word}') #takes a and b instead of mean and std
            self.std_b_word.setText(f'{b_word}')
            noise = np.random.uniform(int(self.mean_label.text()), int(self.std_label.text()), self.phantom_array.shape) #performs normal noise on the image
    self.noisy_phantom_array = np.round(noise + self.phantom_array) #the normal image + its noise rounded to be in a noisy image array
    self.noisy_image.canvas.axes.imshow(self.noisy_phantom_array, cmap='gray') #shows the image after adding noise 
    self.noisy_image.canvas.draw()  
    self.noisy_array = self.noisy_phantom_array.astype(np.uint8)   #converts it to an array to be used later   
    r = cv2.selectROI("select the area", self.noisy_array) #this library used to get region of interest from an image
    self.ROI_part = self.noisy_phantom_array[int(r[1]):int(r[1]+r[3]),
                      int(r[0]):int(r[0]+r[2])] #the ROI will be saved here
    
    cv2.waitKey(0)
    
   
             
    
 def histogram_noise(self):
       

        ROI_flatten = self.ROI_part.flatten().astype(np.uint) #as before flatten the ROI and put it in an array

        if np.amax(ROI_flatten) + 1 < 256:
            depth = 256
        else:
            depth = np.amax(ROI_flatten) + 1

        
        histo_arr = np.zeros(depth, dtype=np.uint) #gets the frequency of pixels 

        
        for i in ROI_flatten: #loop over the flatten image 
            histo_arr[i] = histo_arr[i] + 1 #increment to count

        
        self.histogram_array = histo_arr / self.ROI_part.size #divide by total num of pixels to normalize
        #the next part to get mean and standard deviation from the histogram array
        self.mean = 0 #mean initialization
        for i in range(len(self.histogram_array)):
            self.mean += i * self.histogram_array[i] #location multiplied by its value in histo array then increment 
        self.mean = round(self.mean, 5)
        std = 0 #std initialization
        for i in range(len(self.histogram_array)):
            std += (i - self.mean) ** 2 * self.histogram_array[i] #equation of variance including the mean calculated 

        std = round(np.sqrt(std), 5) #square root the variance and round it to get standard deviation 
        self.mean_histo_label.setText(str(self.mean)) #shows the mean
        self.std_histo_label.setText(str(std)) #shows the standard deviation
        self.noisy_histogram.canvas.axes.bar(np.arange(256),height=self.histogram_array) #printing the histogram array showing the histogram of the image
        self.noisy_histogram.canvas.draw()

 def display_schepp_logan_phantom(self):
       self.gray_img = data.shepp_logan_phantom()
       self.gray_img = np.array(self.gray_img) 
       self.schepp_logan_phantom_img.canvas.axes.imshow(self.gray_img, cmap='gray') #plots the image
       self.schepp_logan_phantom_img.canvas.draw()       

 def back_projection(self):
       #Sinogram:
       theta = np.arange(0, 180, 1) #angles 0 to 180 with step 1
       sinogram = transform.radon(self.gray_img, theta) #makes the sinogram
       self.sinogram_img.canvas.axes.imshow(sinogram, cmap='gray') #plots the image
       self.sinogram_img.canvas.draw() 
       #laminogram step 1
       theta_one = np.arange(0, 180, 1) #angle from 0 to 180 step 1
       radon_one = transform.radon(self.gray_img, theta_one) 
       laminogram_one = transform.iradon(radon_one, theta_one, filter_name= None) #makes the laminogram by doing inverse radon, with no filter
       self.laminogram_img1.canvas.axes.imshow(laminogram_one, cmap='gray') #plots the image
       self.laminogram_img1.canvas.draw()  

       #laminogram step 20, no filters
       theta_20 = np.arange(0, 160, 20) #from 0 to 160 step 20
       radon_twenty = transform.radon(self.gray_img, theta_20)
       laminogram_twenty = transform.iradon(radon_twenty, theta_20, filter_name= None) #makes laminogram by doing inverse radon and no filter
       self.laminogram_img20.canvas.axes.imshow(laminogram_twenty, cmap='gray') #plots the image
       self.laminogram_img20.canvas.draw()
       
       #ram lak filter 
       theta_ram_lak = np.arange(0, 180, 1) #from 0 to 180 step 1 
       radon_ram_lak = transform.radon(self.gray_img, theta_ram_lak)
       ram_lak_applied = transform.iradon(radon_ram_lak, theta_ram_lak, filter_name= 'ramp') #make laminogram by inverse radon and filter name ramp
       self.ram_lak_filter_img.canvas.axes.imshow(ram_lak_applied, cmap='gray') #plots the image
       self.ram_lak_filter_img.canvas.draw() 
       #hamming filter
       theta_hamming = np.arange(0, 180, 1) #from 0 to 10 step 1
       radon_hamming = transform.radon(self.gray_img, theta_hamming)
       hamming_applied = transform.iradon(radon_hamming, theta_hamming, filter_name= 'hamming') #makes laminogram by inverse radon and filter name hamming 
       self.hamming_filter_img.canvas.axes.imshow(hamming_applied, cmap='gray') #plots the image
       self.hamming_filter_img.canvas.draw()      
 def display_binary_image(self):
       file = "binary_image.png"
       image = Image.open(file).convert('1') #converts the image to binary image
       self.binary_img.canvas.axes.imshow(image, cmap='gray') #plots the image
       self.binary_img.canvas.draw()
     
       
      
       grayimg = image
       self.gray_img =  np.array(grayimg)[:,:]
       
       return self.gray_img        
 def apply_on_binary(self):
       self.h_img = self.gray_img.shape[1]
       self.w_img = self.gray_img.shape[0] #gets width and height of the image
       self.st_elem_size = 5 #the structure element size with ones 
       self.offset = (self.st_elem_size - 1) // 2  #offset of start and end points of the structure element
       self.st_elem = np.ones((self.st_elem_size,self.st_elem_size)) #array of ones 

       self.st_elem[0,0] = 0 #assigning zero to the origin
       self.st_elem[self.st_elem_size-1,self.st_elem_size-1] = 0 #assigning zero to the last element 
       self.st_elem[0,self.st_elem_size-1] = 0 #last column first row
       self.st_elem[self.st_elem_size-1,0] = 0 #last row first column
       SE_flattened = self.st_elem.flatten() #flatten for 1D array
       SE_length = len(SE_flattened) #get the length of the 1D array of SE
       self.erosion_image= np.zeros((self.w_img,self.h_img)) #new array of zeros for the image after erosion to be used later
       for i in range(self.offset , self.w_img - self.offset ): #looping over offset and stop point of rows
              for j in range(self.offset ,self.h_img - self.offset ): #looping over offset and stop point of columns
                     c = 0 #todo: count 
                     img_prod = self.gray_img[i - self.offset :i + self.offset + 1, j - self.offset :j + self.offset + 1] 
                     img_prod_flattened = img_prod.flatten() #area of original image under SE
                     for k in range(SE_length): #loop over the SE
                            if SE_flattened[k] == 1 and img_prod_flattened[k] == 1: #if both SE and are under SE equals zero then increment
                                   c += 1

                     if c == ((self.st_elem_size) ** 2) - 4:  #5*5 = 25 -> 25-4 =21 (fit spots)
                            self.erosion_image[i,j] = 1 #perform erosion 



       SE_flattened = self.st_elem.flatten() #SE in 1D array
       SE_length = len(SE_flattened) #gets SE length to loop over 
       self.dilation_image= np.zeros((self.w_img,self.h_img)) #array of zeros of image after dilation to be used later 
       for i in range(self.offset , self.w_img - self.offset ): #loop over the offset to the stop points of rows 
              for j in range(self.offset ,self.h_img - self.offset ): #loop over the offset to the stop points of columns 
                     img_prod = self.gray_img[i - self.offset :i + self.offset + 1, j - self.offset :j + self.offset + 1] #area under SE
                     img_prod_flattened = img_prod.flatten() #area under SE in 1D array
                     for k in range(SE_length): #loop over SE length
                            if SE_flattened[k] == 1 and img_prod_flattened[k] == 1: #if both SE and value under SE equals to 1 (hit)
                                   self.dilation_image[i,j] = 1 

       #opening: start with erosion as original image then perform dilation 
       new_og_image = self.erosion_image #the new image to be used instead of the original image
       SE_flattened = self.st_elem.flatten() #1D array of the SE
       SE_length = len(SE_flattened) #gets its length
       opening_image= np.zeros((self.w_img,self.h_img)) #array of zeros of image after opening to be used later 
       for i in range(self.offset , self.w_img - self.offset ): #loop over offset to stop point of rows
              for j in range(self.offset ,self.h_img - self.offset ): #loop over offset to stop point of columns
                     img_prod = new_og_image[i - self.offset :i + self.offset + 1, j - self.offset :j + self.offset + 1] #area under SE
                     img_prod_flattened = img_prod.flatten() #area under SE in 1D array
                     for k in range(SE_length): #loop over length of SE
                            if SE_flattened[k] == 1 and img_prod_flattened[k] == 1: #if both SE and area under SE equals one
                                   opening_image[i,j] = 1 # (hit)
       #closing: start with dilation as original image then perform erosion 
       new_og_image = self.dilation_image #the new image to be used instead of the original image
       SE_flattened = self.st_elem.flatten() #1D array of the SE
       SE_length = len(SE_flattened) #gets its length
       closing_image= np.zeros((self.w_img,self.h_img))  #array of zeros of image after opening to be used later 
       for i in range(self.offset , self.w_img - self.offset ): #loop over offset to stop point of rows
              for j in range(self.offset ,self.h_img - self.offset ): #loop over offset to stop point of columns
                     c = 0 #todo: count (only in erosion)
                     img_prod = new_og_image[i - self.offset :i + self.offset + 1, j - self.offset :j + self.offset + 1] #area under SE
                     img_prod_flattened = img_prod.flatten() #area under SE in 1D array
                     for k in range(SE_length): #loop over SE length
                            if SE_flattened[k] == 1 and img_prod_flattened[k] == 1: #if both SE and value under SE (fit) equals 1
                                   c += 1  #increment the count 

                     if c == ((self.st_elem_size) ** 2) - 4: #5*5 = 25 -> 25-4 =21 (fit spots)
                          closing_image[i,j] = 1      #perform erosion  
       self.erosion_img.canvas.axes.imshow(self.erosion_image, cmap='gray') #plots the image
       self.erosion_img.canvas.draw()      
       self.dilation_img.canvas.axes.imshow(self.dilation_image, cmap='gray') #plots the image
       self.dilation_img.canvas.draw()  
       self.opening_img.canvas.axes.imshow(opening_image, cmap='gray') #plots the image
       self.opening_img.canvas.draw()      
       self.closing_img.canvas.axes.imshow(closing_image, cmap='gray') #plots the image
       self.closing_img.canvas.draw()    



    

                          



                 
app = QtWidgets.QApplication(sys.argv)

w = MainWindow()
w.show()
sys.exit(app.exec_())