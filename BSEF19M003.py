import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 


#Read image
#cv2 reads image as BGR
image= cv.imread('book.png')

#convert BGR to RGB
image = cv.cvtColor(image, cv.COLOR_BGR2RGB) 

#Separating each of the image channels
red_channel=image[:,:,0] #greyscale individually
green_channel=image[:,:,1] #greyscale individually
blue_channel=image[:,:,2]  #greyscale individually




####   using box filters

# 3*3 smoothing kernel
kernel1=1/9*(np.array([
                        [1,1,1],
                        [1,1,1],
                        [1,1,1]
                        ]))

# 5*5 smoothing kernel
kernel2=1/25*(np.array([
                        [1,1,1,1,1],
                        [1,1,1,1,1],
                        [1,1,1,1,1],
                        [1,1,1,1,1],
                        [1,1,1,1,1]
                        ]))

# 7*7 smoothing kernel
kernel3=1/49*(np.array([
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1]
                        ]))


#storing number of rows and columns of original image
ir,ic =red_channel.shape    

#creating 3 different images to store results of 
#3X3, 5X5, 7X7 box filters applied on image
resultant_image_3X3=np.zeros(shape=(ir,ic,3))
resultant_image_5X5=np.zeros(shape=(ir,ic,3))
resultant_image_7X7=np.zeros(shape=(ir,ic,3))



#function which performs convolution on a 2D array (one channel of image)
def PerformConvolution(img, kernel):
    
    #2D array to store the filtered results
    resultant_channel=np.zeros(shape=(ir,ic))
    
    #storing kernel's shape (3X3, 5X5, 7X7)
    kr,kc=kernel.shape    
    
    #storing the size of original image before it gets padded
    temp_ir=ir # will be used to iterate the image rows
    temp_ic=ic # will be used to iterate the image columns
    
    
    #here we are using shape of kernel to find the padding factor
    x=(kr-1)/2
    x=int(x)
    
    
    #padding the original image by the factor x
    img = cv.copyMakeBorder(img,x,x,x,x,cv.BORDER_CONSTANT, None, value = 0)
    
    
    rci =0
    
    for ii in range(0,temp_ir):#rows of image
        rcc=0
        for ij in range(0,temp_ic):#columns of image
            ans=0
            tempIthIndexOfImage=ii
            
            #this nested loop finds value for one pixel of image
            for mi in range(0,kr):#rows of mask
                tempJthIndexOfImage=ij
                for mj in range(0,kc):#columns of mask
                        ans+=(img[tempIthIndexOfImage][tempJthIndexOfImage]*kernel[mi][mj])
                        tempJthIndexOfImage+=1
                tempIthIndexOfImage+=1
            resultant_channel[rci][rcc]=ans
            rcc+=1
        rci+=1
    return resultant_channel
        




#___________________________SMOOTHENING______________________________

#using 3X3 mask:
resultant_image_3X3[:,:,0]=PerformConvolution(blue_channel, kernel1)
resultant_image_3X3[:,:,1]=PerformConvolution(green_channel, kernel1)
resultant_image_3X3[:,:,2]=PerformConvolution(red_channel, kernel1)


resultant_image_3X3=resultant_image_3X3.astype("uint16")

cv.imwrite('3X3 box filtered.jpg',resultant_image_3X3)
resultant_image_3X3 = cv.cvtColor(resultant_image_3X3, cv.COLOR_BGR2RGB)
plt.imshow(resultant_image_3X3)



#using 5X5 mask:
resultant_image_5X5[:,:,0]=PerformConvolution(blue_channel, kernel2)
resultant_image_5X5[:,:,1]=PerformConvolution(green_channel, kernel2)
resultant_image_5X5[:,:,2]=PerformConvolution(red_channel, kernel2)

resultant_image_5X5=resultant_image_5X5.astype("uint16")

cv.imwrite('5X5 box filtered.jpg',resultant_image_5X5)
resultant_image_5X5 = cv.cvtColor(resultant_image_5X5, cv.COLOR_BGR2RGB) 
plt.imshow(resultant_image_5X5)



#using 7X7 mask:
resultant_image_7X7[:,:,0]=PerformConvolution(blue_channel, kernel3)
resultant_image_7X7[:,:,1]=PerformConvolution(green_channel, kernel3)
resultant_image_7X7[:,:,2]=PerformConvolution(red_channel, kernel3)


resultant_image_7X7=resultant_image_7X7.astype("uint16")

cv.imwrite('7X7 box filtered.jpg',resultant_image_7X7)
resultant_image_7X7 = cv.cvtColor(resultant_image_7X7, cv.COLOR_BGR2RGB) 
plt.imshow(resultant_image_7X7)



#____________________________SHARPENING__________________________________


#sharpening filter
kernel4= np.array([[0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]
                        ])

#creating array to store sharpened image
resultant_sharpened_image=np.zeros(shape=(ir,ic,3))

resultant_sharpened_image[:,:,0]=PerformConvolution(blue_channel, kernel4)
resultant_sharpened_image[:,:,1]=PerformConvolution(green_channel, kernel4)
resultant_sharpened_image[:,:,2]=PerformConvolution(red_channel, kernel4)

resultant_sharpened_image=np.clip(resultant_sharpened_image,0,255)
resultant_sharpened_image=resultant_sharpened_image.astype("uint8")


cv.imwrite('sharpened image.jpg', resultant_sharpened_image)
resultant_sharpened_image=cv.cvtColor(resultant_sharpened_image,cv.COLOR_BGR2RGB)
plt.imshow(resultant_sharpened_image)























