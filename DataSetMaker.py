import os
import shutil
import cv2
import numpy as np
import random
import glob
import time
import sys
import copy
from PIL import Image,ImageOps
from tensorflow.python import keras
#from tensorflow.python.keras.layers.normalization import *
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.preprocessing.image import *
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.applications.vgg16 import VGG16


import tensorflow as tf
from PIL import Image,ImageOps
from tensorflow.python import keras
#from tensorflow.python.keras.layers.normalization import *
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.preprocessing.image import *
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.utils import plot_model
os.environ["OMP_NUM_THREADS"] = "1"
sys.setrecursionlimit(10000)

up = 0
down = 0
right= 0
left = 0

os.system("mkdir x_train")
os.system("mkdir x_test")
os.system("mkdir y_test")
os.system("mkdir y_train")



img_size_h = 416
img_size_w = 416
output_size = 416
img_size = 416
img_list = []


#背景画像の読込

img_dir_list	=	sorted(glob.glob(os.path.join("../test_large",'*'))) 
back_list2		=	sorted(glob.glob(os.path.join("./Back_Ground",'*')))


back_list		=	img_dir_list
#print(back_list)


#検出対象画像の読込


Kendou_list		=	sorted(glob.glob(os.path.join("./Detect/県道/",'*')))
Kokudou_list	=	sorted(glob.glob(os.path.join("./Detect/国道/",'*')))
Stop_list		=	sorted(glob.glob(os.path.join("./Detect/とまれ/",'*')))
Caution_list	=	sorted(glob.glob(os.path.join("./Detect/注意/",'*')))
Restrict_list	=	sorted(glob.glob(os.path.join("./Detect/制限/",'*')))

detect_list		=	[Kendou_list,Kokudou_list,Stop_list,Caution_list,Restrict_list]


draw_num		=	1																										#検出対象を書き込みたい数

#アングルリスト作成

angle_list		=	[j for j in range(360)]																										#アングルを保存するリスト

	


x_train_path	=	"./x_train"
y_train_path	=	"./y_train"


#画像の回転

def rotation(img):
	height = img.shape[0]                         												#wiener画像の高さを取得	
	width = img.shape[1]  																		#wiener画像の幅を取得
	center = (int(width/2), int(height/2))															#画像の中心を算出



	angle = random.choices(angle_list, k=1)[0]
	scale = 1.0


	
	w_rot = int(np.round(height+np.abs(np.sin(angle/180*np.pi))+width*np.abs(np.cos(angle/180*np.pi))))
	h_rot = int(np.round(height+np.abs(np.cos(angle/180*np.pi))+width*np.abs(np.sin(angle/180*np.pi))))
	size_rot = (w_rot,h_rot)

	trans = cv2.getRotationMatrix2D(center, angle , scale)
	affine_matrix = trans.copy()
	affine_matrix[0][2] = affine_matrix[0][2] -int(width/2) + int(w_rot/2)
	affine_matrix[1][2] = affine_matrix[1][2] -int(height/2) + int(h_rot/2) 
	img =cv2.warpAffine(img, affine_matrix, size_rot)
	#cv2.imshow("rotate",wiener)
	return img


def lightness(image , h , s , v ):
	image           =   cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
	image           =   np.int32(image)
	image[:,:,0]    =   image[:,:,0] * h
	image[:,:,1]    =   image[:,:,1] * s
	image[:,:,2]    =   image[:,:,2] * v
	image           =   np.clip(image, 0, 255).astype(np.uint8)
	image           =   cv2.cvtColor(image , cv2.COLOR_HSV2BGR)   
	return image


def Adjustment(img):
	size_list	=	[1,3,5,7,9]
	
	size	=	size_list[random.randint(0,len(size_list)-1)]
	img		=	cv2.GaussianBlur(img,(size,size),0)
	return img


def Placement(detect_image,back_ground_img,segment_img,detect_num):
	v 	= 	random.randint(0,(back_ground_img.shape[0]-detect_image.shape[0]))					#画像を置くポイントの設定
	h 	=	random.randint(0,(back_ground_img.shape[1]-detect_image.shape[1]))					#画像を置くポイントの設定

	alpha_s			=	detect_image[:,:,3] / 255
	alpha_l			=	1.0 - alpha_s

	for c in range(0,3):
		back_ground_img[v:v+detect_image.shape[0],h:h+detect_image.shape[1],c]	= (alpha_s * detect_image[:,:,c] + alpha_l * back_ground_img[v:v+detect_image.shape[0],h:h+detect_image.shape[1],c] )	
	
	Copy				= 	copy.deepcopy(detect_image)
	detect_image2gray	=	cv2.cvtColor(Copy , cv2.COLOR_BGR2GRAY)
	
	segment_img[v:v+detect_image.shape[0],h:h+detect_image.shape[1]]	=	np.uint8(detect_image2gray > 0)*255
	
	points, _ 						= 	cv2.findContours(np.uint8(segment_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	x, y, width, height 			= 	cv2.boundingRect(points[0])
	
	tx	=	(x + width/2)/img_size
	ty 	=	(y + height/2)/img_size
	tw	=	width/img_size
	th	=	height/img_size
	#img	=	np.uint8(copy.deepcopy(back_ground_img))
	#cv2.rectangle(img,(int((tx-tw/2)*img_size),int((ty - th/2)*img_size)),(int((tx+tw/2)*img_size),int((ty + th/2)*img_size)), (0,0,255) , 1 , 1)
	#cv2.rectangle(img,(x, y), (x+width,y+ height), (0,0,255) , 1 , 1)
	
	#cv2.imshow("image",img)
	#cv2.waitKey(0)
	return back_ground_img,(tx,ty,tw,th,detect_num)


def Perspective(detect_image):
	
	p1	=	random.randint(0,detect_image.shape[1]-1-500)
	y	=	random.randint(0,int(detect_image.shape[0]/2))
	p2	=	random.randint(p1+500,detect_image.shape[1]-1)
	pt2	=	np.float32([[p1,y],[p2,y],[0,detect_image.shape[0]],[detect_image.shape[1],detect_image.shape[0]]])
	pt1 = 	np.float32([[0,0],[detect_image.shape[1],0],[0,detect_image.shape[0]],[detect_image.shape[1],detect_image.shape[0]]])
	M 	=	cv2.getPerspectiveTransform(pt1,pt2)

	detect_image	=	cv2.warpPerspective(detect_image,M,(detect_image.shape[1],detect_image.shape[0]) )
	
	
	return detect_image

def Noise_Addition(image , mask):
	row,col,ch	= 	image.shape
	mean 		= 	0
	sigma 		=	15
	noise 		=	np.random.normal(mean,sigma,(row,col))
	noise 		=	noise.reshape(row,col)
	noise		=	noise * (mask/255)
	noise		=	noise[:,:,np.newaxis]
	noise		=	np.concatenate([noise , noise , noise] , axis = 2)
	image		=	np.int32(image) + np.int32(noise)
	image		=	np.uint8(np.clip(image , 0 , 255))
	return		image


def gamma_correction(img, gamma):
    # テーブルを作成する。
    table = (np.arange(256) / 255) ** gamma * 255
    # [0, 255] でクリップし、uint8 型にする。
    table = np.clip(table, 0, 255).astype(np.uint8)

    return cv2.LUT(img, table)

def Progress(per,num,n):
	amp = 20
	per = per*amp
	print("Progress [{:>5}/{:>5}][".format(n,num) , end = "")
	for i in range(amp):
		if(i<per):
			print("=" , end = "")
		else:
			print(" " , end = "")
	print("]\r",end = "")

def Do(num,x_path,y_path):
	old		=	0
	n		=	1
	while 1 :
		per = n/num
		Progress(per,num,n)	
		rand					=	random.randint(0,1)
		#if(rand == 1):
		back_ground_img			=	cv2.imread(back_list[random.randint(0,len(back_list)-1)],-1)								#背景画像の決定
		#else:
			#back_ground_img	=	cv2.imread(back_list2[random.randint(0,len(back_list2)-1)],-1)								#背景画像の決定

		segment_img				=	np.zeros((output_size,output_size))
		back_ground_img			=	cv2.resize(back_ground_img,(output_size,output_size))

		detect_num				=	random.randint(0,4)
		
		detect_image			=	cv2.imread((detect_list[detect_num][random.randint(0,len(detect_list[detect_num])-1)]),-1)								#Detect画像の読み出しを行う

		#p						=	random.randint(0,1)
		
		#if(p == 1):
		#	detect_image		=	Perspective(detect_image)
		h						=	random.randint(9,11) * 0.1
		s						=	random.randint(3,40) * 0.1
		v						=	random.randint(5,20) * 0.1
		detect_image[:,:,:3]	=	lightness(detect_image[:,:,0:3] , h , s , v)
		detect_image[:,:,:3]	=	Noise_Addition(detect_image[:,:,0:3] , detect_image[:,:,3])
		detect_image 			=	rotation(detect_image)
		#detect_image 			= 	gamma_correction(detect_image, gamma=(random.randint(1,20)*0.1))
		
		detect_image			=	cv2.resize( detect_image , (img_size - 10 , img_size -10) )
		#print("detect",detect_image.shape)
		try:
			size 				= 	random.randint(1,10)/10
			detect_image 		= 	cv2.resize(detect_image,(int(detect_image.shape[1]*size),int(detect_image.shape[0]*size)))
			
			back_ground_img,BB_list	= 	Placement(detect_image,back_ground_img,segment_img ,detect_num)
			back_ground_img 		= 	Adjustment(back_ground_img)	
			#back_ground_img =  cv2.resize(np.uint8(back_ground_img),(output_size,output_size))
			#segment_img= cv2.resize(np.uint8(segment_img),(output_size,output_size))
			cv2.imwrite(x_path+"/No"+str(n+1)+".bmp" , back_ground_img)
			with open( y_path + "/No" + str(n+1) + ".txt" , mode='w' ) as f:		
				f.write(str(BB_list[0]) +","+str(BB_list[1]) + "," + str(BB_list[2]) + "," + str(BB_list[3]) + "," + str(BB_list[4]))	
			n+=1
		
		except Exception as e:
			print("例外args:", e.args)
			continue
		if(num < n):
			break


num = 10000
old = 0

print("train...")																					#train画像を生成することを知らせる
Do(num,x_train_path,y_train_path)


x_test_path = "./x_test"
y_test_path = "./y_test"

print("test...")																					#train画像を生成することを知らせる
Do(int(num/10),x_test_path,y_test_path)