import os,sys,time	#подгружаем системные библиотеки
import numpy as np	#подгружаем библиотеку для расчетов
import cv2			#и наконец opencv

fn = "C:\\Users\Artur\Desktop\J1.tif"
print(fn)
img = cv2.imread(fn,0)	#считываем ее как серую (0)
img = cv2.medianBlur(img,5)

	
ret3,th3 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)		#адаптивное пороговое отсечение фона - 1ый вариант
																				#https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))	#морфологическое преобразование - опенинг - 2ой вариант
																				#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
#без опенинга
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th3)	#методом связанных компонент получаем пролейблированные объекты
mth3 = np.zeros(th3.shape)	#создаем черную картинку с габаритами оригинальной
for i in range(1, len(stats)):	#для визуализации решил обвести каждое пятно прямоугольником, может пригодится
		cv2.rectangle(mth3,(stats[i,0],stats[i,1]),(stats[i,0]+stats[i,2],stats[i,1]+stats[i,3]),2,-1)
print("threshold:")
print(np.mean(stats[1:,2]),np.mean(stats[1:,3]),np.mean(stats[1:,4]))	#находим среднее значение по длине/ширине/площади объектов
print(np.min(stats[1:,4]),np.max(stats[1:,4]))							#находим объем максимального/минимального по площали объекта
print(1 - stats[0,4]/(th3.shape[1] * th3.shape[0]))						#из всей картинки вычитаем нулевой лейбл(фон) - находим площадь пузырей


	
	
#с опенингом - операции точно такие же, поэтому - без комментариев
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening)
mop = np.zeros(opening.shape)#, np.int8)
for i in range(1, len(stats)):
	cv2.rectangle(mop,(stats[i,0],stats[i,1]),(stats[i,0]+stats[i,2],stats[i,1]+stats[i,3]),2,-1)
print("opening:")
print(np.mean(stats[1:,2]),np.mean(stats[1:,3]),np.mean(stats[1:,4]))
print(np.min(stats[1:,4]),np.max(stats[1:,4]))
print(1 - stats[0,4]/(mop.shape[1] * mop.shape[0]))
mop = cv2.resize(mop,(int(mop.shape[0]/2),int(mop.shape[1]/2)))
		
#находим разницу между двумя вариантами
diff = cv2.absdiff(opening,th3)
diff[opening > 0] = 0	#убираем размытые границы по опенингу
	
cv2.imwrite( "C:\\Users\Artur\Desktop\J2.jpg", opening );

import cv2
import numpy as np

img = cv2.imread('C:\\Users\Artur\Desktop\J2.jpg',0)

#препроцессинг
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
img = cv2.medianBlur(img,5)
img = cv2.medianBlur(img,5)
img = cv2.medianBlur(img,5)
img = cv2.medianBlur(img,5)

#препроцессинг группа 1
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=8,minRadius=3,maxRadius=7)

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)


#препроцессинг группа 2
circles_2 = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=10,minRadius=8,maxRadius=11)

for i in circles_2[0,:]:
    # draw the outer 
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)



#препроцессинг группа 3
circles_3 = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=15,minRadius=12,maxRadius=40)

for i in circles_3[0,:]:
    # draw the outer 
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

import numpy as np
 
cv2.imshow('detected circles',cimg)

cv2.imwrite("C:\\Users\Artur\Desktop\J5.png", cimg);

print (circles)
print (circles_2)
print (circles_3)
type (circles)



for i in circles[0,:]:
    ramp = i[2]
    print (ramp)
type (ramp)   
 

for i in circles_2[0,:]:
    print (i[2])
    
for i in circles_3[0,:]:
    print (i[2])



cv2.waitKey(0)	#выходим по нажатию любой клавиши
cv2.destroyAllWindows()	#закрываем все окна



