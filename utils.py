#!/usr/bin/env python3
import cv2
import numpy as np
from libSunPos1 import *
# from library_undistort_aberration import *
# import numba
# from numba import jit
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='serif')

def rotate_image(_img, angle):
	image = _img
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return( result )

def cropImage(img , r ):
	h , w	=	img.shape[:2]
	h		=	int(h/2)
	w 		=	int(w/2)
	aju1 = 0#30
	aju2 = 0#55
	
	img_crop = img[ h - r +aju1 : h + r+aju1 , w - r+aju2 : w + r+aju2 ]	

	# ~ plt.imshow(img_crop, 'gray') 
	# ~ plt.show()	
	# ~ cv2.imwrite('img_crop.png',img_crop)
	
	'''
	# r es r_mirror para poder modificar las aberraciones en pasos siguientes
	w , h	=	img.shape[:2] 
	# ~ cv2.line(img,(int(h/2),0),(int(h/2),int(w)),[0,0,255],6)
	# ~ cv2.line(img,(0,int(w/2)),(int(h),int(w/2)),[0,0,255],6)
	h	=	int( h / 2 )
	w	=	int( w / 2 )
	# ~ img_crop = img# h - r : h + r , w - r : w + r ]
	img_crop = img[ 35 -10: 305  +10, 10 -10: 285 +10]	
	'''
	return (img_crop)

def nothing(x):
    pass

def undistort(img_path):
	
    """
    # ===== Borrar
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('undistorted')
    a = 4539
    cv2.createTrackbar('Deformacion','undistorted',1524,a,nothing)
    cv2.createTrackbar('x','undistorted',150,300,nothing)
    cv2.createTrackbar('y','undistorted',150,300,nothing)

    cv2.createTrackbar('Centro2','undistorted',0,3000,nothing)
    cv2.createTrackbar('Para Centro','undistorted',914,1000,nothing)
    cv2.createTrackbar('Para Centro2','undistorted',2572,4000,nothing)
    cv2.createTrackbar('Radio','undistorted',0,1000,nothing)

    # ========

    cam = img_path
	# ~ #~ cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
	# ~ #~ cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
	# ~ cam.set(cv2.CAP_PROP_FRAME_WIDTH,640 )
	# ~ cam.set(cv2.CAP_PROP_FRAME_HEIGHT,  .480)
	# ~ os.system('v4l2-ctl --set-ctrl=exposure_auto=1')
	# ~ #cam.set(15, 0.0002)#0.02 recomendado
	# ~ cam.set(15, 1000)#0.02 recomendado
	# ~ cv2.waitKey(5)	
	
	# ~ 221-100+60+200
	# ~ 19
	# ~ 146
	# ~ radio 134
	
    while(1):
        #-----------------------------------------------------------------------------------------------#		
        k_x = cv2.getTrackbarPos('Deformacion','undistorted')
        Cx = cv2.getTrackbarPos('x','undistorted')-150
        Cy = cv2.getTrackbarPos('y','undistorted') -150
        Cc1=cv2.getTrackbarPos('Centro2','undistorted')
        Cc = cv2.getTrackbarPos('Para Centro','undistorted')-500
        Cc2=cv2.getTrackbarPos('Para Centro2','undistorted')-2000
        
        
        #-----------------------------------------------------------------------------------------------#
        frame = img_path
        img = frame
        # ~ img = img[ 34:320 , 5:283 ]	
        a,b,c = img.shape
        DIM	=	( b , a )

        r	=	k_x #200 + 60 + 
        Cxx = Cx
        Cyy = Cy
        K= (np.array([[r, 0.0, int(b/2)+Cxx], [0.0, r, int(a/2)+Cyy], [0.0, 0.0, 1.0]])  )
        D=np.array([[Cc1], [0.0], [Cc], [Cc2]])
        
        print('K' , K)
        print('D' , D)
        #-----------------------------------------------------------------------------------------------#
        print (r)

        
        new_K =	cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM,  np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        #new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM,  np.eye(3), balance=0.0)
        # ~ map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K,  DIM, cv2.CV_16SC2)
        
        # ~ undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        print (undistorted_img.shape)
            
        RR1 = int ( undistorted_img.shape[1]/2 )
        RR2 = int(undistorted_img.shape[0]/2 )
        # ~ undistorted_img = cv2.resize(undistorted_img, ( RR1 , RR2 ))
        a,b,c = undistorted_img.shape
        cv2.line(undistorted_img,(int(b/2),0),(int(b/2),int(a)),[0,0,255],1)
        cv2.line(undistorted_img,(0,int(a/2)),(int(b),int(a/2)),[0,0,255],1)
        
        center_coordinates = (int(b/2), int(a/2))
        radius = cv2.getTrackbarPos('Radio','undistorted')
        color = (255, 0, 0)
        cv2.circle(undistorted_img, center_coordinates, radius, color, 2)
        
        undistorted_img1 = cv2.resize(undistorted_img, (int(b/3),int(a/3)), interpolation=cv2.INTER_AREA)
        cv2.imshow("undistorted", undistorted_img1)
        cv2.waitKey(1)

	"""

    img = img_path
    h,w = img.shape[:2] 
    #-----------------------------------------------------------------------------------------------#

    DIM=(w, h)
    r = 1524 # contanste de aberracion esferica
    rr = 710 - 100
    Cc0, Cc1 , Cc2, Cc3 = 8.0 , 0.0 , 414,572


    K=np.array([[r, 0.0, rr+00.0], [0.0, r, rr+0.0], [0.0, 0.0, 1.0]])
    D=np.array([[Cc0], [Cc1], [Cc2], [Cc3]])

    # K [[1.411e+03 0.000e+00 6.020e+02]
    # [0.000e+00 1.411e+03 6.010e+02]
    # [0.000e+00 0.000e+00 1.000e+00]]
    # D [[  0.]
    # [  0.]
    # [382.]
    # [ 27.]]

    #-----------------------------------------------------------------------------------------------#
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM,  np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # ~ cv2.imshow('unidist', img)
    # ~ cv2.imshow('dist', undistorted_img)

    return (undistorted_img)
#----------------------------------------


def sunPositionGrap(_img, _rSky, _rMirror, _vector_time , _minutosPrediccion):
	# --------------------------------------------
	img 		= 	_img
	sun 		= 	[]
	altitude 	= 	[]
	rMirror 	=	_rMirror
	rSky 		=	_rSky 	# sky sphere radio [ pixels ]
	minutosPrediccion = _minutosPrediccion
	# --------------------------------------------	
	pySolarSun = sunPosFunc(_vector_time)
	rCamera , altitudSkyMirror =  altitudSkyProyectedMirror(pySolarSun[3], rSky, rMirror)
	pySolarSun.append(rCamera)
	pySolarSun.append(altitudSkyMirror)
	sun.append(pySolarSun)	
	# ~ print('sun:' , sun )
	sun = np.array(sun)  # sun = [date, time, °azimuth , °PyAltitud,  RpixImag, °Altitud  ]
	# ~ print('sun:' , sun )
	y_m,x_m = img.shape[:2] # Zeros 
	''' ------- Draw Sun ----------- '''
	sunaux = 0
	rCamera = float(sun[sunaux,4])
	azimut = float(sun[sunaux,2] )
	y	=	rCamera* np.cos(np.radians(azimut))
	x	=	rCamera* np.sin(np.radians(azimut))
	dot_coordinates = ( int(x_m/2 - x ) , int( y_m/2 + y ) ) 
	radius		= 15
	color			=	(255, 0,0 ) 
	thickness	=	3
	
	# ~ cv2.circle(img, dot_coordinates, radius, color, thickness) 	
	# ~ plt.imshow(img, 'gray') 
	# ~ plt.show()
	return(img ,dot_coordinates)

def sunPosFunc(_vector_time):
    defase_pysolar = -22
    #CIO-Ags --> LAT, LON
    lat, lon = 21.844741 , -102.3438367
    year = int(_vector_time[0])
    month = int(_vector_time[1])
    day = int(_vector_time[2])
    second = 0	

    _hour = int(_vector_time[3])
    _minute = int(_vector_time[4])
    time_fix = ( _hour - 1-1 ) * 60 + ( _minute + defase_pysolar)  ########## cambiar signo por que voy al pasado
    hour = int(time_fix/60)
    minute = time_fix % 60

    date_ = [year, month, day]
    time_ = [hour, minute, second]

    sunpos = solarPos(lat, lon, date_, time_)
    sunpos.compute_position()
    azimut = sunpos.get_azimuth_values()
    altitude = sunpos.get_altitude_values()
        
    dateDefine = str(month) + '/' + str(day)+'/'+str(year) 
    dateDefine2 = str(hour)+':'+str(minute)
    print('---------------------')
    print ('Azimut:   ', azimut)  
    print ('altitud:  ', altitude)
    print('---------------------')
    sun_values = [ dateDefine , dateDefine2 , round( azimut[0]-180 , 3 ) , round(altitude[0] , 3 ) ]
    return ( sun_values )

#-------------- transformacion de esferas en plano
def altitudSkyProyectedMirror( _altitude , _rSky , _rMirror ):
	altitud = _altitude
	rSky = _rSky
	rMirror = _rMirror
	
	skyX = rSky * np.cos( np.radians( altitud ) )
	mirrorP = [skyX, (rMirror**2 - skyX**2 )**0.5 ] # [x mirror , y mirror]
	angleInMirror = np.arctan(mirrorP[1] / mirrorP[0] )	
	rCamera = rMirror * np.cos( angleInMirror )
	angleInMirror = 180 * angleInMirror / np.pi
	# ~ print('angle mirror: ', angleInMirror)
	return(round(rCamera,3), round(angleInMirror,3))

def create_cosine_mask(h, w, radius, position_sun, radius_sun):
	center = (int(w/2), int(h/2))
	y, x = np.ogrid[:h, :w]
	d = np.sqrt((x - center[0])**2 + (y-center[1])**2)
	d = d / radius 
	d=1-d
	d[d<=0] = 0
	#sun blocked
	d2 = np.sqrt((x - position_sun[0])**2 + (y-position_sun[1])**2)
	mask = d2 > radius_sun
	mask_whitout_sun = mask*d
	return mask_whitout_sun

def sunsXYZfunction(_xSun, _ySun, _rSky):
	suns = [_xSun, _ySun]
	suns = np.array(suns)
	suns = - suns
	radio = _rSky
	z_sun = (radio**2 - suns[0]**2 - suns[1]**2)**0.5
	sunsXYZ = np.append(suns, z_sun)	
	return sunsXYZ



''' 2.2 ''' 
''' Extrae solamente la imagen del cielo, quita aquello que no esta en el domo '''
def extractOnlySky(_img, _rSky): 
	img 			=	_img
	rSky			=	_rSky
	cloudImage = cloudProcessOnlySky(_img, _rSky)
	return(only_sky_out(_img , cloudImage))

def cloudProcessOnlySky(_img, _rSky):  # es mejorable con un mascara cirular
	img 			=	_img
	rSky			=	_rSky
	h = img.shape[0]
	w = img.shape[1]
	X, Y = np.ogrid[:h, :w]
	dist_from_center = np.sqrt((X - h/2)**2 + (Y - w/2)**2)
	mask = dist_from_center <= rSky# + 180 #180 ES EL TAMAMO DE LA MASCARA CIRCULAR?
	return np.array(mask, np.uint8 )

def only_sky_out(_img , _cloudImage):
	img , cloudImage = _img , _cloudImage
	B, G, R 		= cv2.split(img)
	output1_R 	= R*cloudImage
	output1_G 	= G*cloudImage
	output1_B 	= B*cloudImage
	img 			= cv2.merge( ( output1_B, output1_G, output1_R ) )
	return(img)

def vectorCloudAngularDistance(_cloudImage, _suns, _rSky, _resolucion):
	'''
	suns posicion en xyz no en pixeles
	j i posicion en pixel que se transforama a xyz
	variable ---resolucion--- es para aumentar los pixeles de la grafica
	'''
	cloudImage	=	_cloudImage
	suns			=	_suns    # xSun , ySun ,zSun
	radio			= 	_rSky
	diametro		=	radio*2
	resolucion	=	_resolucion#*2
	
	# ~ plt.imshow(cloudImage, 'gray') 
	# ~ plt.show()
	
	sunsMag	=	np.sqrt(suns.dot(suns))
	betaAngleMax	=	150#120#120
	alfaAngleMax	=	360 							#Modificarrrrrrrrrr
	h	=	cloudImage.shape[0]
	w	=	cloudImage.shape[1]

	''' Generar imagenes de nubes en angulo beta vs angulo gama'''
	image_cloud_position_angular = np.zeros(( (betaAngleMax  ) * resolucion ,alfaAngleMax * resolucion , 3 ), np.uint8)    # genera imagen de nubes 
	for j in np.arange(h): #np.arange(0, h , 1/resolucion):#
		for i in np.arange(w): #np.arange(0 , w , 1/resolucion):#
			# ~ if cloudImage[int(j),int(i),0] != 0:				
			#---- j i cambian de pixel a xyz --#	
			x = i - radio
			y = -j + radio
			z = (radio**2 - x**2 - y**2)**0.5	
			XYZpixelPosition = np.array((x,y,z))
			#----------------------------------#
			''' Calculo de alpha en xy y Beta en xyz '''			
			dotA = np.dot(suns,XYZpixelPosition) / radio / sunsMag			

			betaAngle = np.arccos(dotA)*180/np.pi
			pMinusSun = XYZpixelPosition - suns
			gamaAngle = np.arctan2( pMinusSun[1].real , pMinusSun[0].real ) * 180 / np.pi
			if gamaAngle < 0: 
				gamaAngle += 360
			''' generar zona de interes '''	
			if betaAngle <= betaAngleMax :
				image_cloud_position_angular[np.int_(betaAngle * resolucion), np.int_(gamaAngle * resolucion ),:] = cloudImage[int(j),int(i),:]
	

    # ~ plt.imshow(image_cloud_position_angular, 'gray') 
	# ~ plt.show()	
	return(image_cloud_position_angular )	