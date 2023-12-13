
#!/usr/bin/env python3
import cv2
from os.path import exists
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import math 
import scipy.misc
from libSunPos1 import *
from utils import *
# from library_lucas_kanade import *
# from library_undistort_aberration import *
# from library_image_processing_sun_image import *  # step 2
# from library_clouds_labels import *
# from library_clear_sky_models_panda import *
# from library_tracking_in_original_from_iams_features import *
# from library_rbf import *
# import datetime
# import numba
# from numba import jit


# from matplotlib import rc
# rc('font', family='serif')

''' resolucion variable muy importante '''
#---------- Concideraaciones
#  -la variable time_fix tiene 2 compesaciones de tiempo, eso es por que el RTC que maneje se atrazaba en minutos cada dia.
#	y el cambio de horario cabia, lo programe para el horario de verano y despuedes del 27 de oct suele cambiar, por lo que
# 	utilizo -1 o  -0 para compenzar la hora
#  -path_image variable de direccion de imagen
#-----------------------------------------------------------------------------------------------------------------------------

''' 1. Arreglar la dircloud_transmitance_data = pd.read_csv('spilineFile.csv', sep=",")
	
	rbf_tau_c = Rbf(	cloud_transmitance_data.V_Tc_mean.values,
							cloud_transmitance_data.V_Tc_std.values,
							cloud_transmitance_data.tau_c.values,
							epsilon=2)
	ti_Tc_mean = np.linspace(cloud_transmitance_data.V_Tc_mean.values.min(), cloud_transmitance_data.V_Tc_mean.values.max(), 500)
	ti_Tc_std = np.linspace(cloud_transmitance_data.V_Tc_std.values.min(), cloud_transmitance_data.V_Tc_std.values.max(), 500)
	
	XI, YI = np.meshgrid(ti_Tc_mean, ti_Tc_std)
	ZI = rbf_tau_c(XI, YI) 
	
	fig = plt.figure()
	ax = plt.axes(projection="3d")	 
	ax.plot_surface(XI, YI, ZI, color="red")
	
	ZI = rbf_tau_c(786.8,46.99594643560088 ) 
	print(f'el valor de z interpolado: {ZI}')
	plt.show()eccion de la imagen '''
def imagePathFix(_nameImage,_minutes):  # Output -> [year, month, day, hour, minute, timeExposition]
	nameImage = _nameImage
	i = _minutes
	
	vect_time1 = nameImage.split('.')
	vect_time = vect_time1[0].split('_')
	date_image = vect_time[0].split('-')
	vect_time = date_image + vect_time[1:]	
	time_fix = int(vect_time[3]) *60 + int(vect_time[4]) + i 
	hour = int(time_fix/60)
	minute = time_fix % 60
	
	
	pathNew = vect_time[0]+'-'+vect_time[1]+'-'+vect_time[2]+'_'+str(hour)+'_'+ str(minute)+'.'+vect_time1[1]
	return(pathNew)

def imagePathFix_aux(_nameImage,_minutes):  # Output -> [year, month, day, hour, minute, timeExposition]
	path_image = _nameImage
	i = _minutes
	
	vect_time = [		path_image[0:4], 
							path_image[4:6],
							path_image[6:8],
							path_image[8:10],
							path_image[10:12] ,
							path_image[12:] ,
							  ]   	
	
	time_fix = int(vect_time[3]) *60 + int(vect_time[4]) - i 
	hour = int(time_fix/60)
	minute = time_fix % 60
	if hour < 10:
		hour = '0'+str(hour)
	if minute < 10:
		minute = '0'+str(minute)	
		
	pathNew = vect_time[0]+'-'+vect_time[1]+'-'+vect_time[2]+'_'+str(hour)+'_'+ str(minute)+'_'+vect_time[5]
	print("asfsdf" ,vect_time[5])
	return(pathNew)

def next_path(_nameImage,_minutes):  # Output -> [year, month, day, hour, minute, timeExposition]
	path_image = _nameImage
	i = _minutes
	
	vect_time = [		path_image[0:4], 
							path_image[4:6],
							path_image[6:8],
							path_image[8:10],
							path_image[10:12] ,
							path_image[12:] ,
							  ]   	
	time_fix = int(vect_time[3]) *60 + int(vect_time[4]) + i 
	hour = int(time_fix/60)
	minute = time_fix % 60
	if hour < 10:
		hour = '0'+str(hour)
	if minute < 10:
		minute = '0'+str(minute)	
	pathNew = vect_time[0]+vect_time[1]+vect_time[2]+str(hour)+ str(minute)+vect_time[5]
	return(pathNew)



#--------------------------------------------------------------------------------------------------------------
# 	1.1 library_undistort_aberration
#--------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------
''' 3. Rellenar negros '''
def rellenar_pixel_negro_Nubes_alfa_beta_2( img1 ):
	''' copy image, this image will be modified '''
	image = img1.copy()
	max_dist_permitida = 25 #pixeles maximos permitidos
	
	i_row, i_col = image.shape[:2]
	# print(i_row)
	for i in range(i_row):
		# extract row
		row_img = img1[ i,:,0 ]
		# extract position of empty pixels
		zero_pos = np.where(row_img == 0)[0]
		non_zero_pos = np.where(row_img != 0)[0]
		#revisar si la longitud del vector zeros es distinta a la cantidad de columnas 
		if zero_pos.shape[0] != i_col: 
			for j in zero_pos: #for each
				
				der_vec = np.where( non_zero_pos > j  )[0]
				izq_vec = np.where( non_zero_pos < j  )[0]
				
				
				rojo , verde , azul = 0 , 0 , 0
				if len(der_vec) != 0 and len(izq_vec) != 0: 
					der_pos = non_zero_pos[np.where(non_zero_pos > j  )[0][0] ]
					izq_pos =non_zero_pos[  np.where(non_zero_pos < j  )[0][-1] ]
					if der_pos - j < max_dist_permitida and j - izq_pos < max_dist_permitida:
						rojo	+= img1[ i , der_pos , 0 ]
						verde	+= img1[ i , der_pos , 1 ]
						azul	+= img1[ i , der_pos , 2 ]
						rojo	+= img1[ i , izq_pos , 0 ]
						verde	+= img1[ i , izq_pos , 1 ]
						azul	+= img1[ i , izq_pos , 2 ]
					
						image[ i , j , 0 ]	=	rojo / 2
						image[ i , j , 1 ]	=	verde / 2
						image[ i , j , 2 ]	=	azul / 2
	
	return( image )

def plot_day_before_data( day_before_data ):
	day_before_data['GHI'].plot.line( ) #title= 'GHdsaI')
	day_before_data['I_bn_ineichen'].plot.line(  )
	day_before_data['SSPC'].plot.line( )
	day_before_data['theta_z'].plot.line( )
	
	plt.legend(['$I_{G_{meas}}$', '$I_{bn_{Ineichen}}$', '$I_{bn_{SSPC}}$'] )
	# plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
	plt.xlabel("Time", fontsize=14,color='black' ,  usetex=True)
	plt.ylabel("Irradiance [$W/m^2$]", fontsize=14,color='black' ,  usetex=True)
	plt.grid(True)
	plt.xticks(fontsize=12 ,  usetex=True),plt.yticks(fontsize=12 ,  usetex=True)
	plt.tight_layout()
	plt.savefig('I_bn_sintetitation.pdf')
	# plt.rcParams["text.usetex"] =True
	# plt.show()
	plt.close()

def compute_SSPC_vector(path_image, a_sspc,b_sspc,I_ext):
	return( 	[compute_SSPC_using_a_b( get_theta_z_value(path_image), a_sspc,b_sspc,I_ext) ,
				compute_SSPC_using_a_b( get_theta_z_value(imagePathFix(path_image, 1)), a_sspc,b_sspc,I_ext),
				compute_SSPC_using_a_b( get_theta_z_value(imagePathFix(path_image, 3)), a_sspc,b_sspc,I_ext),
				compute_SSPC_using_a_b( get_theta_z_value(imagePathFix(path_image, 5)), a_sspc,b_sspc,I_ext),
				compute_SSPC_using_a_b( get_theta_z_value(imagePathFix(path_image, 10)), a_sspc,b_sspc,I_ext)]
			)

def sunVector( _path_image, _rSky, _rMirror, _minutosPrediccion ):
    path_image = _path_image
    rSky = _rSky
    rMirror = _rMirror
    vector_time = vectorTimeImage(path_image)  # [year, month, day, hour, minute, timeExposition]	
    
    pySolarSun = sunPosFunc(vector_time) # [date, h:m, azimut, altitude]
    # print(vector_time)
    rCamera, altitudSkyMirror = altitudSkyProyectedMirror(pySolarSun[3], rSky, rMirror) #################esta tambien

    print("========================")
    print(rCamera , altitudSkyMirror)
    
    pySolarSun.append(rCamera)
    pySolarSun.append(altitudSkyMirror)
    sun = pySolarSun	
    sun = np.array( sun )		# sun = [date, time, °azimuth , °PyAltitud,  RpixImag, °Altitud  ]
    # print ('Sun pos', sun)
    return(sun)



''' 2. proceso general de creacion de mapeo angular alpha | beta '''
def imageProcessingComplete(_path_image, _pathNew, _rMirror , _rSky, _resolucion, _minutosPrediccion, _i  ):  
    path_pysolar = _path_image
    pathNew = _pathNew
    rSky = _rSky
    rMirror = _rMirror
    resolucion = _resolucion
    minutosPrediccion = _minutosPrediccion + _i

    #----   2.1  Modificacion de aberracion de imagen     ---- 
    img_cropOrginal, img_RGB_Sun , cosine_mask = imageProcessing(
		pathNew,
		rSky, #rMirror,
		rSky,
		minutosPrediccion
	)
    cloudImage = extractOnlySky(img_RGB_Sun, _rSky)
    

    # plt.imshow(img_RGB_Sun, 'gray'); plt.show()
    # a = 1/0
    # __import__('pdb').set_trace()
    # plt.imshow(cosine_mask, 'gray'); plt.show()
    # plt.imshow(img_cropOrginal, 'gray'); plt.show()
    
    #----     Vector de sol    ----
    sun = sunVector(path_pysolar, rSky, rMirror, minutosPrediccion) 		# sun = [date, time, °azimuth , °PyAltitud,  RpixImag, °Altitud  ]

    rCamera  =  float(sun[4])
    azimut   =  float(sun[2])
    altitud  =  float(sun[3])
    ySun = rCamera* np.cos(np.radians(azimut))  # azimuth referido al sur
    xSun = rCamera* np.sin(np.radians(azimut))
    zSun = rMirror# rCamera* np.tan(np.radians(altitud))
    '''       sunsCoordenatesInImage = [ x  , y , z ]  ''' 
    sunsXYZ = sunsXYZfunction(xSun , ySun ,zSun)
    # print(sunsXYZ)
    '''    Funcion de vector de nubes con posicion angular    '''
    image_cloud_position_angular = vectorCloudAngularDistance(cloudImage, sunsXYZ, rMirror, resolucion)
    return (img_cropOrginal,  img_RGB_Sun, cloudImage, image_cloud_position_angular , cosine_mask)


''' 2.1 '''
def imageProcessing(_path_image , _rMirror , _rSky, _minutosPrediccion ):
	path_image = _path_image
	rMirror = _rMirror
	rSky = _rSky
	minutosPrediccion = _minutosPrediccion

	img1 = cv2.imread(path_image)	

	#rotar img?
	img1 = rotate_image(img1, 0)  
	
	vector_time=	vectorTimeImage(path_image)  # [year, month, day, hour, minute, timeExposition]	
	img_crop	=	cropImage(img1,_rMirror ) 			## falta aplicar maskara negra
	img_crop	=	cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)	
	# cv2.imwrite('img_crop.png',img_crop)
	# plt.imshow(img_crop, 'gray') 	
	# plt.show()
	
	imgNoDist	=	undistort(img_crop)				# modifico aberracion
	# cv2.imshow('maskRedvis6', imgNoDist)
	img_no_distorcion = imgNoDist 
	img_RGB_Sun , sun_position_x_y= sunPositionGrap(img_no_distorcion, rSky,rMirror, vector_time, minutosPrediccion)  # grafica posicion del sol
	
	''' create cosine mask without sun '''
	h,w,_ = img_crop.shape
	cosine_mask = create_cosine_mask(h, w, rSky, sun_position_x_y, 10)
	
	plt.imshow(img_RGB_Sun, 'gray') 
	plt.show()
	
	return (img_crop, img_RGB_Sun , cosine_mask)

def get_theta_z_value( _path_image ):
    path_image = _path_image
    vector_time = vectorTimeImage(path_image) # [year, month, day, hour, minute, timeExposition]	
    pySolarSun = sunPosFunc(vector_time) # [date, h:m, azimut, altitude]
    return(90- pySolarSun[3])


def vectorTimeImage(_nameImage):  # Output -> [year, month, day, hour, minute, timeExposition]
    nameImage = _nameImage
    nameImage = nameImage.split('.')
    vect_time = nameImage[0].split('_')
    date_image = vect_time[0].split('-')
    vect_time = date_image + vect_time[1:]
    return(vect_time)



#lat, lon	=	19.326754 , -99.175553
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
def main():
	'''--------------------------------------------------------------------------------'''
	# analisys_day = '2022-05-10'
	# start_time = '_14_3.jpg'

	# Older image
	analisys_day = '2019-10-25'
	start_time = '_13_22.jpg'

	'''--------------------------------------------------------------------------------'''
	# rMirror , rSky =  730 , 610 #730 , 665+50#768 , 800
	rMirror , rSky =  610 , 610 #730 , 665+50#768 , 800
	resolucion = 3
	predictionMinutes = 1# 5	# minutos de prediccion para momento cero
	

	path_image = analisys_day+ start_time
	
	''' --------------------------------------------------- '''
	''' ------------- Creation of AMIs -------------- '''
	''' --------------------------------------------------- '''
	''' 1. '''
	print(path_image)

	pathNew = path_image
	''' 2. '''
	''' path_image  va a pysolar ||| pathNew -> nueva imagen consecutiva en minutos '''
	a, b, c, d,f = imageProcessingComplete(path_image, pathNew, rMirror , rSky, resolucion, predictionMinutes, -1 )  
	''' 3. '''
	iams_fixed = rellenar_pixel_negro_Nubes_alfa_beta_2 (d.copy()) 
	plt.imshow(iams_fixed, 'gray') 
	plt.show()
	


if __name__ == '__main__':
    main()
    
