from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt


def read_image(path):
	imagen = imread(path,flatten = True)
	return imagen


def save_image(image, name):
	imsave(name,image,'png')



def normalize_image(image):
	return image / 255



def fix_bounds(image, offset):
	n, m = image.shape
	average = image.mean()
	newImage = np.full((n + 2*offset, m + 2*offset), average)
	for i in range(0, n):
		for j in range(0, m):
			newImage[i + offset, j + offset] = image[i, j]

	return newImage


def ftransform(image):
	image_transformation = fft2(image, axes = (0,0))
	return np.absolute(image_transformation)


def convolve_2D(origin_signal, kernel):
	# Asumiendo kernel cuadrados
	n,m = kernel.shape
	#Este offser indica la cantidad de pixeles extra que necesita la imagen para soportar las características del kernel
	offset = int(n/2)

	#Se ajustan los bordes de la señal original de acuerdo al offset
	origin_signal = fix_bounds(origin_signal, offset)


	r,s = origin_signal.shape
	#Matriz para soportar resultado de la convolución a lo largo de la señal original
	g = np.zeros((r - 2*offset, s - 2*offset))
	for k in range(0+offset,r-offset):
		for l in range(0+offset,s-offset):
			for i in range(0,n):
				for j in range(0,m):
					kOff = k - offset
					lOff = l - offset
					g[kOff,lOff] = g[kOff,lOff] + origin_signal[kOff+i,lOff+j]*kernel[i,j]

	return g

def process_image(path):
	image = read_image(path)
	image = normalize_image(image)

	#Transformadada de fourier de señal original
	original_fft = ftransform(image)
	save_image(original_fft, "original_fft")

	#Transformadada de fourier  inversa de señal orginal transformada
	#original_ifft = iftransform(original_fft)
	#save_image(original_ifft, "original_ifft")
	

	kernel_boundary_detector= np.asarray([[1, 2, 0, -2, -1]] * 5)
	kernel_gauss = (1/256)*np.asarray([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])

	#Aplicación de convolución
	result_gauss = convolve_2D(image,kernel_gauss)
	result_boundary = convolve_2D(image,kernel_boundary_detector)
	save_image(result_gauss,"gauss")
	save_image(result_boundary,"boundary")

	#Aplicación de transformada de fourier de dos dimensiones
	gauss_fft = ftransform(result_gauss)
	boundary_fft = ftransform(result_boundary)
	save_image(gauss_fft,"gauss_fft")
	save_image(boundary_fft, "boundary_fft")

	#Transformadada de fourier  inversa de señal con gauss transformada
	#gauss_ifft = iftransform(gauss_fft)
	#save_image(gauss_ifft, "gauss_ifft")

	#Transformadada de fourier  inversa de señal con boundary transformada
	#boundary_ifft = iftransform(boundary_fft)
	#save_image(boundary_ifft, "boundary_ifft")

	plt.show()

##PROCESAMIENTO	
process_image("leena512.bmp")

