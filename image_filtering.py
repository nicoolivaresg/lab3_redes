from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np


def read_image(path):
	imagen = imread(path,flatten = True)
	return imagen


def save_image(image, name):
	imsave(name,image,'png')



def normailize_image(image):
	return image / 255


def fix_bounds(image, offset):
	n, m = image.shape
	average = image.mean()
	newImage = np.full((n + 2*offset, m + 2*offset), average)
	for i in range(0, n):
		for j in range(0, m):
			newImage[i + offset, j + offset] = image[i, j]

	return newImage

def convolve_2D(origin_signal, kernel):
	# Asumiendo kernel cuadrados
	n,m = kernel.shape
	offset = int(n/2)

	origin_signal = fix_bounds(origin_signal, offset)

	r,s = origin_signal.shape
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
	image = normailize_image(image)
	kernel_boundary_detector= np.asarray([[1, 2, 0, -2, -1]] * 5)
	kernel_gauss = (1/256)*np.asarray([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
	result_gauss = convolve_2D(image,kernel_gauss)
	result_boundary = convolve_2D(image,kernel_boundary_detector)
	save_image(result_gauss,"gauss")
	save_image(result_boundary,"boundary")

process_image("leena512.bmp")

