import torch
import numpy as np
from config import *
from dataset import *
import random

device = "cuda" if torch.cuda.is_available() else "cpu"


''' Parse file names
-- Training
	'red_rubber_cylinder_shade_base_stretch_normal_scale_large_brightness_normal_view_-2_-2_2'
-- Testing
	'yellow_rubber_sponge_shade_base_stretch_normal_scale_small_brightness_dim_view_1.5_1.5_3'

	** 0_1_2
	red_rubber_cylinder_
	** 3_4
	shade_base_
	** 5_6
	stretch_normal_
		** 7_8
		scale_large_
	** 9_10
	brightness_normal_
		** 11_12_13_14_
		view_-2_-2_2_
	** x.png
	rgba.png

	diffn_list = [(i,j) for i, j in zip(n1, n2) if i != j]
	x[3:6] = [''.join(x[3:6])]
'''


def pareFileNames(base_name):
	# get different attr names
	n1 = base_name.split('_')
	# regroup shape names with '_', e.g. torus_knot
	if len(n1) > 15:
		n1[2:4] = ['_'.join(n1[2:4])]
	# regroup view points
	n1[12:15] = ['_'.join(n1[12:15])]

	nm1 = {}
	nm1["color"] = n1[0]
	nm1["material"] = n1[1]
	nm1["shape"] = n1[2]
	nm1["shade"] = n1[4]
	nm1["stretch"] = n1[6]
	nm1["scale"] = n1[8]
	nm1["brightness"] = n1[10]
	nm1["view"] = n1[12]

	return nm1


def get_mse_loss(recons, images):
	recons_loss = F.mse_loss(recons, images)
	return recons_loss


def get_mse_loss_more(recons, images):
	recons_loss = 0
	for i in range(images.shape[0]):
		recons_loss += F.mse_loss(recons[0], images[i])
	return recons_loss/images.shape[0]


def get_sim_loss(z):
	centroid = torch.mean(z, dim=0)
	loss = 0
	for i in range(z.shape[0]):
		loss += F.mse_loss(centroid, z[i].view(-1))

	return centroid, loss/z.shape[0]


def get_sim_not_loss(centroid, z):
	loss = 0
	for i in range(z.shape[0]):
		loss += F.mse_loss(centroid, z[i].view(-1))

	return loss/z.shape[0]

def get_cos_sim(a,b):
	return torch.nn.functional.cosine_similarity(a, b, dim=1)

def h_get_sim_loss(z, centroid):
	loss = 0
	for i in range(z.shape[0]):
		loss += F.mse_loss(centroid, z[i].view(-1))

	return loss/z.shape[0]

class Buffer:
	def __init__(self, alpha:float, beta:float, size:int):
		self.data = []
		self.numbers = []
		self.largest_idx = None
		self.size = size
		self.alpha = alpha
		self.beta = beta

	def get_sample(self) -> object:
		if len(self.data) > 1:
			idx = int(random.uniform(0, len(self.data)-1))
			return self.data[idx]
		else: return None
		
	def add_sample(self, x:object) -> None:
		""" Add new object with reservoir strategy """
		random_no = random.uniform(0, 1)
		if len(self.data) < self.size: 
			self.data.append(x)
			self.numbers.append(random_no)
			if self.largest_idx is None or random_no > self.numbers[self.largest_idx]:
				self.largest_idx = len(self.data) - 1
		elif random_no < self.numbers[self.largest_idx]:
			self.data[self.largest_idx] = x
			self.numbers[self.largest_idx] = random_no
			self.largest_idx = np.argmax(self.numbers)
