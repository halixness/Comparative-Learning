import os
import torch
import clip
import time
import pickle
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import numpy as np
import wandb
import torch.nn.functional as F

from torch.utils.data import DataLoader
from polytropon import SkilledMixin

from config import *
from dataset import *
from models import *

random.seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
wandb_run = None

def get_buffer_distribution(buffer):
	if len(buffer.data) > 0:
		notions = {}
		for sample in buffer.data:
			notions.setdefault(sample["x_lesson"], 0)
			notions[sample["x_lesson"]] += 1 / len(buffer.data)
		return notions
	else: return None

class Buffer:
	def __init__(self, alpha:float, beta:float, size:int, warmup:int=1):
		self.data = []
		self.numbers = []
		self.largest_idx = None
		self.size = size
		self.alpha = alpha
		self.beta = beta
		self.warmup = warmup

	def get_sample(self) -> object:
		if len(self.data) > self.warmup:
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

def my_concept_fwdpass(dt, model, attr, lesson, memory, task_id, buffer, progressbar, epoch):

	task_ids = torch.LongTensor([task_id] * sim_batch).to(device) # [task_id], it won't matter?

	# get model
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	
	model.train()

	loss_sim = None
	loss_dif = None
	loss = 10
	ct = 0
	
	if lesson not in memory.keys():
		centroid_sim = torch.rand(1, latent_dim).to(device)
	else: centroid_sim = memory[lesson]["centroid"].to(device)

	# Get Inputs: sim_batch, (sim_batch, 4, 128, 128)
	base_name_sim, images_sim = dt.get_better_similar(attr, lesson)
	images_sim = images_sim.to(device)
	with torch.no_grad():
		sim_emb = clip_model.encode_image(images_sim).float()

	# run similar model
	z_sim = model(task_ids, sim_emb)
	centroid_sim = centroid_sim.detach()
	centroid_sim, loss_sim = get_sim_loss(torch.vstack((z_sim, centroid_sim)))

	# Run Difference
	base_name_dif, images_dif = dt.get_better_similar_not(attr, lesson)
	images_dif = images_dif.to(device)
	with torch.no_grad():
		dif_emb = clip_model.encode_image(images_dif).float()
	
	# run difference model
	z_dif = model(task_ids, dif_emb)
	loss_dif = get_sim_not_loss(centroid_sim, z_dif)

	# Dark Experience Replay (++)
	reg = None
	if buffer is not None:
		sample1 = buffer.get_sample()
		sample2 = buffer.get_sample()
		if sample1 and sample2:
			# Component 1: matching the logits
			n1_z_sim = model(task_ids, sample1["sim_emb"].to(device))
			n1_z_dif = model(task_ids, sample1["dif_emb"].to(device))
			reg_loss1 = buffer.alpha * (F.mse_loss(n1_z_sim, sample1["z_sim"].to(device)) + F.mse_loss(n1_z_dif, sample1["z_dif"].to(device)))
			# Component 2: matching the labels (but it's unsupervised)
			n2_z_sim = model(task_ids, sample2["sim_emb"].to(device))
			n2_z_dif = model(task_ids, sample2["dif_emb"].to(device))
			reg_loss_sim = h_get_sim_loss(n2_z_sim, sample2["centroid"].to(device))
			reg_loss_dif = get_sim_not_loss(sample2["centroid"].to(device), n2_z_dif)
			reg_loss2 = buffer.beta * ((reg_loss_sim)**2 + (reg_loss_dif-1)**2)
			# DER++
			reg =  reg_loss1 + reg_loss2

	# compute loss
	loss = (loss_sim)**2 + (loss_dif-1)**2
	if reg: loss = loss + reg

	log = {
		"train/loss": loss.detach().item(),
		"train/loss_sim": loss_sim.detach().item(),
		"train/loss_dif": loss_dif.detach().item(),
		"epoch": epoch,
		"centroid": torch.mean(centroid_sim)
	}
	if reg is not None: log["train/regularizer"] = reg

	progressbar.set_description(f"epoch: {epoch}; loss: {loss.item():.4f}; task_id: {task_id}; lesson: {lesson}")
	if wandb_run: wandb_run.log(log)

	# Update reservoir
	if buffer is not None:
		with torch.no_grad():
			buffer.add_sample({
				"x_lesson": lesson,
				"sim_emb": sim_emb.detach().cpu(),
				"dif_emb": dif_emb.detach().cpu(),
				"z_sim": z_sim.detach().cpu(),
				"z_dif": z_dif.detach().cpu(),
				"centroid": centroid_sim.detach().cpu()
			})
		del sample1
		del sample2
		
	############ save model #########
	with torch.no_grad():
		memory[lesson] = {"centroid": centroid_sim, "task_id": task_id}
	# 	memory[lesson]["params"] = model.get_weights(lesson)
	return model, loss

def my_clip_evaluation(in_path, source, model, in_base, types, dic, vocab, memory, task_id):

	with torch.no_grad():
		# get vocab dictionary
		if source == 'train':
			dic = dic_test
		else:
			dic = dic_train

		# get dataset
		clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
		dt = MyDataset(in_path, source, in_base, types, dic, vocab,
					clip_preprocessor=clip_preprocess)
		data_loader = DataLoader(dt, batch_size=128, shuffle=True)

		model.eval()

		top3 = 0
		top3_color = 0
		top3_material = 0
		top3_shape = 0
		tot_num = 0

		for base_is, images in data_loader:
			# Prepare the inputs
			images = images.to(device)
			ans = []
			batch_size_i = len(base_is)

			# go through memory
			for label in vocab:

				# Skip unlearned lesson
				if label not in memory.keys():
					ans.append(torch.full((batch_size_i, 1), 1000.0).squeeze(1))
					continue

				with torch.no_grad():
					imgs_emb = clip_model.encode_image(images).float()

				# compute stats
				task_ids = torch.LongTensor([memory[label]["task_id"]] * images.shape[0]).to(device)
				z = model(task_ids, imgs_emb)
				z = z.squeeze(0)
				centroid_i = memory[label]["centroid"]
				centroid_i = centroid_i.repeat(batch_size_i, 1)
				disi = ((z - centroid_i)**2).mean(dim=1)
				ans.append(disi.detach().to('cpu'))

			# get top3 incicies
			ans = torch.stack(ans, dim=1)
			values, indices = ans.topk(3, largest=False)
			_, indices_lb = base_is.topk(3)
			indices_lb, _ = torch.sort(indices_lb)

			# calculate stats
			tot_num += len(indices)
			for bi in range(len(indices)):
				ci = 0
				mi = 0
				si = 0
				if indices_lb[bi][0] in indices[bi]:
					ci = 1
				if indices_lb[bi][1] in indices[bi]:
					mi = 1
				if indices_lb[bi][2] in indices[bi]:
					si = 1

				top3_color += ci
				top3_material += mi
				top3_shape += si
				if (ci == 1) and (mi == 1) and (si == 1):
					top3 += 1

		scores = {
			"test/top3_color": top3_color/tot_num,
			"test/top3_material": top3_material/tot_num,
			"test/top3_shape": top3_shape/tot_num,
			"test/top3": top3/tot_num,
			"learned_concepts": len(memory.keys())
		}
		if wandb_run: wandb_run.log(scores)
		print(scores)

	return top3/tot_num

def my_clip_train(in_path, out_path, model_name, source, in_base,
				types, dic, vocab, pre_trained_model=None, n_skills=None, buffer_size=200):
	# get data
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	dt = MyDataset(in_path, source, in_base, types, dic, vocab,
					clip_preprocessor=clip_preprocess)

	# load encoder models from memory
	model = CLIP_AE_Encode(hidden_dim=hidden_dim_clip, latent_dim=latent_dim).to(device)
	n_concepts = len(colors) + len(shapes) + len(materials)
	model = SkilledMixin(
		model=model,
		n_tasks=n_concepts,
		n_skills=n_skills, # domains: colors, materials, shapes
		skilled_variant="learned",
		freeze=False
	).to(device)

	# inductive bias as in the paper
	lr_skills = lr
	lr_allocation = lr_skills * 1e2
	print(f"[-] LR skills: {lr_skills}; LR task allocation: {lr_allocation}")
	optimizer = optim.Adam(
		[
			{"params": model.model.fc1.weight},
			{"params": model.model.fc1.bias},
			{"params": model.model.fc1.skills_weight_A},
			{"params": model.model.fc1.skills_weight_B},
			{"params": model.model.fc1.skill_logits, "lr": lr_allocation},
			{"params": model.model.fc2.weight},
			{"params": model.model.fc2.bias},
			{"params": model.model.fc2.skills_weight_A},
			{"params": model.model.fc2.skills_weight_B},
			{"params": model.model.fc2.skill_logits, "lr": lr_allocation},
		],
		lr=lr_skills,
	)

	print(f"[-] # params: {count_parameters(model)}")

	# Define a buffer
	alpha, beta = 0.5, 0.5
	if buffer_size is not None:
		buffer = Buffer(alpha=alpha, beta=beta, size=buffer_size, warmup=iters_per_concept)
	else: buffer = None

	best_nt = 0
	t_tot = 0
	memory = {}

	all_concepts = [(vi, tl) for tl in types_learning for vi in dic[tl]] # all lessons
	all_concepts = list(zip(list(range(len(all_concepts))), all_concepts))

	accumulation_steps = len(all_concepts) // 4
	for epoch in range(epochs):
		random.shuffle(all_concepts) # shuffle and pass all concepts
		progressbar = tqdm(all_concepts)
		epoch_loss = None
		for concept_idx, concept in progressbar:
			vi, tl = concept
			model, loss = my_concept_fwdpass(dt, model, tl, vi, memory, concept_idx, buffer, progressbar, epoch)
			if not epoch_loss: epoch_loss = loss
			else: epoch_loss += loss
		# aggregated concepts loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss = None

		# Evaluate
		if epoch % 5 == 0:
			my_clip_evaluation(in_path, 'novel_test/', model,
							bsn_novel_test_1, ['rgba'], dic_train, vocab, memory, epoch)
			torch.save(model.state_dict(), "polytropon.pth")

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(
		prog="Complex Comparative Learning (Polytropon)",
		description="Alternative to hypernetworks: Polytropon, combining modular skills in multitask learning. Paper: https://arxiv.org/abs/2202.13914"
	)
	argparser.add_argument('--in_path', '-i',
				help='Data input path', required=True)
	argparser.add_argument('--out_path', '-o',
				help='Model memory output path', required=True)
	argparser.add_argument('--model_name', '-n', default='best_mem.pickle',
				help='Best model memory to be saved file name', required=False)
	argparser.add_argument('--pre_train', '-p', default=None,
				help='Pretrained model import name (saved in outpath)', required=False)
	argparser.add_argument('--skills', '-s', default=3, type=int,
				help='Number of skills', required=True)
	argparser.add_argument('--wandb', '-w', default=None,
				help='Enable wandb logging', required=False)
	argparser.add_argument('--buffer_size', '-bf', default=None, type=int,
				help='Enable DER++ regularization with reservoir sampling replay buffer of given size')
	argparser.add_argument('--lr', '-l', default=1e-4, type=float,
				help='Learning rate')
	args = argparser.parse_args()

	lr = args.lr

	if args.wandb is not None:
		wandb.login()
		config = {
			"lr": lr,
			"sim_batch": sim_batch,
			"gen_batch": gen_batch,
			"epochs": epochs,
			"batch_size": batch_size,
			"latent_dim": latent_dim,
			"n_tasks": len(colors) + len(shapes) + len(materials),
			"n_skills": args.skills,
			"buffer_size": args.buffer_size
		}
		wandb_run = wandb.init(name="polytropon", project="hypernet-concept-learning", config=config)

	my_clip_train(args.in_path, args.out_path, args.model_name,
				'novel_train/', bn_n_train, ['rgba'], dic_train, vocabs, args.pre_train, args.skills, args.buffer_size)
