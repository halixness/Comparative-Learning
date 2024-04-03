import os
import json
import torch
import time
import pickle
import argparse
import torch.optim as optim
from datetime import datetime
import wandb
from tqdm import tqdm
from typing import List
from dataset import MyDataset

from config import *
from dataset import *
from my_models import *
from util import *
from polytropon import SkilledMixin
from models import CLIP_AE_Encode

import torch.multiprocessing as mp
import torch.utils.data as data
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.nn.functional as F

PORT = 19777
TASK_IDS = {
	"and": 0,
	"or": 1,
	"not": 2
}
# TASK_IDS["primitive"] = 3

class TorchDataset(data.Dataset):

	def __init__(self, in_path:str):        
		# samples:List[object]    {predicate, subject, fact, belief}
		self.samples = self.get_training_data(in_path=in_path)
		self.uniform_batches(shuffle=True)
		self.allocate_tasks()

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx:int) -> dict:
		return self.samples[idx]

	def get_training_data(self, in_path:str):
		""" Load lessons batches from json file """
		# path = os.path.join(in_path, 'train_new_objects_dataset.json')
		# path = os.path.join(in_path, "final_splits.json") -
		print("[-] Loading dataset...")
		path = os.path.join(in_path, "train_new_objects_200_dataset.json")
		with open(path, 'r') as file:
			# Load JSON data from the file
			training_data = json.load(file)
		return training_data

	def uniform_batches(self, shuffle:bool=False):
		""" Checks # examples per batch, uniforms it """
		size = None
		# Aligning
		for s in self.samples:
			if not size: size = len(s["base_names_sim"])
			if size != len(s["base_names_sim"]): s["base_names_sim"] = s["base_names_sim"][:size] 
			if size != len(s["base_names_dif"]): s["base_names_dif"] = s["base_names_sim"][:size]
		# Check
		for s in self.samples:
			if not size: size = len(s["base_names_sim"])
			assert size == len(s["base_names_sim"]) 
			assert size == len(s["base_names_dif"]) 
		if shuffle: random.shuffle(self.samples)

	def allocate_tasks(self):
		""" Sets task IDs for modular skill sharing """
		filtered = []
		for s in self.samples: # parse lesson name and label by logical concept or base concept
			for operator in TASK_IDS.keys():
				if operator in s["lesson"].lower(): 
					s["task_id"] = TASK_IDS[operator]
					filtered.append(s)
					break
		del self.samples
		self.samples = filtered

def get_batches(base_names, in_path, source):
	images = []
	for base_name in base_names:
		path = os.path.join(in_path, source, f"{base_name}_rgba.pickle")
		with open(path, 'rb') as file:
			emb = pickle.load(file)
			images.append(emb)
	images = torch.stack(images, dim = 0)
	return images

def my_concept_fwdpass(sample, clip_model, model, memory, progressbar, epoch, in_path, source, device, config):

	task_ids = torch.LongTensor([sample["task_id"]] * sim_batch).to(device)
	model.train()

	# Get Inputs: sim_batch, (sim_batch, 4, 128, 128)
	lesson = sample['lesson']
	base_names_sim = sample['base_names_sim']
	base_names_dif = sample['base_names_dif']

	# Initialize centroid
	if lesson not in memory.keys():
		centroid_sim = torch.rand(1, latent_dim).unsqueeze(0).to(device)
	else: centroid_sim = memory[lesson]["centroid"].to(device)

	# Lesson embedding
	with torch.no_grad():
		tok_lesson = clip.tokenize([lesson]).to(device)
		txt_lesson = clip_model.encode_text(tok_lesson).float()
	
	# Get Inputs: sim_batch, (sim_batch, 4, 132, 132)
	images_sim = get_batches(base_names_sim, in_path, source)
	images_sim = images_sim.to(device)
	
	# run similar model
	z_sim, z_lesson = model(task_ids, images_sim, txt_lesson.repeat(images_sim.shape[0], 1))
	centroid_sim = centroid_sim.detach() # 1, 16
	centroid_sim, loss_sim = get_sim_loss(torch.vstack((z_sim, centroid_sim)))

	# Run Difference
	images_dif = get_batches(base_names_dif, in_path, source)
	images_dif = images_dif.to(device)

	# run difference model
	z_dif, _ = model(task_ids, images_dif, txt_lesson.repeat(images_sim.shape[0], 1))
	loss_dif = get_sim_not_loss(centroid_sim, z_dif)

	# compute loss
	loss = (loss_sim)**2 + (loss_dif-1)**2
	textual_loss = -F.cosine_similarity(z_lesson[0].unsqueeze(0), centroid_sim).mean()
	loss += textual_loss

	log = {
		"train/loss": loss.detach().item(),
		"train/loss_sim": loss_sim.detach().item(),
		"train/loss_dif": loss_dif.detach().item(),
		"train/txt_loss": textual_loss.item(),
		"epoch": epoch,
		"centroid": torch.mean(centroid_sim.detach())
	}

	wandb.log(log)
	progressbar.set_description(f"epoch: {epoch}; loss: {loss.detach().item():.4f}; task_id: {sample['task_id']}; lesson: {lesson}")

	############ save model #########
	with torch.no_grad():
		memory[lesson] = {"centroid": centroid_sim.detach()}
	return model, loss

def my_clip_train(rank, checkpoint, resume_iter, world_size, in_path, out_path, model_name, source, config):  
	
	clip_model, _ = clip.load("ViT-B/32", device=device)

	# Training data
	training_data = TorchDataset(in_path=in_path)

	# Training the model
	model = CLIP_AE_Encode(hidden_dim=hidden_dim_clip, latent_dim=latent_dim).to(rank)
	# n_concepts = len(colors) + len(shapes) + len(materials)
	# n_concepts = 6 # color, material, shape, and, or, not
	n_concepts = config["n_concepts"]
	n_tasks = len(TASK_IDS.keys()) # and, or, not
	model = SkilledMixin(
		model=model,
		n_tasks=n_tasks,
		n_skills=n_concepts, # domains: colors, materials, shapes
		skilled_variant="learned",
		freeze=False
	).to(rank)

	if checkpoint:
		print(f"[+] Loading from checkpoint: {checkpoint}")
		model.load_state_dict(torch.load(checkpoint))

	# inductive bias as in the paper
	lr_skills = config["lr"]
	lr_allocation = lr_skills * 1e2 # 1e-2
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

	# Start training run
	best_nt = 0
	t_tot = 0
	memory = {}
	if resume_iter: print(f"[+] Resuming from iter: {resume_iter}")

	accumulation_steps = config["accumulation_steps"]
	epochs = config["epochs"]
	for epoch in range(epochs):
		if resume_iter and epoch < resume_iter: continue # skipping if requested
		progressbar = tqdm(training_data)
		epoch_loss = None
		idx = 0
		for concept in progressbar:
			model, loss = my_concept_fwdpass(concept, clip_model, model, memory, progressbar, epoch, in_path, source, rank, config)
			if not epoch_loss: epoch_loss = loss.item()
			else: epoch_loss += loss.item()
			if idx % accumulation_steps == 0:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			idx += 1

		# Save model snapshot
		torch.save(model.state_dict(), f"polytropon_e{epoch}_{time.strftime('%Y%m%d-%H%M%S')}.pth")

# Main
argparser = argparse.ArgumentParser()
argparser.add_argument('--in_path', '-i',
			help='Data input path', default="/mnt/cimec-storage6/shared/filippo_diego_datasets/my_datasets")

argparser.add_argument('--out_path', '-o',
			help='Model memory output path', default="out")

argparser.add_argument('--run_name', '-r',
			help='Model memory output path', default="polytropon", type=str)

argparser.add_argument('--model_name', '-n', default='complex_model',
			help='Best model memory to be saved file name')


argparser.add_argument('--accumulation_steps', type=int)
argparser.add_argument('--lr', type=float)
argparser.add_argument('--epochs', type=int)
argparser.add_argument('--n_concepts', type=int)

argparser.add_argument('--checkpoint', '-w', default=None, help='Resume from checkpoint', type=str, required=False)
argparser.add_argument('--resume_iter', '-ri', default=None, help='Resume from given iteration', type=int, required=False)

args = argparser.parse_args()
checkpoint = args.checkpoint
resume_iter = args.resume_iter
config = {
	"lr": args.lr,
	"epochs": args.epochs,
	"n_concepts": args.n_concepts,
	"accumulation_steps": args.accumulation_steps,
}
# Running
wandb.init(project="complex-concept-learning", name="agent")
my_clip_train(0, checkpoint, resume_iter, 1, args.in_path, args.out_path, args.model_name, 'train/', config)
