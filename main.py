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
from models.hypernetwork import HyperMem, count_parameters

import torch.multiprocessing as mp
import torch.utils.data as data
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

PORT=19777

class TorchDataset(data.Dataset):

	def __init__(self, in_path:str):        
		# samples:List[object]    {predicate, subject, fact, belief}
		self.samples = self.get_training_data(in_path=in_path)

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx:int) -> dict:
		return self.samples[idx]

	def get_training_data(self, in_path:str):
		# path = os.path.join(in_path, 'train_new_objects_dataset.json')
		# path = os.path.join(in_path, "final_splits.json") 
		path = os.path.join(in_path, "train_new_objects_200_dataset.json")
		with open(path, 'r') as file:
			# Load JSON data from the file
			training_data = json.load(file)
		return training_data

def ddp_setup(rank, world_size:int, port:int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def get_batches(base_names, in_path, source):
	images = []
	for base_name in base_names:
		path = os.path.join(in_path, source, f"{base_name}_rgba.pickle")
		with open(path, 'rb') as file:
			emb = pickle.load(file)
			images.append(emb)
	images = torch.stack(images, dim = 0)
	return images

def my_train_clip_encoder(resume_iter, rank, training_data, n_split, memory, in_path, out_path, source, model_name, model):
	
	if rank == 0:
		# Logging
		wandb.login()
		config = {
			"sim_batch": sim_batch,
			"gen_batch": gen_batch,
			"epochs": epochs,
			"batch_size": batch_size,
			"latent_dim": latent_dim,
		}
		wandb_run = wandb.init(name="hypernet-logic-der++", project="hypernet-concept-learning", config=config)
	
	# Model
	optimizer = optim.Adam(model.parameters(), lr=lr)
	model.train()
	centroid_sim = torch.rand(1, latent_dim).to(rank)
	
	# Replay buffer
	buffer = Buffer(alpha=0.5, beta=0.5, size=5000)

	loss_sim = None
	loss_dif = None
	loss = 10

	t_tot = 0
	count = 0
	lesson = None
	previous_lesson = 'first_lesson'
	i = 0
	for batch in tqdm(training_data):
		if resume_iter and i < resume_iter: 
			i += 1
			continue # skipping steps

		# Get Lesson
		lesson = batch['lesson'][0]
		base_names_sim = [b[0] for b in batch['base_names_sim']]
		base_names_dif = [b[0] for b in batch['base_names_dif']]
		
		optimizer.zero_grad()

		# Get Inputs: sim_batch, (sim_batch, 4, 132, 132)
		images_sim = get_batches(base_names_sim, in_path, source)
		images_sim = images_sim.to(rank)

		# run similar model
		z_sim, centroid_sim = model(lesson, images_sim)
		centroid_sim = centroid_sim.squeeze(0)
		loss_sim = h_get_sim_loss(z_sim, centroid_sim)

		# Run Difference
		images_dif = get_batches(base_names_dif, in_path, source)
		images_dif = images_dif.to(rank)

		# run difference model
		z_dif, _ = model(lesson, images_dif)
		loss_dif = get_sim_not_loss(centroid_sim, z_dif)

		# compute loss
		loss = (loss_sim)**2 + (loss_dif-1)**2

		# Dark Experience Replay (++)
		sample1 = buffer.get_sample()
		sample2 = buffer.get_sample()
		reg = None
		if sample1 and sample2:
			# Component 1: matching the logits
			n1_z_sim, _ = model(sample1["x_lesson"], sample1["x_sim_emb"])
			n1_z_dif, _ = model(sample1["x_lesson"], sample1["x_dif_emb"])
			reg_loss1 = buffer.alpha * (F.mse_loss(n1_z_sim, sample1["z_sim"]) + F.mse_loss(n1_z_dif, sample1["z_dif"]))
			# Component 2: matching the labels (but it's unsupervised)
			n2_z_sim, n2_centroid = model(sample1["x_lesson"], sample1["x_sim_emb"])
			n2_z_dif, _ = model(sample1["x_lesson"], sample1["x_dif_emb"])
			n2_centroid = n2_centroid.squeeze(0)
			reg_loss_sim = h_get_sim_loss(n2_z_sim, n2_centroid)
			reg_loss_dif = get_sim_not_loss(n2_centroid, n2_z_dif)
			reg_loss2 = buffer.beta * ((reg_loss_sim)**2 + (reg_loss_dif-1)**2)
			# DER++
			reg =  reg_loss1 + reg_loss2
			loss = loss + reg

		# Backprop
		loss.backward()
		optimizer.step()

		# Log
		log = {
			"train/loss": loss.detach().item(),
			"train/loss_sim": loss_sim.detach().item(),
			"train/loss_dif": loss_dif.detach().item(),
			"centroid": torch.mean(centroid_sim).detach().item()
		}
		if reg: log["train/regularizer"] = reg.item()

		# Update reservoir
		with torch.no_grad():
			buffer.add_sample({
				"x_lesson": lesson,
				"x_sim_emb": images_sim.detach(),
				"x_dif_emb": images_dif.detach(),
				"z_sim": z_sim.detach(),
				"z_dif": z_dif.detach(),
			})

		if rank == 0: wandb_run.log(log)

		# Batches for the same lesson are presented in sequence
		# So for each lesson switch -> save model
		if (rank == 0) and (lesson != previous_lesson):
			with torch.no_grad():
				try:
					memory[lesson] = True
					torch.save(model.state_dict(), os.path.join(out_path,"checkpoints", f"hypernet_learned{len(memory.keys())}_{int(round(datetime.now().timestamp()))}.pth"))
				except:
					print(f"[-] Error saving model for lesson: {lesson}")
		previous_lesson = lesson
		i += 1
		
		torch.distributed.barrier()
	return memory

def my_clip_train(rank, checkpoint, resume_iter, world_size, in_path, out_path, n_split, model_name, source):  

	ddp_setup(rank, world_size=world_size, port=PORT)

	# Load encoder models from memory
	clip_model, _ = clip.load("ViT-B/32", device=rank)
	model = HyperMem(lm_dim=512, knob_dim=128, input_dim=512, hidden_dim=128, output_dim=latent_dim, clip_model=clip_model).to(rank)
	model = DDP(model, device_ids=[rank])

	# Loading model if requested
	if checkpoint:
		print(f"[+] Loading checkpoint: {checkpoint}")
		loaded_state_dict = torch.load(checkpoint)
		model.load_state_dict(loaded_state_dict)
	
	# Parallel
	print(f"[-] # params: {count_parameters(model)}")

	# Training data
	training_data = TorchDataset(in_path=in_path)
	size = None
	# Aligning
	for s in training_data.samples:
		if not size: size = len(s["base_names_sim"])
		if size != len(s["base_names_sim"]): s["base_names_sim"] = s["base_names_sim"][:size] 
		if size != len(s["base_names_dif"]): s["base_names_dif"] = s["base_names_sim"][:size]
	# Check
	for s in training_data.samples:
		if not size: size = len(s["base_names_sim"])
		assert size == len(s["base_names_sim"]) 
		assert size == len(s["base_names_dif"]) 

	sola_dataloader = DataLoader(training_data, batch_size=1, sampler=DistributedSampler(training_data), shuffle=False)

	# Run training
	memory = {}
	memory = my_train_clip_encoder(resume_iter, rank, sola_dataloader, n_split, memory, in_path, out_path, source, model_name, model)
	destroy_process_group()

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--in_path', '-i',
				help='Data input path', required=True)
	
	argparser.add_argument('--out_path', '-o',
				help='Model memory output path', required=True)

	argparser.add_argument('--n_split', '-s', default=0,
				help='Split number', required=None)
	
	argparser.add_argument('--model_name', '-n', default='first_try_model',
				help='Best model memory to be saved file name', required=False)
	
	argparser.add_argument('--gpu_idx', '-g', default=0,
				help='Select gpu index', required=False)

	argparser.add_argument('--checkpoint', '-w', default=None, help='Resume from checkpoint', type=str, required=False)
	argparser.add_argument('--resume_iter', '-ri', default=None, help='Resume from given iteration', type=int, required=False)

	args = argparser.parse_args()
	checkpoint = args.checkpoint
	resume_iter = args.resume_iter

	# Running in parallel
	n_split = args.n_split		
	ngpus = torch.cuda.device_count()
	mp.spawn(my_clip_train, args=(checkpoint, resume_iter, ngpus, args.in_path, args.out_path, n_split, args.model_name, 'train/'), nprocs=ngpus)
