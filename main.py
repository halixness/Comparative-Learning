import os
import torch
import clip
import time
import pickle
import random
import numpy as np
import argparse
import torch.nn as nn
from typing import List
import torch.utils.data as data
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from config import *
from dataset import *
from models.novel import *

torch.autograd.set_detect_anomaly(True)
random.seed(1337)
torch.manual_seed(1337)

wandb_run = None

def ddp_setup(rank, world_size:int, port:int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

# =================================================

# https://en.wikipedia.org/wiki/Reservoir_sampling#:~:text=Reservoir%20sampling%20is%20a%20family,to%20fit%20into%20main%20memory.
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

class TorchDataset(data.Dataset):

    def __init__(self, samples:List[dict]):        
        """
            samples:List[object]    {predicate, subject, fact, belief}
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx:int) -> dict:
        return self.samples[idx]

def get_torch_dataset(dt, types_learning, dic, samples_per_lesson=200):
	samples = []
	tot_iters = sum([len(dic[tl]) for tl in types_learning]) * samples_per_lesson
	progressbar = tqdm(range(tot_iters))
	for tl in types_learning:  # attr
		random.shuffle(dic[tl])
		for vi in dic[tl]:  # lesson
			for itx in range(samples_per_lesson):
				samples.append({
					"attr": tl,
					"lesson": vi,
				})
				progressbar.update(1)
	return TorchDataset(samples)

# =================================================

def get_buffer_distribution(buffer):
	if len(buffer.data) > 0:
		notions = {}
		for sample in buffer.data:
			notions.setdefault(sample["x_lesson"], 0)
			notions[sample["x_lesson"]] += 1 / len(buffer.data)
		return notions
	else: return None

def my_train_clip_encoder(dt, model, attr, lesson, memory, epoch, buffer, rank, ngpus, ntasks, wandb_run):

	is_parallel = ntasks > 1
	if ngpus == 1: rank = 0

	# get model
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=rank)
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	model.train()

	loss_sim = None
	loss_dif = None
	loss = 10
	ct = 0

	while loss > 0.008:
		ct += 1
		if ct >= 4:
			break
		progressbar = tqdm(range(200 // ngpus))
		for i in progressbar:

			optimizer.zero_grad()
			
			# Get Inputs: sim_batch, (sim_batch, 4, 128, 128)
			base_name_sim, images_sim = dt.get_better_similar(attr, lesson)
			images_sim = images_sim.to(rank)
			with torch.no_grad():
				sim_emb = clip_model.encode_image(images_sim).float().detach() # B, 512
				
			# run similar model
			z_sim, centroid_sim = model(lesson, sim_emb)
			centroid_sim = centroid_sim.squeeze(0)
			loss_sim = h_get_sim_loss(z_sim, centroid_sim)

			# Run Difference
			base_name_dif, images_dif = dt.get_better_similar_not(attr, lesson)
			images_dif = images_dif.to(rank)
			with torch.no_grad():
				dif_emb = clip_model.encode_image(images_dif).float().detach() # B, 512
				
			# run difference model
			z_dif, _ = model(lesson, dif_emb)
			loss_dif = get_sim_not_loss(centroid_sim, z_dif)

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

			# Loss
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

			# Backprop
			loss.backward()
			optimizer.step()
			# Log
			progressbar.set_description(f"loss: {loss.item():.2f}")
			if wandb_run: wandb_run.log(log)
			else: print(log)
			
			# Update reservoir
			with torch.no_grad():
				buffer.add_sample({
					"x_lesson": lesson,
					"x_sim_emb": sim_emb.detach(),
					"x_dif_emb": dif_emb.detach(),
					"z_sim": z_sim.detach(),
					"z_dif": z_dif.detach(),
				})

		print('[', ct, ']', loss.detach().item(), loss_sim.detach().item(),
				loss_dif.detach().item())
		print(f"buffer: {get_buffer_distribution(buffer)}")

	if is_parallel: torch.distributed.barrier()

	############ save model #########
	with torch.no_grad():
		memory[lesson] = {} # (L)
	return model, memory

def my_clip_evaluation(in_path, source, model, in_base, types, dic, vocab, memory, epoch, rank):
	with torch.no_grad():
		# get vocab dictionary
		if source == 'train':
			dic = dic_test
		else:
			dic = dic_train

		# get dataset
		clip_model, clip_preprocess = clip.load("ViT-B/32", device=rank)
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
			images = images.to(rank)
			ans = []
			batch_size_i = len(base_is)

			# go through memory
			for label in vocab:

				# Skip unlearned lesson
				if label not in memory.keys():
					ans.append(torch.full((batch_size_i, 1), 1000.0).squeeze(1))
					continue

				with torch.no_grad():
					emb = clip_model.encode_image(images).float() # B, 512

				# compute stats
				z, centroid_i = model(label, emb)
				z = z.squeeze(0)
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

		if wandb_run:
			wandb_run.log({
				"test/top3_color": top3_color/tot_num,
				"test/top3_material": top3_material/tot_num,
				"test/top3_shape": top3_shape/tot_num,
				"test/top3": top3/tot_num,
				"learned_concepts": len(memory.keys()),
				"epoch": epoch
			})
		print(tot_num, top3_color/tot_num, top3_material/tot_num,
				top3_shape/tot_num, top3/tot_num)
	return top3/tot_num

def my_clip_train(rank, in_path, out_path, model_name, source, in_base,
				types, dic, vocab, pre_trained_model, hyperparams, ngpus, ntasks, port, wandb_run):

	is_parallel = ntasks > 1
	if ngpus == 1: rank = 0 # fix to one gpu if multi processes on one
	if is_parallel: ddp_setup(rank, world_size=ntasks, port=port)

	# Get data
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=rank)
	dt = MyDataset(in_path, source, in_base, types, dic, vocab,
					clip_preprocessor=clip_preprocess)

	# Load encoder models from memory
	model = HyperMem(lm_dim=768, knob_dim=128, input_dim=512, hidden_dim=128, output_dim=latent_dim).to(rank)
	print(f"[-] #params: {count_parameters(model)}")
	
	if is_parallel: model = DDP(model, device_ids=[rank])
	
	# Define a buffer
	alpha, beta, buffer_size = hyperparams
	buffer = Buffer(alpha=alpha, beta=beta, size=buffer_size)

	best_nt = 0
	t_tot = 0
	memory = {}
	
	for i in range(epochs):
		for tl in types_learning:
			random.shuffle(dic[tl])
			for vi in dic[tl]:
				print("#################### Learning: " + str(i) + " ----- " + str(vi))
				
				# Training
				t_start = time.time()
				model, memory = my_train_clip_encoder(dt, model, tl, vi, memory, i, buffer, rank, ngpus, ntasks, wandb_run)
				t_end = time.time()
				t_dur = t_end - t_start
				t_tot += t_dur
				print("Time: ", t_dur, t_tot)

				# Evaluate
				if (is_parallel == False) or (is_parallel and rank == 0):
					top_nt = my_clip_evaluation(in_path, 'novel_test/', model, bsn_novel_test_1, ['rgba'], dic_train, vocab, memory, i, rank)
					if top_nt > best_nt:
						best_nt = top_nt
						print("++++++++++++++ BEST NT: " + str(best_nt))
				if is_parallel: torch.distributed.barrier()
	
	if is_parallel: destroy_process_group()

if __name__ == "__main__":

	argparser = argparse.ArgumentParser()
	argparser.add_argument('--in_path', '-i',
				help='Data input path', required=True)
	argparser.add_argument('--out_path', '-o',
				help='Model memory output path', required=True)
	argparser.add_argument('--model_name', '-n', default='best_mem.pickle',
				help='Best model memory to be saved file name', required=False)
	argparser.add_argument('--pre_train', '-p', default=None,
				help='Pretrained model import name (saved in outpath)', required=False)
	argparser.add_argument('--wandb', '-w', default=False, type=bool,
				help='Enable wandb')
	argparser.add_argument('--run_name', '-r', default="hypernet", type=str,
				help='Wandb run name')
	argparser.add_argument('--device', '-d', default="cuda", type=str,
				help='Device')
	argparser.add_argument('--lesson_iterations', '-ct', default=4, type=int,
				help='Iterations per concept')
	argparser.add_argument('--buffer_size', '-bf', default=200, type=int,
				help='Replay buffer size')
	argparser.add_argument('--alpha', '-a', default=0.5, type=int,
				help='Regularization hyperparam alpha')
	argparser.add_argument('--beta', '-b', default=0.5, type=int,
				help='Regularization hyperparam alpha')
	argparser.add_argument('--parallel', '-pp', default=False, type=bool,
				help='Enable multi-gpu computing')
	argparser.add_argument('--port', '-po', default=12355, type=int,
				help='Multiprocessing port network')
	argparser.add_argument('--tasks', '-ts', default=1, type=int,
				help='Number of parallel tasks')
	args = argparser.parse_args()

	if args.wandb:
		wandb.login()
		config = {
			"sim_batch": sim_batch,
			"gen_batch": gen_batch,
			"epochs": epochs,
			"batch_size": batch_size,
			"latent_dim": latent_dim,
			"alpha": args.alpha,
			"beta": args.beta,
			"buffer_size": args.buffer_size
		}
		wandb_run = wandb.init(name=args.run_name, project="hypernet-concept-learning", config=config)
		
	port = args.port

	if not args.parallel:
		ngpus = 1
		ntasks = 1
		my_clip_train(0, args.in_path, args.out_path, args.model_name, 'novel_train/', bn_n_train, ['rgba'], dic_train, vocabs, args.pre_train, (args.alpha, args.beta, args.buffer_size), ngpus, ntasks, port, wandb_run)
	else:
		ngpus = torch.cuda.device_count()
		ntasks = args.tasks
		mp.spawn(my_clip_train, args=(args.in_path, args.out_path, args.model_name, 'novel_train/', bn_n_train, ['rgba'], dic_train, vocabs, args.pre_train, (args.alpha, args.beta, args.buffer_size), ngpus, ntasks, port, wandb_run), nprocs=ntasks)

	
	
