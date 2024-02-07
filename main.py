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

def my_train_clip_encoder(dt, model, attr, lesson, memory):

	n_concepts = len(colors) + len(shapes) + len(materials)
	task_ids = torch.LongTensor(list(range(0, n_concepts))).to(device)

	# get model
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	
	optimizer = optim.Adam(model.parameters(), lr=lr)
	model.train()

	loss_sim = None
	loss_dif = None
	loss = 10
	ct = 0
	
	centroid_sim = torch.rand(1, latent_dim).to(device)

	while loss > 0.008:
		ct += 1
		if ct > 5: break
		progressbar = tqdm(range(200))
		for i in progressbar:
			# Get Inputs: sim_batch, (sim_batch, 4, 128, 128)
			base_name_sim, images_sim = dt.get_better_similar(attr, lesson)
			images_sim = images_sim.to(device)

			# run similar model
			z_sim = model(task_ids, clip_model, images_sim)
			centroid_sim = centroid_sim.detach()
			centroid_sim, loss_sim = get_sim_loss(torch.vstack((z_sim, centroid_sim)))

			# Run Difference
			base_name_dif, images_dif = dt.get_better_similar_not(attr, lesson)
			images_dif = images_dif.to(device)
			
			# run difference model
			z_dif = model(task_ids, clip_model, images_dif)
			loss_dif = get_sim_not_loss(centroid_sim, z_dif)

			# compute loss
			loss = (loss_sim)**2 + (loss_dif-1)**2

			log = {
				"train/loss": loss.detach().item(),
				"train/loss_sim": loss_sim.detach().item(),
				"train/loss_dif": loss_dif.detach().item()
			}

			progressbar.set_description(f"loss: {loss.item():.4f}")
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if wandb_run: wandb_run.log(log)

		print('[', ct, ']', loss.detach().item(), loss_sim.detach().item(),
				loss_dif.detach().item())
		
	############ save model #########
	with torch.no_grad():
		memory[lesson] = {"centroid": centroid_sim}
	# 	memory[lesson]["params"] = model.get_weights(lesson)
	return model

def my_clip_evaluation(in_path, source, model, in_base, types, dic, vocab, memory):
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

				# compute stats
				z, _ = model(clip_model, images)
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

		if wandb_run:
			wandb_run.log({
				"test/top3_color": top3_color/tot_num,
				"test/top3_material": top3_material/tot_num,
				"test/top3_shape": top3_shape/tot_num,
				"test/top3": top3/tot_num,
				"learned_concepts": len(memory.keys())
			})
		print(tot_num, top3_color/tot_num, top3_material/tot_num,
				top3_shape/tot_num, top3/tot_num)
	return top3/tot_num


def my_clip_train(in_path, out_path, model_name, source, in_base,
				types, dic, vocab, pre_trained_model=None, n_skills=None):
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
		freeze=False
	).to(device)

	print(f"[-] # params: {count_parameters(model)}")

	best_nt = 0
	t_tot = 0
	memory = {}
	for i in range(epochs):
		for tl in types_learning:  # attr
			random.shuffle(dic[tl])
			for vi in dic[tl]:  # lesson
				# Train
				print("#################### Learning: " + str(i) + " ----- " + str(vi))
				t_start = time.time()
				model = my_train_clip_encoder(dt, model, tl, vi, memory)
				t_end = time.time()
				t_dur = t_end - t_start
				t_tot += t_dur
				print("Time: ", t_dur, t_tot)

				# Evaluate
				top_nt = my_clip_evaluation(in_path, 'novel_test/', model,
								bsn_novel_test_1, ['rgba'], dic_train, vocab, memory)
				if top_nt > best_nt:
					best_nt = top_nt
					print("++++++++++++++ BEST NT: " + str(best_nt))
					# with open(os.path.join(out_path, model_name), 'wb') as handle:
					#	pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
	argparser.add_argument('--skills', '-s', default=3, type=int,
				help='Number of skills', required=True)
	argparser.add_argument('--wandb', '-w', default=None,
				help='Enable wandb logging', required=False)
	args = argparser.parse_args()

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
		}
		wandb_run = wandb.init(name="polytropon", project="hypernet-concept-learning", config=config)

	my_clip_train(args.in_path, args.out_path, args.model_name,
				'novel_train/', bn_n_train, ['rgba'], dic_train, vocabs, args.pre_train, args.skills)

	
