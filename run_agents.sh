#!/bin/bash
agents=$2
agents1=$((agents / 2))
agents2=$((agents - agents1))

# GPU 0 runs
for ((i=1; i<=agents1; i++))
do
  tmux new -d "CUDA_VISIBLE_DEVICES=0 ./agent.sh $1"
done

# GPU 1 runs
for ((i=1; i<=agents2; i++))
do
  tmux new -d "CUDA_VISIBLE_DEVICES=1 ./agent.sh $1"
done