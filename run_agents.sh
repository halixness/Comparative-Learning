# GPU 0 runs
for i in {1..5}
do
  tmux new -d "CUDA_VISIBLE_DEVICES=0 ./agent.sh $1"
done
# GPU 1 runs
for i in {1..4}
do
  tmux new -d "CUDA_VISIBLE_DEVICES=1 ./agent.sh $1"
done
