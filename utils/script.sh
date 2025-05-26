tensorboard --port=8010 --samples_per_plugin=images=100000 --logdir=./

nohup python train.py > output0.log 2>&1 &

kill -9 1511652
