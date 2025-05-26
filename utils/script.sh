tensorboard --port=8010 --samples_per_plugin=images=100000 --logdir=./logs/dev

nohup python train_SC_TransNet.py > output0.log 2>&1 &

kill -9 1511652
