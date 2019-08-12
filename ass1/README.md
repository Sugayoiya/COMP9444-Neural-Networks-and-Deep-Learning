# COMP9444-Neural-Networks-and-Deep-Learning

assignment requirement: https://www.cse.unsw.edu.au/~cs9444/18s2/hw1/index.html

`export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}`

`export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64\   
                        ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
`

`source ~/tensorflow/bin/activate`

`python3 train.py 5153042 conv`

```
2019-08-12 18:41:35.730317: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5035 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2019-08-12 18:41:35.800765: W tensorflow/core/framework/allocator.cc:108] Allocation of 245862400 exceeds 10% of system memory.
2019-08-12 18:41:35.896415: W tensorflow/core/framework/allocator.cc:108] Allocation of 245862400 exceeds 10% of system memory.
2019-08-12 18:41:35.998067: W tensorflow/core/framework/allocator.cc:108] Allocation of 245862400 exceeds 10% of system memory.
2019-08-12 18:41:36.097611: W tensorflow/core/framework/allocator.cc:108] Allocation of 245862400 exceeds 10% of system memory.
2019-08-12 18:41:36.209230: W tensorflow/core/framework/allocator.cc:108] Allocation of 245862400 exceeds 10% of system memory.
Starting Training...
Epoch 0, Training Loss: 0.018212206422342597, Test accuracy: 0.981270032051282,                 time: 12.1s, total time: 15.17s
Epoch 1, Training Loss: 0.004084383489490077, Test accuracy: 0.9825721153846154,                 time: 10.89s, total time: 29.05s
Epoch 2, Training Loss: 0.002315733622309151, Test accuracy: 0.9869791666666666,                 time: 10.89s, total time: 42.93s
Epoch 3, Training Loss: 0.0013078811863234928, Test accuracy: 0.9884815705128205,                 time: 10.89s, total time: 56.81s
Epoch 4, Training Loss: 0.0007702748125581404, Test accuracy: 0.9847756410256411,                 time: 10.88s, total time: 70.69s
Epoch 5, Training Loss: 0.0008010272732658408, Test accuracy: 0.9877804487179487,                 time: 10.89s, total time: 84.57s
Epoch 6, Training Loss: 0.0005487489053177053, Test accuracy: 0.9865785256410257,                 time: 10.89s, total time: 98.44s
Epoch 7, Training Loss: 0.0006534495447578304, Test accuracy: 0.9873798076923077,                 time: 10.89s, total time: 112.33s
Epoch 8, Training Loss: 0.00036110246194248163, Test accuracy: 0.9894831730769231,                 time: 10.86s, total time: 126.2s
Epoch 9, Training Loss: 0.00037176820177311663, Test accuracy: 0.9848758012820513,                 time: 10.88s, total time: 140.08s
Epoch 10, Training Loss: 0.0008195386672864133, Test accuracy: 0.9860777243589743,                 time: 10.88s, total time: 153.93s
Epoch 11, Training Loss: 0.0003915308924911825, Test accuracy: 0.9875801282051282,                 time: 10.88s, total time: 167.8s
Epoch 12, Training Loss: 0.0006406960241841083, Test accuracy: 0.9855769230769231,                 time: 10.88s, total time: 181.7s
Epoch 13, Training Loss: 0.00021456077441517858, Test accuracy: 0.9875801282051282,                 time: 10.9s, total time: 195.63s
Epoch 14, Training Loss: 7.300364056760866e-05, Test accuracy: 0.987479967948718,                 time: 10.88s, total time: 209.53s
Epoch 15, Training Loss: 5.543237583309062e-05, Test accuracy: 0.9866786858974359,                 time: 10.85s, total time: 223.37s
Epoch 16, Training Loss: 0.00040909229486875485, Test accuracy: 0.9830729166666666,                 time: 10.88s, total time: 237.26s
Epoch 17, Training Loss: 0.00038573470453471924, Test accuracy: 0.98828125,                 time: 10.86s, total time: 251.11s
Epoch 18, Training Loss: 0.0003483740963368452, Test accuracy: 0.9877804487179487,                 time: 10.86s, total time: 264.96s
Epoch 19, Training Loss: 0.0002969205463991529, Test accuracy: 0.9849759615384616,                 time: 10.85s, total time: 278.8s
Total training time: 278.8s
Training Complete

```
