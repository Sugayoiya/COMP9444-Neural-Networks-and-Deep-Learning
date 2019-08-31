# COMP9444-Neural-Networks-and-Deep-Learning

cuda9.0 on ubuntu 18.04 install reference: 

https://blossomnoodles.github.io/2018/04/30/ubuntu-18.04-cuda-installation.html

change 
```
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}
```
to
```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64\ 
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

then

`source ~/tensorflow/bin/activate`
