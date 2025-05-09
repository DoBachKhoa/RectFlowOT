# Linear-Time Transport with Rectified Flows 

Demo code.

## Stippling from an image

In this folder: 

```
c++ -O3 -o main main.cpp flow.cpp-I.
```

Running this example should produce an `outAdvectedBNOT.svg` file with the stippling of the image `lionOrigami.bmp`.

Alternatively, CUDA code is available to run on the GPU:

```
nvcc -O3 -o stippling_cuda stippling_cuda.cu
```

Running this code should similarly produce `output.svg'.

## Interactive stippling

These demos require the opencv library.

```
cd opencv-demo
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Then, the  `brush` executable samples a density you can adjust with the mouse (left and right clicks). Upsampling can be achieved by the "u" key.

The `webcam` uses the default webcam (if present) and samples it as a density.


## Interpolation

Interpolation CUDA demo (not much faster than analogous CPU version):

```
nvcc -O3 -o interp_cuda interp_cuda.cu
```

Running this code should similarly produce `out_interp.png`, interpolating from `caterpillar.png` to `butterfly_simple.png` with 5 interpolation steps.