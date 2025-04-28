# Linear-Time Transport with Rectified Flows 

Demo code.

## Stippling from an image

In this folder: 

```
c++ -O3 -o main main.cpp flow.cpp-I.
```

Running this example should produce an `outAdvectedBNOT.svg` file with the stippling of the image `lionOrigami.bmp`.


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
