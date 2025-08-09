# Introduction

Recently I implemented a rust + onnxruntime [inference library](https://github.com/jason-ni/parakeet-rs) for 
[Parakeet-tdt-0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2). As you know, onnxruntime has various 
hardware backends, including CUDA, OpenVINO, and TensorRT, which can significantly improve the inference performance. 
However, when I tested the library on my MacBook Pro with an M4 processor, the inference speed was not as fast as expected. 
Especially compared to the [MLX version of parakeet implementation](https://github.com/senstella/parakeet-mlx).

In my observation, the coreml backend of onnxruntime is not well optimized. It can not even compete with the CPU backend. 
And onnxruntime has no support for the metal MPS backend, which is used by the MLX version of parakeet. As my goal is to 
implement a cross-platform inference library, I would prefer to use an inference engine with a wide range of hardware 
support.

Therefore, I moved my eyes to GGML, which is the tensor engine for llama.cpp. In ASR area, we know that the whisper.cpp
project is built on top of GGML, and it has a good performance. Actually in my investigation, I found that there is a 
[feature request for parakeet implementation on ggml in whisper.cpp project](https://github.com/ggml-org/whisper.cpp/issues/3118#issuecomment-3162156474).
So I decided to spent some time to take this challenge.

# Progress

Currently, the encoder part is done. I can verify the output tensor using the [parakeet-rs library](https://github.com/jason-ni/parakeet-rs/blob/master/examples/from_cpp.rs). 

However, I found the ggml implementation is not as efficient as expected. It actually surprised me that the gap is so large. 
The inference time of the encoder part in parakeet-mlx is 0.001s for my test audio. However, in ggml, the inference time is 
around 1s. The onnxruntime CPU backend inference time is around 1.5s. So I decided to pause here for more review from public.

# How to run the code

1. Install python environment for running parakeet-mlx. Because we need to export gguf model from this project.
2. Download the [parakeet-tdt-0.6B v2 model](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) from huggingface. Please note that it's the mlx version of the model.
3. Clone this repository
 
```bash
git clone https://github.com/jason-ni/parakeet.cpp.git
cd parakeet.cpp
```

4. Export the gguf model using my script.

```bash
python scripts/generate_parakeet_gguf.py
```

5. Build the ggml project.

```bash
mkdir build
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release
```

6. Generate the audio features tensor data.

```bash
python scripts/generate_input_data.py
```

7. Run the inference.

```bash
./build/bin/parakeet_cpp parakeet-tdt-0.6b-v2-float32.gguf assets/pe.bin input.data
```

The expected output looks like:

```
/Users/jason/prj/parakeet.cpp/src/framework.cpp:630:<run> run_schedule took 924304 microseconds
/Users/jason/prj/parakeet.cpp/bin/main.cpp:214:<operator()> tensor data: [
 [ 0.011379,  0.009602,  0.055848, ...  0.079716, -0.007456,  0.065984, ]
 [ 0.018310,  0.006686,  0.035923, ... -0.065628, -0.012651,  0.051945, ]
 [ 0.020104, -0.010873,  0.033619, ... -0.060412, -0.025365,  0.054132, ]
...
 [-0.086789,  0.040847,  0.009242, ... -0.005162, -0.032376, -0.023982, ]
 [-0.067029,  0.009823,  0.035980, ... -0.018739, -0.033858,  0.008342, ]
 [-0.071880, -0.017643,  0.110965, ... -0.091322, -0.004971, -0.038205, ]
],
shape: [1024, 597, 1, 1], type: f32

/Users/jason/prj/parakeet.cpp/bin/main.cpp:215:<operator()> output tensor buft: Metal
/Users/jason/prj/parakeet.cpp/bin/main.cpp:214:<operator()> tensor data: [
 [-0.784943,  0.619568,  0.861500, ...  0.998092,  0.060645,  0.998160, ]
 [-0.945455, -0.325753,  0.056010, ...  0.998098,  0.060543,  0.998166, ]
 [-0.236720, -0.971578, -0.799305, ...  0.998105,  0.060441,  0.998172, ]
...
 [ 0.236720, -0.971578,  0.799305, ...  0.998105, -0.060441,  0.998172, ]
 [ 0.945455, -0.325753, -0.056010, ...  0.998098, -0.060543,  0.998166, ]
 [ 0.784943,  0.619568, -0.861500, ...  0.998092, -0.060645,  0.998160, ]
],
shape: [1024, 1193, 1, 1], type: f32

/Users/jason/prj/parakeet.cpp/bin/main.cpp:215:<operator()> output tensor buft: Metal
duration: 1691640
```


