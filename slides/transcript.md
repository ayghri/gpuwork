# Workshop Transcript: GPU Programming with CUDA and Triton

Total duration: 120 minutes (2 hours)

Speaking rate assumption: 150 words/minute

---

## PART 1: Why GPU Programming? [0:00 , 0:18] (18 min)

### Slide: Title Page [0:00 , 0:03] (3 min, ~450 words)

Welcome everyone to the GPU programming workshop , "CUDA and Triton, From Zero to Hero." My name is Ayoub, and over the next two hours, we're going to go from zero experience with GPU programming to writing real, working CUDA kernels and Triton programs that you'll run yourselves.

Here's what you'll walk away with. You'll understand how a GPU is structured and why it's different from a CPU. You'll write a vector addition kernel in CUDA , that's the "hello world" of GPU programming. Then you'll write a matrix multiply, which is where things get really interesting , shared memory, cache tiling, bank conflicts. And finally, you'll see how Triton lets you do all of that in Python with a fraction of the code.

Now, the best part: you don't need a GPU on your laptop. We're going to use a platform called LeetGPU , think of it like LeetCode, but for GPU kernels. It runs in your browser, it has a real GPU backend, and you just paste code and hit submit. So if you haven't already, go ahead and open leetgpu.com right now. Sign in with Google or GitHub , it takes ten seconds. You'll want that tab ready when we get to the first exercise in about fifteen minutes.

One more thing about LeetGPU that we'll use throughout: it shows you the generated PTX for your submissions. PTX is the assembly language of NVIDIA GPUs. Being able to see what your CUDA compiles down to is incredibly valuable, and we'll look at it after every exercise.

All the exercise files are in the workshop repo at github.com/ayghri/gpuwork. Each one is a `.cu` file with starter code and TODOs for you to fill in. Solutions are also there if you get stuck. Go ahead and clone it or just browse the files on GitHub.

### Slide: Workshop Outline [0:03 , 0:03] (~30 words)

Here's the table of contents. Six sections. We won't dwell on this , you'll see it unfold.

### Slide: Workshop Roadmap [0:03 , 0:05] (2 min, ~300 words)

Let me walk you through the plan. Part 1, which we're in now, covers why GPUs exist and how they're different from CPUs. This is the conceptual foundation , memory hierarchy, the memory wall, why parallelism matters.

Part 2 is where we start coding. We'll learn the CUDA execution model , grids, blocks, threads , and immediately write a vector addition kernel. We'll look at the CPU baseline first, with OpenMP parallelism, and then the CUDA version. After that, we'll look at the PTX output to see what the GPU actually executes. You'll do two exercises: basic vec_add, then a fused vec_add plus ReLU to see kernel fusion in action.

Part 3 is the meat of CUDA. Naive matrix multiply, then we'll see why it's slow, introduce shared memory, and build a tiled version. We'll dig into bank conflicts and how to fix them. There are two more exercises here.

Then we take a short break, and we switch to Triton for parts 4 and 5. Triton is a Python-based GPU programming language that handles most of the low-level stuff for you. We'll redo vector add, softmax, and matmul in Triton and compare.

Part 6 is the wrap-up, and if we have time, we'll look at Flash Attention as a bonus.

All the CUDA exercises use LeetGPU , browser-based, just copy-paste and run. The Triton exercises are Python scripts you can run locally if you have a GPU, or just read through.

### Slide: CPU vs GPU: Design Philosophy [0:05 , 0:09] (4 min, ~600 words)

Alright, let's start with the fundamental question: why do GPUs exist? Why can't we just use CPUs for everything?

A CPU is designed to execute a single thread of instructions as fast as humanly possible. It has huge caches , sometimes 30 or 40 megabytes of L3 cache , sophisticated branch prediction that guesses which way your if-statements will go, out-of-order execution that rearranges your instructions to avoid stalls. All of this machinery exists to minimize the latency of one thread. A modern CPU might have 8, 16, maybe 64 cores, and each core is an incredibly complex piece of engineering.

A GPU takes the opposite approach. It says: I don't care about making one thread fast. Instead, I'm going to have thousands of tiny, simple cores, and I'm going to run thousands of threads simultaneously. Each individual thread is much slower than a CPU thread , smaller caches, no branch prediction, simpler execution units. But the aggregate throughput , the total amount of work done per second across all threads , is enormous.

The analogy I like is: a CPU is a sports car. One passenger, very fast, great handling. A GPU is a city bus. Each trip is slower, you don't get a nice leather seat, but you move a hundred people at once. If your job is to move one VIP across town, use the sports car. If your job is to move ten thousand commuters, you want the bus fleet.

So when do you reach for a GPU? When you have the same operation applied to millions of elements, and each element can be processed independently. Vector addition: c[i] = a[i] + b[i] for a million values of i , every element is independent. Matrix multiplication: each output element is an independent dot product. Convolutions, reductions, elementwise activations , all massively parallel.

Deep learning is the poster child for GPU computing, but it's not the only one. Scientific simulations, rendering, financial Monte Carlo, signal processing , anything with data parallelism.

Conversely, if your code is inherently sequential , each step depends on the previous one , or if it's branchy with lots of if-else decisions, or if your dataset is small, the CPU will probably be faster. The overhead of moving data to the GPU and launching kernels has a cost, and for small problems, that cost dominates.

### Slide: GPU Architecture Overview [0:09 , 0:12] (3 min, ~450 words)

Let's look at what's actually inside a GPU. I'll use the NVIDIA A100 as our running example since it's widely deployed.

The main building block is called a Streaming Multiprocessor, or SM. Think of it as a semi-independent processor. The A100 has 108 SMs.

Each SM contains several things. First, CUDA cores , these are the arithmetic units that do integer and floating-point math. The A100 has 6912 FP32 cores total, spread across those 108 SMs. Then there are Tensor Cores , specialized hardware for matrix multiply-accumulate operations. We'll talk about those later, but they're what make modern GPUs absurdly fast at deep learning. Each SM also has shared memory and an L1 cache , fast on-chip memory that we'll spend a lot of time on. There's a register file, and there are warp schedulers.

Now, the most important concept on this slide: a warp. A warp is a group of 32 threads that execute the same instruction in lockstep. This execution model is called SIMT , Single Instruction, Multiple Threads. When thread 0 in a warp executes an add instruction, threads 1 through 31 all execute that same add instruction at the same time, just on different data. If you remember one number from this whole workshop, remember 32.

The A100 runs at 1.41 GHz boost clock, has 80 GB of HBM2e memory with 2 terabytes per second of bandwidth. Those bandwidth numbers will become very important when we talk about the memory wall.

### Slide: GPU Memory Hierarchy [0:12 , 0:15] (3 min, ~450 words)

This slide is arguably the most important in the entire workshop. Everything we do in GPU programming comes down to this memory hierarchy.

On the left you can see the physical layout. Each SM has its own registers and shared memory , that's the fast, on-chip storage. Then there's a shared L2 cache across all SMs, and at the bottom, global memory , the big HBM DRAM. On the right, the latency stack tells the story.

At the bottom: global memory. HBM, 80 gigabytes, about 2 terabytes per second bandwidth. Sounds fast, right? But the latency is roughly 400 clock cycles. Every time a thread needs data that's only in global memory, it stalls for 400 cycles.

Next up: L2 cache. 40 megabytes, about 4 TB/s, roughly 200 cycles. You don't control what goes here , the hardware manages it , but it helps with repeated accesses to the same region.

Then: shared memory, also called L1 or SMEM. Look at the diagram , each SM has its own 164 KB block. About 20 cycles latency. This is the one we'll spend the most time on. Unlike L2, shared memory is programmer-managed. You explicitly load data into it, and you explicitly read from it. It's fast, it's on-chip, and it's the key to writing efficient GPU code.

Finally: registers. 256 KB per SM, 1 cycle. The fastest memory available, but private to each thread. Your local variables live here.

Here's the punchline. The entire art of GPU programming is keeping your data as high up this pyramid as possible. Every trip down to global memory costs you 400 cycles where your compute units are doing nothing. The best GPU code loads data once from global memory, stages it in shared memory, does as much computation as possible, and only then writes the result back.

### Slide: The Memory Wall [0:15 , 0:18] (3 min, ~450 words)

Now let's put numbers to this. Here's the uncomfortable truth about GPU programming: most kernels are memory-bound, not compute-bound.

The A100 can do 312 TFLOPS of FP16 math. That's 312 trillion floating-point operations per second. But it can only feed data to those compute units at 2 terabytes per second. Let's do the math: 312 TFLOPS divided by 2 TB/s equals 156 FLOPs per byte. That means you need to do at least 156 floating-point operations for every byte you load from memory just to keep the compute units busy. That ratio , FLOPs per byte , is called the arithmetic intensity.

Now take a simple operation like ReLU: max(x, 0). You load one float , 4 bytes. You do one comparison. You store one float , 4 bytes. That's 1 FLOP for 8 bytes of memory traffic. Arithmetic intensity of 0.125. The compute units finish their one operation and then wait 400 cycles for the next piece of data. You're using a tiny fraction of the GPU's potential.

This is called the memory wall, and it's why kernel fusion is the single most important optimization technique. If you have two operations , say, a vector add followed by a ReLU , and you run them as separate kernels, each one reads from global memory and writes back to global memory. That's four trips through HBM per element. But if you fuse them into one kernel, you load the inputs once, do the add and the ReLU in registers, and write once. Two trips instead of four. We'll do exactly this in Exercise 1b.

This principle applies everywhere. Softmax in PyTorch is five separate kernels , max, subtract, exp, sum, divide. Fused softmax is one kernel. Same result, half the memory traffic, two to four times faster.

Any questions before we start writing code?

*[Pause 30 seconds for questions]*

---

## PART 2: CUDA , Vector Addition [0:18 , 0:55] (37 min)

### Slide: Grid, Blocks, and Threads [0:18 , 0:23] (5 min, ~750 words)

Alright, let's write some code. But first, we need to understand how CUDA organizes parallel work, because this is the mental model you'll use every time you write a kernel.

When you launch a kernel on the GPU, you specify a Grid. The grid is the entire collection of threads that will execute your kernel. The grid is divided into Blocks , also called CTAs, Cooperative Thread Arrays. And each block contains individual Threads.

Look at this diagram. We have a grid with two blocks , block (0,0) and block (1,0). Each block has 8 threads, labeled T0 through T7. Now look at the dotted yellow outlines , those are warps. T0 through T3 form warp 0, T4 through T7 form warp 1. In real hardware, a warp is 32 threads, but we're using 4 here to keep the diagram simple.

Now, here's the key hierarchy and what each level means for you as a programmer:

Threads are the finest grain. Each thread executes the kernel function independently with its own unique ID. It has its own registers , its private scratchpad.

Warps are groups of 32 threads that execute in lockstep. You don't explicitly create warps , the hardware groups consecutive threads automatically. Thread 0 through 31 are warp 0, thread 32 through 63 are warp 1, and so on. This is important because if threads in a warp take different branches in an if-statement, performance degrades.

Blocks , or CTAs , are the unit of cooperation. All threads in a block can share data through shared memory and synchronize with barriers. A block runs on a single SM and cannot migrate. Blocks can have up to 1024 threads.

The grid is the collection of all blocks. Blocks within a grid are independent , they can run in any order, on any SM. This is what makes GPUs scalable: if you have 108 SMs and 1000 blocks, the GPU scheduler assigns blocks to SMs as they become available.

For your kernel code: you write the function that one thread executes. Then at launch time, you say how many blocks and how many threads per block. The GPU does the rest.

### Slide: Thread Indexing [0:23 , 0:25] (2 min, ~300 words)

Every thread needs to figure out what piece of data it's responsible for. CUDA provides built-in variables for this.

`threadIdx.x` gives you the thread's position within its block , a number from 0 to blockDim.x minus 1. `blockIdx.x` tells you which block this thread belongs to. `blockDim.x` is the total number of threads per block. And `gridDim.x` is the total number of blocks.

The standard formula for a 1D global thread ID is: `int i = blockIdx.x * blockDim.x + threadIdx.x`. This gives every thread in the entire grid a unique index.

Concrete example: if you launch 4 blocks of 256 threads each, block 0 gets global indices 0 through 255, block 1 gets 256 through 511, block 2 gets 512 through 767, block 3 gets 768 through 1023. Every thread knows its unique position. You'll write this formula in every single CUDA kernel you ever write. Memorize it.

These can also be 2D or 3D , `threadIdx.y`, `blockIdx.y`, and so on , which we'll use for matrix multiply.

### Slide: CPU Baseline: Vector Addition [0:25 , 0:27] (2 min, ~300 words)

Before we write the GPU version, let's establish a baseline. Here's vector addition on the CPU.

The sequential version is as simple as it gets: a for loop, one element at a time. On a modern CPU, this might process a few hundred million elements per second. Single-threaded.

Below that, the OpenMP version. One pragma , `#pragma omp parallel for` , and suddenly the loop is parallelized across all CPU cores. If your machine has 16 cores, you get roughly 16 times the throughput. This is the CPU's best effort at parallelism.

The file `exercises/00_cpu_vec_add.cu` has both versions with timing code. You can compile it with `gcc -O2 -fopenmp` and run it locally to get actual numbers for your machine. Keep those numbers in mind , when we do the GPU version, we'll see how the comparison works out. Spoiler: for vector addition, the GPU isn't always faster, because the operation is so memory-bound that the time is dominated by data transfer. The GPU really shines when the computation per element is higher.

### Slide: First Kernel: Vector Addition [0:27 , 0:31] (4 min, ~600 words)

Here's our first CUDA kernel. Let me walk through every single line.

`__global__` , this keyword is CUDA-specific. It tells the compiler: this function is a kernel. It gets called from the CPU but runs on the GPU. There are also `__device__` functions that run on the GPU and are called from other GPU functions, and `__host__` functions which are normal CPU functions. `__global__` is the bridge.

The parameters: `float *a, float *b, float *c, int n`. These pointers point to GPU memory , memory that was allocated with `cudaMalloc`. The CPU passes these pointers to the kernel, and every thread on the GPU receives the same pointers.

`int i = blockIdx.x * blockDim.x + threadIdx.x` , our global index formula. Thread 0 of block 0 gets i=0. Thread 1 of block 0 gets i=1. Thread 0 of block 1 gets i=256, assuming 256 threads per block. Every thread computes a unique i.

`if (i < n)` , the boundary guard. If we have, say, 1000 elements and we launch 4 blocks of 256, that's 1024 threads. Threads 1000 through 1023 have no work to do, so they skip the computation. Without this check, they'd read and write out-of-bounds memory , a segfault on the GPU.

`c[i] = a[i] + b[i]` , the actual computation. Each thread does one addition. That's the entire kernel.

Below that, the launch syntax: `vec_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n)`. The triple angle brackets are CUDA's kernel launch syntax. The first number is how many blocks, the second is how many threads per block. `(n + block_size - 1) / block_size` is a ceiling division that ensures we launch enough threads to cover all n elements.

### Slide: What Does the GPU Actually Run? PTX [0:31 , 0:34] (3 min, ~450 words)

Now here's something really cool. When you write CUDA and compile it with `nvcc`, it doesn't produce x86 machine code. It produces PTX , Parallel Thread Execution , which is NVIDIA's virtual instruction set for GPUs. Think of it as GPU assembly language.

Look at what our vec_add kernel compiles to. Four lines of PTX:

`ld.global.f32` , load a 32-bit float from global memory. That's reading `a[i]`. Another `ld.global.f32` for `b[i]`. Then `add.f32` , add two floats. And `st.global.f32` , store the result to global memory. That's it. That's what the GPU is actually executing per thread.

Why should you care about PTX? Three reasons.

First, it tells you what the compiler actually did. Did it use shared memory? How many registers per thread? Did it unroll your loop? Sometimes the compiler makes choices you didn't expect, and PTX is where you see it.

Second, LeetGPU shows you the generated PTX for every submission. After you solve each exercise, click the PTX tab. You'll see your kernel's actual instructions. This is an incredibly valuable learning tool , it connects your high-level C code to the hardware.

Third, there's an amazing tool called Compiler Explorer , godbolt.org , that lets you type CUDA code on the left and see the PTX and SASS output on the right, in real time. SASS is the final machine code for a specific GPU architecture. I highly recommend bookmarking it. You can generate PTX locally too: `nvcc -ptx vec_add.cu` produces a `.ptx` file.

As we go through the exercises, I'll point out specific PTX instructions to look for. For now, remember: `ld.global` is a global memory load, `st.global` is a global store, and `ld.shared` / `st.shared` are shared memory operations. When we add shared memory tiling later, you'll see those `ld.global` instructions in the inner loop get replaced by `ld.shared` , that's the 20x speedup in action.

### Slide: Exercise 1c: Vec Add with Inline PTX [0:34 , 0:36] (2 min, ~300 words)

You can actually write PTX instructions directly in your CUDA code using inline assembly. File `exercises/01c_ptx_vec_add.cu` shows how.

The key line is: `asm("add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b))`. Let me break that down. `"add.f32 %0, %1, %2"` is the PTX instruction , add two 32-bit floats. `%0`, `%1`, `%2` are placeholders. After the first colon, `"=f"(c)` says: `%0` is an output, it's a float register, and bind it to the C variable `c`. After the second colon, `"f"(a)` and `"f"(b)` are the inputs bound to `a` and `b`.

The punchline: the normal version `c = a + b` compiles to the exact same `add.f32` instruction. The inline PTX version just makes it explicit. But this technique becomes essential when you need instructions that the compiler doesn't emit on its own , for example, warp-level shuffles, special math instructions, or Tensor Core `mma.sync` operations.

Take a look at the file. It has both the PTX version and the normal version side by side. Try running `nvcc -ptx` on it and comparing the generated PTX for both kernels , they should be nearly identical.

### Slide: Host Code: Allocate, Copy, Launch, Copy Back [0:36 , 0:39] (3 min, ~450 words)

Now let's see the full host-side code , the CPU part that orchestrates everything.

Step 1: `cudaMalloc` allocates memory on the GPU. Notice we use `float *d_a` , the `d_` prefix is a convention for "device" pointers. These pointers point to GPU memory and cannot be dereferenced on the CPU.

Step 2: `cudaMemcpy` with `cudaMemcpyHostToDevice` copies data from CPU RAM to GPU memory. This goes over the PCIe bus , or NVLink if you're on a server , and it's relatively slow. For large transfers, this can take milliseconds.

Step 3: launch the kernel with the triple angle bracket syntax. This call returns immediately , the CPU doesn't wait for the GPU to finish. It just enqueues the work.

Step 4: `cudaMemcpy` with `cudaMemcpyDeviceToHost` copies results back. This call is synchronous , it waits for the kernel to complete before starting the transfer.

Step 5: `cudaFree` releases GPU memory.

This allocate-copy-launch-copy-free pattern is the fundamental workflow of CUDA. It's verbose, it's boilerplate, and you'll write it hundreds of times. Later, when we see Triton, you'll appreciate that it handles all of this for you through PyTorch tensor integration.

One important note: the `<<<grid, block>>>` syntax is the center of this slide. That's where you control parallelism. Too few threads and the GPU is underutilized. Too many and you waste resources. For vector add, 256 threads per block is a safe default.

### Slide: What Just Happened? [0:39 , 0:41] (2 min, ~300 words)

Let me visualize the flow with this diagram. Four boxes: cudaMalloc, then memcpy host-to-device, then the kernel launch, then memcpy device-to-host.

The two critical things to understand: first, CPU and GPU have completely separate memory spaces. A pointer allocated with `malloc` is useless on the GPU, and a pointer from `cudaMalloc` is useless on the CPU. You must explicitly copy data back and forth.

Second, the kernel launch is asynchronous. The CPU fires off the kernel and immediately moves to the next line of code. It's the `cudaMemcpy` device-to-host that forces synchronization , the CPU waits for the kernel to finish before starting the copy.

For our million-element vec_add: 4096 blocks times 256 threads per block equals 1,048,576 threads. All running the same kernel function, each processing one element. That's the GPU's strength , launching a million threads is essentially free.

### Slide: Warps and SIMT [0:41 , 0:43] (2 min, ~300 words)

One more concept before we exercise. We talked about warps , groups of 32 threads executing in lockstep. This has an important consequence called warp divergence.

If you write code like `if (threadIdx.x % 2 == 0) { do_A(); } else { do_B(); }`, the warp can't execute both branches simultaneously. Instead, it executes `do_A()` with half the threads masked off, then `do_B()` with the other half masked off. Both branches run sequentially. You've lost half your throughput.

In our vec_add kernel, the `if (i < n)` guard only causes divergence in the very last block, where a few threads are past the end of the array. That affects at most one warp out of thousands , totally negligible.

The rule of thumb: avoid branching within a warp. If you need conditional logic, try to make it so that all threads in a warp take the same branch. Branch across warp boundaries, not within them.

### Slide: Exercise 1: Vector Addition [0:43 , 0:49] (6 min: ~1 min intro + 5 min exercise)

*[~150 words intro]*

OK, your turn! Open leetgpu.com and paste the code from `exercises/01_vec_add.cu`.

You need to do two things: compute the global thread index `i` using the formula we just learned, and write the boundary check plus the addition. The `main()` function is already written for you , it handles allocation, copying, launching, and verification.

You have 5 minutes. If you finish early, click the PTX tab on LeetGPU and look at the output. Find the `ld.global.f32`, `add.f32`, and `st.global.f32` instructions , those are your kernel in action. Also try changing the block size from 256 to 128 or 512 and see if performance changes.

*[5 min exercise. Walk around, help people. Common issues: forgetting the boundary check, using wrong index formula, using threadIdx.y instead of threadIdx.x.]*

Has everyone gotten it to pass? Show of hands? Great. If you didn't finish, the solution is in `exercises/01_vec_add_solution.cu`.

### Slide: Exercise 1b: Fused Vector Add + ReLU [0:49 , 0:55] (6 min: ~1.5 min intro + 4 min exercise + 30 sec debrief)

*[~225 words intro]*

Now let's apply the memory wall lesson. Open `exercises/01b_fused_vec_add_relu.cu`.

This file has two versions. The unfused version runs two separate kernels , first vec_add writes to a temporary buffer, then relu reads that buffer and writes the final output. That's four global memory accesses per element: read A, read B, write tmp, then read tmp, write C.

Your job: write the fused version. One kernel that computes `C[i] = max(A[i] + B[i], 0.0f)` in a single step. Use the CUDA function `fmaxf`. That cuts it down to two memory accesses per element , read A, read B, write C. The intermediate value never touches global memory.

You have 4 minutes. This is a quick one. When you're done, check the PTX: in the fused version, you should see two `ld.global`, one `add`, one `max`, and one `st.global` , no intermediate store and reload.

*[4 min exercise]*

This pattern , fusing operations to avoid intermediate memory traffic , is the single most impactful optimization in GPU programming. It's the reason `torch.compile` exists, it's the reason Triton was created, and it's the reason Flash Attention is fast. You'll see this theme repeat throughout the workshop.

---

## PART 3: CUDA , Matrix Multiply [0:55 , 1:25] (30 min)

### Slide: Naive Matrix Multiplication [0:55 , 0:59] (4 min, ~600 words)

Let's level up. Matrix multiplication is the workhorse of deep learning, and it's where GPU programming gets genuinely interesting.

We want to compute C = A times B, where A is M-by-K and B is K-by-N. Each element of C is a dot product: C[row][col] = sum over k of A[row][k] times B[k][col].

In the naive CUDA version, each thread computes one element of C. We use a 2D grid and 2D blocks. `blockIdx.y` and `threadIdx.y` give us the row, `blockIdx.x` and `threadIdx.x` give us the column. Same indexing idea as vec_add, but in two dimensions.

Inside the kernel, each thread loops over the K dimension, accumulating the dot product. `sum += A[row * K + k] * B[k * N + col]` , note the row-major indexing. A is stored row by row in memory, so `A[row][k]` is at address `row * K + k`.

We launch this with `dim3 block(16, 16)` , that's 256 threads per block, arranged in a 16-by-16 grid. And `dim3 grid(N/16, M/16)` blocks, rounded up.

This works. It produces correct results. But look at the memory access pattern. Each thread in row `row` reads the entire row of A , all K elements. And every other thread in that same row reads the exact same K elements. That's massive redundancy. For a 4096x4096 matrix, each element of A is loaded from global memory 4096 times. Same for B. If you check the PTX, you'll see `ld.global` in the inner loop , every single multiply-add goes to HBM.

### Slide: Exercise 2: Naive Matrix Multiplication [0:59 , 1:05] (6 min: ~1 min intro + 5 min exercise)

*[~150 words intro]*

Your turn. File: `exercises/02_naive_matmul.cu`.

Fill in the row and col computation using 2D block and thread indices, then write the dot product loop. Remember the row-major layout: `A[row][k]` is `A[row * K + k]`, and `B[k][col]` is `B[k * N + col]`. The launch configuration is set to `dim3 block(16, 16)`.

You have 5 minutes. After it passes, check the PTX and look at the inner loop , it's all `ld.global` instructions. We're about to fix that.

*[5 min exercise. Walk around. Common issues: mixing up M, N, K; using row * N + k instead of row * K + k; forgetting boundary check.]*

Good. Now let's understand why this version is slow.

### Slide: Why the Naive Version is Slow [1:05 , 1:08] (3 min, ~450 words)

With M = N = K = 4096: there are 16.8 million output elements. Each one requires 4096 multiply-add operations, and each multiply-add loads two floats from global memory. That's 2 times 4096 cubed times 4 bytes, which comes out to about 549 gigabytes of global memory traffic.

But think about it , row i of A is needed by every element in row i of C. That's 4096 elements all reading the same row. The same row is loaded 4096 separate times from global memory, once per thread. Column j of B has the same problem.

This is insane. We're loading the same data thousands of times from the slowest memory tier. Remember the memory hierarchy , global memory is 400 cycles. Shared memory is 20 cycles.

What if, instead of each thread loading from global memory every time, we had the block collaboratively load a tile of A and a tile of B into shared memory, and then every thread in the block reads from that fast shared copy? Each tile would be loaded once instead of hundreds of times.

That's the idea behind tiled matrix multiplication, and it's what separates a naive GPU programmer from an effective one.

### Slide: Shared Memory: On-Chip Cache You Control [1:08 , 1:11] (3 min, ~450 words)

Shared memory is fast, on-chip SRAM that all threads in a block can read and write. You declare it with the `__shared__` keyword.

Here's the usage pattern: you declare a shared memory array, like `__shared__ float tile[16][16]`. Each thread loads one element from global memory into the shared array. Then you call `__syncthreads()` , this is a barrier that makes every thread in the block wait until all of them have finished their loads. After the barrier, any thread can safely read any element from the shared array.

This is crucial: `__syncthreads()` is the thing that makes cooperative loading work. Without it, thread 5 might try to read from `tile[3][7]` before thread 47 has finished writing to it. That's a race condition. The barrier prevents this.

Shared memory on the A100 is 164 KB per SM, configurable between shared memory and L1 cache. Latency is about 20 cycles , 20 times faster than global memory. But it's small, which is why we load tiles, not entire matrices.

The pattern is: load a tile collaboratively, synchronize, compute from the tile, synchronize again before overwriting. Load, sync, compute, sync. That's the rhythm of tiled GPU computing.

### Slide: Tiled Matmul: Loading Tiles into Shared Memory [1:11 , 1:14] (3 min, ~450 words)

Here's the tiled version. We define TILE as 16 , our tile size.

The outer loop iterates over the K dimension in steps of TILE. Each iteration, we load a 16x16 tile of A and a 16x16 tile of B into shared memory. Every thread loads exactly one element of each , that's 256 threads loading 256 elements, perfectly parallel.

`As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x]` , thread at position (ty, tx) in the block loads the element at row `row`, column `t*TILE + tx` from the global A matrix.

Similarly for Bs. Then `__syncthreads()` , everyone waits for all loads to complete.

Now the inner loop: for k from 0 to TILE, we accumulate `sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]`. This entire inner loop reads only from shared memory. 16 multiply-adds, all at 20-cycle latency instead of 400.

Then another `__syncthreads()` before the next tile iteration , we need to make sure nobody is still reading from the old tile before we overwrite it with new data.

After all tiles are processed, each thread writes its final `sum` to the output matrix C in global memory.

### Slide: Cache Reuse: Why Tiling Works [1:14 , 1:16] (2 min, ~300 words)

Let's quantify the improvement.

Without tiling, each multiply-add loads two floats from global memory. Total traffic: O(MNK), which for 4096 cubed is about 549 GB.

With tiling at tile size T=16, each tile of A and B is loaded once from global memory and then reused T times. The total traffic drops to O(MNK/T) , about 34 GB. That's 16 times less memory traffic. In practice, you see a big speedup just from this one optimization.

Let me walk through one tile iteration step by step. Step 1: all 256 threads each load one element of A's tile and one element of B's tile from global memory into shared memory. Step 2: `__syncthreads()`, everyone waits. Step 3: each thread computes 16 multiply-adds entirely from shared memory , fast. Step 4: another `__syncthreads()`, safe to overwrite. Then the next tile.

The data stays on-chip during the compute phase. That's the whole point.

### Slide: Shared Memory Banks [1:16 , 1:18] (2 min, ~300 words)

Shared memory is fast, but there's a subtlety you need to know about: bank conflicts.

Shared memory is divided into 32 banks , one per thread in a warp. Consecutive 4-byte words map to consecutive banks. So address 0 is in bank 0, address 1 is in bank 1, and so on up to address 31 in bank 31. Then it wraps: address 32 is back in bank 0.

Each bank can serve one request per cycle. If two threads in the same warp access different addresses that happen to be in the same bank, those accesses get serialized. That's a bank conflict , it costs extra cycles.

But if all threads access the same address, the hardware does a broadcast , no conflict. And if every thread accesses a different bank, all 32 accesses happen simultaneously in one cycle. That's the ideal case.

### Slide: Bank Conflicts in Our Matmul [1:18 , 1:19] (1 min, ~150 words)

In our tiled matmul, reading `As[threadIdx.y][k]` is fine , threads in a warp differ in `threadIdx.y` so they hit different rows, which map to different banks. No conflict.

But reading `Bs[k][threadIdx.x]` can be a problem. If the row stride is a multiple of 32 , and with TILE=16, the stride is 16 which divides 32 , threads reading from different columns can end up in the same bank. That's a potential conflict.

### Slide: Fixing Bank Conflicts: Padding [1:19 , 1:20] (1 min, ~150 words)

The fix is dead simple. Change `Bs[TILE][TILE]` to `Bs[TILE][TILE + 1]`. Adding one extra column shifts each row by one bank, breaking the regular pattern that causes conflicts. The cost is 16 extra floats , 64 bytes , of shared memory. Totally negligible. The benefit on large matrices can be 5 to 15 percent speedup.

You can also use `ncu`, NVIDIA's kernel profiler, which reports bank conflicts directly.

### Slide: Exercise 3: Tiled Matmul with Shared Memory [1:20 , 1:25] (5 min: ~1 min intro + 4 min exercise)

*[~150 words intro]*

This is the hardest exercise. File: `exercises/03_tiled_matmul.cu`.

The scaffolding is there: the tile loop, the syncs, the final store. You need to fill in the shared memory loads , with boundary checks, load 0.0f for out-of-bounds , and the inner accumulation loop.

Hint: for tile t, A's tile starts at column `t*TILE`, and B's tile starts at row `t*TILE`. You have 4 minutes.

After it passes, check the PTX. The key difference from the naive version: look for `ld.shared` instead of `ld.global` in the inner loop. That's the tiling doing its job , data comes from on-chip memory now.

*[4 min exercise. Walk around actively , this is where people struggle most. The boundary checks are the tricky part.]*

If you finished and want more, try Exercise 4 in `04_bank_conflict_fix.cu` , add the +1 padding to Bs and see if you get a speedup.

---

## Break [1:25 , 1:28] (3 min)

Let's take a 3-minute break. Stand up, stretch, get water. When we come back, we're switching from CUDA to Triton , same GPU, much less code.

*[3 minute break]*

---

## PART 4: Introduction to Triton [1:28 , 1:41] (13 min)

### Slide: CUDA Optimization Summary [1:28 , 1:29] (1 min, ~150 words)

Welcome back. Quick recap of the CUDA principles we've learned: coalesce global memory accesses , consecutive threads should access consecutive addresses. Cache data in shared memory to avoid repeated global loads. Pad shared memory to avoid bank conflicts. Minimize warp divergence. And maximize occupancy by launching enough blocks and threads. Profile with nsys for the big picture and ncu for per-kernel analysis. And always check the PTX to see what the compiler actually generated.

### Slide: What About Tensor Cores? [1:29 , 1:32] (3 min, ~450 words)

Our tiled matmul uses regular FP32 CUDA cores. Each core does one multiply-add per cycle. But modern GPUs have a secret weapon: Tensor Cores.

Tensor Cores are dedicated matrix-multiply-accumulate hardware. Instead of computing one multiply-add per cycle per core, a Tensor Core computes an entire small matrix multiply , for example, a 16x16x16 tile , in a single operation. On the A100, this gives you 312 TFLOPS in FP16, compared to 19.5 TFLOPS with regular FP32 CUDA cores. That's 16 times faster.

But there's a catch, and it's a big one. To use Tensor Cores, each of the 32 threads in a warp must hold specific fragments of the input and output matrices in its registers. The layout , which thread holds which element , is defined by NVIDIA and varies between GPU generations. Volta uses one layout, Ampere uses another, Hopper uses yet another.

In CUDA, you access Tensor Cores through intrinsics: `wmma` (Warp Matrix Multiply-Accumulate) or the lower-level `mma.sync` PTX instructions. Yes, PTX again , you can actually see the `mma.sync` instructions in the PTX output when Tensor Cores are in use. You have to manage the fragment loading, the register layout, and the accumulation yourself. It works, it's fast, but it's complex and fragile code.

This is one of the biggest motivations for Triton. When you call `tl.dot(a, b)` in Triton, the compiler figures out the optimal Tensor Core instructions and register layouts for your specific GPU. You write one line; the compiler does the rest.

### Slide: From CUDA Pain to Triton [1:32 , 1:34] (2 min, ~300 words)

Let's be honest about the state of affairs. Writing high-performance CUDA is genuinely hard. You have to manage memory allocation, data transfers, block and grid dimensions, shared memory staging with manual tiling, synchronization barriers, bank conflict avoidance, memory coalescing, and now Tensor Core register fragment layouts. Different GPU architectures need different optimizations. An optimized matmul can be hundreds of lines.

What if the compiler could handle all of that? What if you wrote code that said "load this tile, multiply these two tiles, store the result" , at the block level, not the thread level , and the compiler figured out the shared memory, the bank conflicts, the coalescing, and the Tensor Core instructions?

That's exactly what Triton does. You write block-level Python. The compiler generates everything else , shared memory staging, bank-conflict-free layouts, coalesced loads, and hardware-specific Tensor Core instructions. Automatically.

### Slide: What is Triton? [1:34 , 1:36] (2 min, ~300 words)

Triton is an open-source language and compiler created at OpenAI, first released in 2021. You write GPU kernels in Python with near-CUDA performance.

The core insight: the programmer shouldn't think about individual threads. Instead, think about blocks of data. Tell the compiler: "I want to load this tile, do this math on it, and store it." The compiler maps that to threads, shared memory, warps, and hardware instructions.

In practice, Triton gives you about 10x less code than equivalent CUDA. Performance is typically 90 to 100 percent of hand-tuned CUDA. It integrates natively with PyTorch , no C++ bindings, no compilation step, just Python decorators. And it's actively maintained , it's what powers `torch.compile` and `torch.inductor` in PyTorch 2.0 and beyond.

### Slide: CUDA vs Triton: Programming Model [1:36 , 1:38] (2 min, ~300 words)

Let me put it side by side.

In CUDA, you write code for one thread. You are responsible for thread indexing, shared memory allocation and loading, synchronization barriers, memory coalescing, and Tensor Core utilization. You have full control, but you carry the full burden.

In Triton, you write code for one block. The compiler handles thread mapping, shared memory management, synchronization, coalescing, and Tensor Core usage. You have less control, but dramatically less boilerplate.

The mental model shift is fundamental. In CUDA, you think: "thread 47 loads element A[3][12] into shared memory slot [3][12], then waits at a barrier, then computes its portion of the dot product." In Triton, you think: "this block loads a 64x32 tile of A and a 32x64 tile of B, dots them together, and stores the 64x64 result." Much closer to how you think about the math.

### Slide: Triton Core Concepts [1:38 , 1:41] (3 min, ~450 words)

Let me give you the Triton vocabulary you'll need.

`tl.program_id(axis)` , this is Triton's equivalent of `blockIdx`. It tells you which block you are. If your grid is 2D, `tl.program_id(0)` is the first axis and `tl.program_id(1)` is the second.

`tl.arange(0, BLOCK_SIZE)` , this creates a vector of integers from 0 to BLOCK_SIZE-1. It's like creating the thread indices for your block, but in a vectorized way. You use this to compute the memory offsets for all elements your block will process.

`tl.load(pointer, mask=..., other=...)` , loads a block of data from global memory. The mask parameter handles boundary conditions: elements where the mask is False get the `other` value instead of loading from memory. This replaces the `if (i < n)` guard in CUDA.

`tl.store(pointer, value, mask=...)` , writes a block of data to global memory, again with masking for boundaries.

`tl.dot(a, b)` , matrix multiplication of two tiles. This is the big one. The compiler turns this into Tensor Core instructions when the shapes and types are right.

`tl.max`, `tl.sum`, `tl.exp` , reductions and elementwise operations within a block.

One important difference from NumPy or PyTorch: all operations in Triton are block-local. When you write `tl.max(x, axis=0)`, that's the max within your block's data, not across the entire tensor. There's no implicit global communication.

---

## PART 5: Triton in Practice [1:41 , 2:02] (21 min)

### Slide: Triton: Vector Addition [1:41 , 1:44] (3 min, ~450 words)

Let's see it in action. Here's vector addition in Triton.

The `@triton.jit` decorator marks this function as a Triton kernel. `a_ptr`, `b_ptr`, `c_ptr` are raw pointers to GPU memory , Triton extracts these from PyTorch tensors automatically.

`pid = tl.program_id(0)` , which block am I? `offsets = pid * BLOCK + tl.arange(0, BLOCK)` , a vector of global indices this block is responsible for. If BLOCK is 1024 and pid is 3, offsets is [3072, 3073, ..., 4095].

`mask = offsets < n` , a boolean vector. True for valid indices, False for out-of-bounds. This replaces the `if (i < n)` guard.

`a = tl.load(a_ptr + offsets, mask=mask)` , loads a block of values from global memory, substituting zeros for masked-off positions. `b = tl.load(...)` , same for b. `tl.store(c_ptr + offsets, a + b, mask=mask)` , stores the result.

The launch syntax: `vec_add_kernel[grid](a, b, c, n, BLOCK=1024)`. Square brackets for the grid size, parentheses for the arguments. No `cudaMalloc`, no `cudaMemcpy`, no `cudaFree`. Triton works directly with PyTorch tensors that are already on the GPU.

Six lines of kernel code versus about 25 lines in CUDA including the host code. And the performance is essentially identical. Under the hood, Triton compiles this to PTX just like CUDA does , the same `ld.global`, `add.f32`, `st.global` instructions.

### Slide: Exercise 5: Triton Vector Addition [1:44 , 1:48] (4 min: ~1 min intro + 3 min exercise)

*[~150 words intro]*

File: `exercises/05_triton_vec_add.py`. This one is already complete , just run it and study the code. It creates random tensors, calls the Triton kernel, and checks against PyTorch's built-in addition.

As you read it, map each Triton concept to its CUDA equivalent. `tl.program_id(0)` is `blockIdx.x`. `tl.arange` is the thread index computation. The mask is the boundary check. No memory management.

Try changing BLOCK to 512, 2048, or 4096. What happens?

*[3 min exercise/exploration]*

### Slide: Triton: Fused Softmax [1:48 , 1:51] (3 min, ~450 words)

Now let's see where Triton really earns its keep. Softmax is defined as: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x))). It involves five operations: compute the max, subtract it, exponentiate, sum, divide.

In PyTorch, each of these is a separate CUDA kernel launch. Each one reads from global memory and writes back to global memory. That's five round-trips through HBM for what should be a single pass over the data.

In Triton, we write one kernel. Each program instance handles one row of the input matrix. We load the entire row into the block's registers , tl.load. Compute the max , tl.max. Subtract , simple arithmetic, stays in registers. Exponentiate , tl.exp. Sum , tl.sum. Divide. Store the result , tl.store.

One read from global memory, all computation on-chip, one write. The intermediate values , the max, the shifted values, the exponentials, the sum , never touch HBM. They stay in registers and shared memory the entire time.

Look how clean the code is. About 15 lines for the kernel, and it's readable Python. No shared memory declarations, no sync barriers, no bank conflict worries. The compiler handles all of it.

### Slide: Why Fused Softmax is Faster [1:51 , 1:53] (2 min, ~300 words)

Let me put the comparison clearly. PyTorch unfused: five kernel launches, five global memory round-trips per row. Triton fused: one kernel launch, one read plus one write per row.

The speedup is typically 2 to 4x over PyTorch for this operation, purely from eliminating memory traffic. The compute is identical , same number of multiplications, same number of additions. The only difference is where the intermediate data lives: HBM versus on-chip.

This is the same kernel fusion principle we saw with vec_add plus ReLU, but applied to a more realistic and complex operation. And in Triton, it's trivial to implement.

This is also exactly why `torch.compile` exists , it analyzes your PyTorch code, identifies opportunities for fusion, and generates Triton kernels that fuse those operations automatically. You don't even have to write the Triton kernel yourself in many cases.

### Slide: Exercise 6: Triton Fused Softmax [1:53 , 1:56] (3 min: ~45 sec intro + 2 min exercise)

*[~110 words intro]*

File: `exercises/06_triton_softmax.py`. Run it , it computes softmax and checks against `torch.softmax`. Read through the kernel: one load, all compute on-chip, one store.

Try two experiments: change the matrix size to (128, 8192) , does it still work? And remove the `other=-float("inf")` from the tl.load call , what breaks? That negative infinity is critical: masked positions need to contribute nothing to the max and sum.

*[2 min exercise]*

### Slide: Triton: Matrix Multiplication (1/2) [1:56 , 1:58] (2 min, ~300 words)

Now the crown jewel , matmul in Triton.

Part 1: setup. We get our block IDs for the M and N dimensions. Create offset ranges with `tl.arange`. Set up pointers to the first tiles of A and B using broadcasting: `rm[:, None]` creates a column vector, `rk[None, :]` creates a row vector, and when you add them to the base pointer, you get a 2D grid of pointers. This is the Triton idiom for 2D tile addressing.

Initialize the accumulator `acc` to zeros. This will hold our running sum as we iterate over tiles.

### Slide: Triton: Matrix Multiplication (2/2) [1:58 , 2:00] (2 min, ~300 words)

Part 2: the tiled loop. For each K-chunk, load tiles of A and B with masks for boundary handling. Call `tl.dot(a, b)` , this single line compiles to Tensor Core instructions when the types and sizes are right. You can verify this: if you inspect the PTX output, you'll see `mma.sync` instructions instead of individual `fma` operations. Advance the pointers by BLOCK_K.

After the loop, store the accumulated result with a boundary mask.

Compare this to our CUDA tiled matmul: no `__shared__` declarations, no `__syncthreads()` barriers, no manual bank conflict padding, no concern about register fragment layouts. The compiler does all of that. About 30 lines total versus 150 or more for an optimized CUDA matmul.

### Slide: Auto-Tuning in Triton [2:00 , 2:01] (1 min, ~150 words)

Triton also has built-in auto-tuning. You list candidate configurations , different BLOCK_M, BLOCK_N, BLOCK_K sizes and different numbers of warps. Triton benchmarks all of them on the first call, picks the fastest, and caches it. On subsequent calls with the same matrix shapes, it uses the cached winner.

In CUDA, you'd have to write this benchmarking harness yourself. In Triton, it's one decorator.

### Slide: Exercise 7: Triton Matrix Multiplication [2:01 , 2:05] (4 min: ~1 min intro + 3 min exercise)

*[~150 words intro]*

File: `exercises/07_triton_matmul.py`. Run it , it multiplies 512x512 matrices and checks against `torch.mm`.

Compare the code side by side with your CUDA tiled matmul from Exercise 3. Note what's gone: no shared memory, no sync barriers, no bank conflict padding. `tl.dot` handles Tensor Cores automatically.

If time permits, try increasing to 2048x2048 and compare Triton's speed with `torch.mm`, which calls cuBLAS under the hood. How close does Triton get?

*[3 min exercise]*

---

## PART 6: Wrap-Up [2:05 , 2:12] (7 min)

### Slide: CUDA vs Triton: When to Use What? [2:05 , 2:08] (3 min, ~450 words)

Let me walk through this comparison table because these are the decision criteria you'll use in practice.

Learning curve: CUDA is steep , C++, pointer arithmetic, manual memory management, architecture-specific tuning. Triton is gentle , Python, block-level thinking, the compiler handles the details.

Code length: a Triton kernel is typically 5 to 10x shorter than the equivalent CUDA.

Peak performance: CUDA can squeeze out every last percent because you control everything. Triton gets 90 to 100 percent , close enough for almost everyone.

Thread-level control: CUDA gives you full access to warp shuffles, inline PTX, custom memory access patterns. Triton abstracts these away , you can't do warp-level programming.

Shared memory: in CUDA, you manage it manually. In Triton, the compiler manages it. This is a feature, not a limitation , the compiler often does a better job than hand-written code.

Auto-tuning: manual in CUDA (or use external tools), built-in in Triton.

PyTorch integration: CUDA requires C++ extension modules. Triton is native Python.

My recommendation: start with PyTorch. If you need something custom, try `torch.compile` first. If that's not enough, write a Triton kernel. Only reach for CUDA when you need warp-level control or maximum performance in a library that ships to millions of users.

### Slide: The GPU Programming Landscape [2:08 , 2:09] (1 min, ~150 words)

Here's the ecosystem at a glance. At the top: PyTorch and JAX, tensor-level frameworks. Below: CUDA on the left for thread-level control, Triton on the right for block-level Python. `torch.compile` sits above Triton and auto-generates kernels. Everything compiles down to PTX and SASS machine code at the bottom.

Climb the abstraction ladder only as far as you need. Most of the time, `torch.compile` is enough.

### Slide: Summary [2:09 , 2:12] (3 min, ~450 words)

Let me recap the key takeaways.

First: GPUs are throughput machines. Thousands of cores, simple but numerous. The memory hierarchy is the dominant factor in performance. Keep data close to compute , registers, then shared memory, then global memory as a last resort.

Second: CUDA gives you full control over the GPU. You manage the grid, blocks, warps, and threads. Shared memory gives you a fast, programmer-controlled cache for data reuse. But that control comes at a cost , tiling, synchronization, bank conflicts, Tensor Core layouts, architecture-specific tuning. It's powerful but labor-intensive. And always check the PTX to see what the compiler actually generated , it's the best way to understand what's happening.

Third: Triton raises the abstraction to the block level. You write Python, the compiler generates shared memory staging, coalesced accesses, Tensor Core instructions, and more. Built-in auto-tuning picks the best configuration. It integrates natively with PyTorch. For most custom kernels, Triton is the right choice.

Fourth: kernel fusion is the single most impactful optimization for GPU workloads. Fewer trips to HBM, more compute per byte loaded. This is why fused softmax is 2 to 4x faster, why Flash Attention exists, and why `torch.compile` is a game-changer.

The exercise files are yours to keep , the full repo is at github.com/ayghri/gpuwork. For further learning: Triton documentation at triton-lang.org. The CUDA Programming Guide at docs.nvidia.com/cuda. The PTX ISA reference at docs.nvidia.com/cuda/parallel-thread-execution , essential if you want to read PTX output. Compiler Explorer at godbolt.org for interactive CUDA to PTX to SASS visualization. And the GPU MODE community on YouTube and Discord for lectures on all of these topics.

Thank you for your time.

*[Open Q&A for remaining time]*

---

## Bonus: Flash Attention [if time allows, ~8 min]

### Slide: Bonus Material Title Card (~30 words)

If we have extra time, let's look at a real-world example that ties everything together: Flash Attention.

### Slide: Flash Attention: The Idea (3 min, ~450 words)

Standard self-attention computes the product QK^T, which for sequence length n produces an n-by-n matrix. For a sequence length of 8192, that's 67 million elements per attention head , just for the intermediate matrix. Memory usage is O(n^2), and it grows quadratically.

Flash Attention, introduced by Tri Dao in 2022, reformulates the computation. Instead of materializing the full n-by-n attention matrix, it processes Q, K, and V in blocks. For each block of Q rows, it iterates over all blocks of K and V, computing partial attention scores and maintaining a running softmax using online normalization. The full n-by-n matrix is never created. Memory usage drops to O(n).

The result: 2 to 4x faster wall-clock time and 10 to 20x less memory. It's what enables context lengths of 100K or more in modern LLMs.

### Slide: Online Softmax: The Key Trick (2 min, ~300 words)

The core mathematical trick is online softmax. Normally, to compute softmax, you need the max of all values first, which means you need to see every value before you can start. Online softmax breaks this requirement.

You process blocks one at a time. After each block, you maintain a running max and a running sum of exponentials. When a new block has a larger max, you rescale the previous exponentials by the correction factor exp(old_max - new_max). This maintains numerical accuracy while only requiring a single pass.

Applied to attention: you maintain a running output vector, running max, and running sum. For each new K/V block, you compute partial attention scores, rescale the previous output, and accumulate. At the end, divide by the final sum. It's mathematically exact , not an approximation.

### Slide: Flash Attention in Triton (Sketch) (3 min, ~450 words)

Here's what it looks like in Triton. Each program instance handles one block of Q rows. Load the Q block. Initialize running max to negative infinity, running sum to zero, accumulator to zero.

Then the inner loop: for each block of K and V, load them, compute QK^T with `tl.dot`, apply the scaling factor, compute the online softmax update , new max, rescaling factor alpha, new exponentials, updated sum and accumulator. At the end, normalize the output by the final sum and store.

This kernel would be 500 or more lines in optimized CUDA, dealing with shared memory tiling of Q, K, and V, double-buffering, warp-level reduction for the softmax, and Tensor Core fragment management. In Triton, it's about 30 lines, and it achieves 85 to 95 percent of the hand-tuned CUDA performance.

This is the kernel that powers most LLM inference and training today. And you now understand every concept it uses: tiling, shared memory caching, kernel fusion, and online algorithms. That's the journey from zero to hero.

