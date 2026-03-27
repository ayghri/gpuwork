CC      = gcc
NVCC    = nvcc
CFLAGS  = -O2
NVFLAGS = -O2

# CPU targets
vec_add_cpu: 00_cpu_vec_add.cu
	$(CC) $(CFLAGS) -fopenmp -x c $< -o $@

# CUDA targets
vec_add: 01_vec_add_solution.cu
	$(NVCC) $(NVFLAGS) $< -o $@

ptx_vec_add: 01c_ptx_vec_add.cu
	$(NVCC) $(NVFLAGS) $< -o $@

special_regs: 01d_special_registers.cu
	$(NVCC) $(NVFLAGS) $< -o $@

vec_add_relu: 01b_fused_vec_add_relu_solution.cu
	$(NVCC) $(NVFLAGS) $< -o $@

naive_matmul: 02_naive_matmul_solution.cu
	$(NVCC) $(NVFLAGS) $< -o $@

tiled_matmul: 03_tiled_matmul_solution.cu
	$(NVCC) $(NVFLAGS) $< -o $@

bank_conflict: 04_bank_conflict_fix.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# Generate PTX from any .cu file (e.g. make 01_vec_add_solution.ptx)
%.ptx: %.cu
	$(NVCC) $(NVFLAGS) -ptx $< -o $@

# Triton targets (run the python scripts)
triton_vec_add:
	python3 05_triton_vec_add.py

triton_softmax:
	python3 06_triton_softmax.py

triton_matmul:
	python3 07_triton_matmul.py

# Groups
cpu: vec_add_cpu
cuda: vec_add ptx_vec_add special_regs vec_add_relu naive_matmul tiled_matmul bank_conflict
all: cpu cuda

clean:
	rm -f vec_add_cpu vec_add ptx_vec_add special_regs vec_add_relu naive_matmul tiled_matmul bank_conflict *.ptx

.PHONY: all cpu cuda clean triton_vec_add triton_softmax triton_matmul
