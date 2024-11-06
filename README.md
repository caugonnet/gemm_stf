# Compiling

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

# 30 iterations of 200 GEMMs with 128x128 blocks.

```bash
./gemm_stf 128 200 30 0
```

# 30 iterations of 200 GEMMs with 128x128 blocks with CUDA graphs.

```bash
./gemm_stf 128 200 30 1
```
