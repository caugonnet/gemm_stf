#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main(int argc, char **argv)
{
    size_t N = 640;
    size_t NBLOCKS = 128;
    size_t NITER = 20;

    if (argc > 1) {
       N = atol(argv[1]);
       fprintf(stderr, "N = %zu\n", N);
    }

    if (argc > 2) {
       NBLOCKS = atol(argv[2]);
       fprintf(stderr, "NBLOCKS = %zu\n", NBLOCKS);
    }

    if (argc > 3) {
       NITER = atol(argv[3]);
       fprintf(stderr, "NITER = %zu\n", NITER);
    }

    cublasHandle_t handle;
    cuda_safe_call(cublasCreate(&handle));

    context ctx;
    if (argc > 4) {
        if (atoi(argv[4])) {
           fprintf(stderr, "Using CUDA graphs.\n");
           ctx = graph_ctx();
        }
    }

    std::vector<logical_data<slice<double, 2>>> vA;
    std::vector<logical_data<slice<double, 2>>> vB;
    std::vector<logical_data<slice<double, 2>>> vC;

    for (size_t k = 0; k < NBLOCKS; k++)
    {
        vA.push_back(ctx.logical_data(shape_of<slice<double, 2>>(N, N)));
        vB.push_back(ctx.logical_data(shape_of<slice<double, 2>>(N, N)));
        vC.push_back(ctx.logical_data(shape_of<slice<double, 2>>(N, N)));
    }

    for (size_t k = 0; k < NBLOCKS; k++)
    {
        ctx.parallel_for(vA[k].shape(), vA[k].write(), vB[k].write(), vC[k].write())->*[N]__device__(size_t i, size_t j, auto a, auto b, auto c) {
            a(i, j) = (1.0*(i+j))/N;
            b(i, j) = (1.0*(i-j))/N;
            c(i, j) = 1.0;
        };
    }

    cudaEvent_t start, stop;
    cuda_safe_call(cudaEventCreate(&start));
    cuda_safe_call(cudaEventCreate(&stop));
    cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

    for (size_t iter = 0; iter < NITER; iter++)
    {
        fprintf(stderr, "Iteration %zu\n", iter);

        for (size_t k = 0; k < NBLOCKS; k++) {
            ctx.task(vA[k].read(), vB[k].read(), vC[k].rw())->*[&](cudaStream_t stream, auto a, auto b, auto c) {
                    const double alpha = 1.0;
                    const double beta = 1.0;
                    cuda_safe_call(cublasSetStream(handle, stream));
                    cuda_safe_call(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N,
                        &alpha, a.data_handle(), N, b.data_handle(), N,
                        &beta,  c.data_handle(), N));
            };
        }

        // As a side effect, it will generate a new CUDA graph
        ctx.task_fence();
    }

    cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

    ctx.finalize();

    float elapsed;
    cuda_safe_call(cudaEventElapsedTime(&elapsed, start, stop));
    fprintf(stderr, "Elapsed : %f ms per iteration.\n", elapsed/NITER);
}
