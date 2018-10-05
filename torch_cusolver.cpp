#include <torch/torch.h>
#undef NDEBUG

#include <assert.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

// #include <ATen/CUDAStream.h>

namespace torch_cusolver {
    // solve AV = wV  a.k.a. syevj, where A (batch, m, m), V (batch, m, m), w (batch, m)
    // see also https://docs.nvidia.com/cuda/cusolver/index.html#batchsyevj-example1
    std::tuple<at::Tensor, at::Tensor> batch_symmetric_eigenvalue_solve(
        at::Tensor a, bool in_place=false, bool use_lower=true,
        double tol=1e-7, int max_sweeps=15, bool sort_eig=false)
    {
        // TODO use singleton handler instead of ondemand handle
        // TODO check cutorch or ATen does not handle cusolver
        // https://github.com/torch/cutorch/blob/master/lib/THC/THCGeneral.h.in
        // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAContext.h

        // initialization
        cusolverDnHandle_t handle;
        cusolverStatus_t status;
        status = cusolverDnCreate(&handle);
        assert(status == CUSOLVER_STATUS_SUCCESS);
        // TODO use non blocking stream?
        auto batch_size = a.size(0);
        auto m = a.size(2);
        auto lda = a.stride(1);
        auto w = at::empty({a.size(0), a.size(2)}, a.type());
        // C++ to C API
        // FIXME is this V should be contiguous?
        auto V = in_place ? a.contiguous() : a.clone();
        auto d_V = V.data<float>();
        auto d_W = w.data<float>();
        auto uplo = use_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

        // configure
        syevjInfo_t syevj_params;
        status = cusolverDnCreateSyevjInfo(&syevj_params);
        assert(CUSOLVER_STATUS_SUCCESS == status);
        /* default value of tolerance is machine zero */
        status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
        assert(CUSOLVER_STATUS_SUCCESS == status);
        /* default value of max. sweeps is 100 */
        status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
        assert(CUSOLVER_STATUS_SUCCESS == status);
        /* disable sorting */
        status = cusolverDnXsyevjSetSortEig(syevj_params, sort_eig);
        assert(CUSOLVER_STATUS_SUCCESS == status);

        // FIXME use at::Tensor buffer from arg instead
        // query working space of syevjBatched
        int lwork;
        status = cusolverDnSsyevjBatched_bufferSize(
            handle,
            CUSOLVER_EIG_MODE_VECTOR,
            uplo,
            m,
            d_V,
            lda,
            d_W,
            &lwork,
            syevj_params,
            batch_size
            );
        assert(CUSOLVER_STATUS_SUCCESS == status);
        float* d_work;
        auto stat = cudaMalloc(&d_work, sizeof(float) * lwork);
        assert(cudaSuccess == stat);
        int* d_info;
        auto status_info = cudaMalloc(&d_info, sizeof(int) * batch_size);
        assert(cudaSuccess == status_info);

        // compute eigenvalues/vectors
        status = cusolverDnSsyevjBatched(
            handle,
            CUSOLVER_EIG_MODE_VECTOR,
            uplo,
            m,
            d_V,
            lda,
            d_W,
            d_work,
            lwork,
            d_info,
            syevj_params,
            batch_size
            );
        assert(CUSOLVER_STATUS_SUCCESS == status);

        // free
        if (d_info ) { cudaFree(d_info); }
        if (d_work ) { cudaFree(d_work); }
        if (handle) { cusolverDnDestroy(handle); }
        return std::make_tuple(w, V);
    }

    // solve AV = wBV  a.k.a. syevj, where A (m, m), B (m, m), V (m, m), w (m)
    // see also https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1
    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    generalized_symmetric_eigenvalue_solve(
        at::Tensor a, bool in_place_a, at::Tensor b, bool in_place_b,
        bool use_jacob, double tol=1e-7, int max_sweeps=100
        ) {
        // step 0: create cusolver/cublas handle
        cusolverDnHandle_t cusolverH;
        auto status_create = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == status_create);

        // step 1: copy A and B to device
        // FIXME is this V/B should be contiguous?
        auto m = a.size(0);
        auto lda = a.stride(0);
        auto ldb = b.stride(0);
        // NOTE: V will be overwritten from A to orthonormal eigenvectors
        auto V = in_place_a ? a.contiguous() : a.clone();
        auto d_A = V.data<float>();
        // NOTE: B_ will be overwritten from B to LU-Choresky factorization
        auto B_LU = in_place_b ? b.contiguous() : b.clone();
        auto d_B = B_LU.data<float>();
        // NOTE: w will be sorted
        auto w = at::empty({m}, a.type());
        auto d_W = w.data<float>();
        int* dev_info;
        auto stat_info = cudaMalloc(&dev_info, sizeof(int)); // should be heap allocated?
        assert(cudaSuccess == stat_info);

        cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A V = w B V
        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

        int* d_info;
        auto status_info = cudaMalloc(&d_info, sizeof(int));
        assert(cudaSuccess == status_info);
        float* d_work;
        if (use_jacob)
        {
            // step 2. configure
            syevjInfo_t syevj_params;
            auto status_param = cusolverDnCreateSyevjInfo(&syevj_params);
            assert(CUSOLVER_STATUS_SUCCESS == status_param);
            /* default value of tolerance is machine zero */
            status_param = cusolverDnXsyevjSetTolerance(syevj_params, tol);
            assert(CUSOLVER_STATUS_SUCCESS == status_param);
            /* default value of max. sweeps is 100 */
            status_param = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
            assert(CUSOLVER_STATUS_SUCCESS == status_param);

            // step 3: query working space of sygvd
            int lwork;
            auto status_buffer = cusolverDnSsygvj_bufferSize(
                cusolverH,
                itype,
                jobz,
                uplo,
                m,
                d_A,
                lda,
                d_B,
                ldb,
                d_W,
                &lwork,
                syevj_params);
            assert(status_buffer == CUSOLVER_STATUS_SUCCESS);
            auto stat_work = cudaMalloc(&d_work, sizeof(float)*lwork);
            assert(cudaSuccess == stat_work);

            // step 4: compute spectrum of (A,B)
            auto status_compute = cusolverDnSsygvj(
                cusolverH,
                itype,
                jobz,
                uplo,
                m,
                d_A,
                lda,
                d_B,
                ldb,
                d_W,
                d_work,
                lwork,
                d_info,
                syevj_params);
            auto status_sync = cudaDeviceSynchronize();
            assert(cudaSuccess == status_sync);
            assert(CUSOLVER_STATUS_SUCCESS == status_compute);
        }
        else
        {
            int lwork;
            auto cusolver_status = cusolverDnSsygvd_bufferSize(
                cusolverH,
                itype,
                jobz,
                uplo,
                m,
                d_A,
                lda,
                d_B,
                ldb,
                d_W,
                &lwork);
            assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
            auto cudaStat1 = cudaMalloc(&d_work, sizeof(float)*lwork);
            assert(cudaSuccess == cudaStat1);

            // step 4: compute spectrum of (A,B)
            cusolver_status = cusolverDnSsygvd(
                cusolverH,
                itype,
                jobz,
                uplo,
                m,
                d_A,
                lda,
                d_B,
                ldb,
                d_W,
                d_work,
                lwork,
                dev_info);
            cudaStat1 = cudaDeviceSynchronize();
            assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
            assert(cudaSuccess == cudaStat1);
        }

        // free
        if (dev_info) cudaFree(dev_info);
        if (d_work) cudaFree(d_work);
        if (cusolverH) cusolverDnDestroy(cusolverH);
        return std::make_tuple(w, V, B_LU);
    }
} // namespace torch_cusolver


// generate wrappers
// FIXME do not use legacy preprocessor macro
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cusolver_batch_eigh", &torch_cusolver::batch_symmetric_eigenvalue_solve,
          "cusolver based batched eigh implementation");
    m.def("cusolver_generalized_eigh", &torch_cusolver::generalized_symmetric_eigenvalue_solve,
          "cusolver based generalized eigh implementation");
}
