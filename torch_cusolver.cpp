#include <torch/torch.h>
#undef NDEBUG

#include <assert.h>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

// #include <ATen/CUDAStream.h>

namespace torch_cusolver
{
    template<int success = CUSOLVER_STATUS_SUCCESS, class T, class Status> // , class A = Status(*)(P), class D = Status(*)(T)>
    std::unique_ptr<T, Status(*)(T*)> unique_allocate(Status(allocator)(T**),  Status(deleter)(T*))
    {
        T* ptr;
        auto stat = allocator(&ptr);
        assert(stat == success);
        return {ptr, deleter};
    }

    template <class T>
    std::unique_ptr<T, decltype(&cudaFree)> unique_cuda_ptr(size_t len) {
        T* ptr;
        auto stat = cudaMalloc(&ptr, sizeof(T) * len);
        assert(stat == cudaSuccess);
        return {ptr, cudaFree};
    }

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
        auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);
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
        auto param_ptr = unique_allocate(cusolverDnCreateSyevjInfo, cusolverDnDestroySyevjInfo);
        auto syevj_params = param_ptr.get();
        /* default value of tolerance is machine zero */
        auto status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
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
            handle_ptr.get(),
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
        auto work_ptr = unique_cuda_ptr<float>(lwork);
        auto info_ptr = unique_cuda_ptr<int>(batch_size);

        // compute eigenvalues/vectors
        status = cusolverDnSsyevjBatched(
            handle_ptr.get(),
            CUSOLVER_EIG_MODE_VECTOR,
            uplo,
            m,
            d_V,
            lda,
            d_W,
            work_ptr.get(),
            lwork,
            info_ptr.get(),
            syevj_params,
            batch_size
            );
        assert(CUSOLVER_STATUS_SUCCESS == status);
        return std::make_tuple(w, V);
    }

    // solve AV = wBV  a.k.a. syevj, where A (m, m), B (m, m), V (m, m), w (m)
    // see also https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1
    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    generalized_symmetric_eigenvalue_solve(
        at::Tensor a, bool in_place_a, at::Tensor b, bool in_place_b,
        bool use_upper, bool use_jacob, double tol=1e-7, int max_sweeps=100
        ) {
        auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);

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
        auto info_ptr = unique_cuda_ptr<int>(1);

        cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A V = w B V
        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
        cublasFillMode_t uplo = use_upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

        if (use_jacob)
        {
            syevjInfo_t syevj_params;
            auto status_param = cusolverDnCreateSyevjInfo(&syevj_params);
            assert(CUSOLVER_STATUS_SUCCESS == status_param);
            status_param = cusolverDnXsyevjSetTolerance(syevj_params, tol);
            assert(CUSOLVER_STATUS_SUCCESS == status_param);
            status_param = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
            assert(CUSOLVER_STATUS_SUCCESS == status_param);

            int lwork;
            auto status_buffer = cusolverDnSsygvj_bufferSize(
                handle_ptr.get(),
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
            auto work_ptr = unique_cuda_ptr<float>(lwork);
            auto status_compute = cusolverDnSsygvj(
                handle_ptr.get(),
                itype,
                jobz,
                uplo,
                m,
                d_A,
                lda,
                d_B,
                ldb,
                d_W,
                work_ptr.get(),
                lwork,
                info_ptr.get(),
                syevj_params);
            assert(CUSOLVER_STATUS_SUCCESS == status_compute);
        }
        else
        {
            int lwork;
            auto cusolver_status = cusolverDnSsygvd_bufferSize(
                handle_ptr.get(),
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
            auto work_ptr = unique_cuda_ptr<float>(lwork);
            cusolver_status = cusolverDnSsygvd(
                handle_ptr.get(),
                itype,
                jobz,
                uplo,
                m,
                d_A,
                lda,
                d_B,
                ldb,
                d_W,
                work_ptr.get(),
                lwork,
                info_ptr.get());
            assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        }
        return std::make_tuple(w, V, B_LU);
    }

    // solve L S U = svd(A)  a.k.a. syevj, where A (b, m, m), L (b, m, m), S (b, m), U (b, m, m)
    // see also https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1
    std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_svd(at::Tensor a, bool in_place, bool use_upper, double tol=1e-7, int max_sweeps=100)
    {
        auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);
        auto A = in_place ? a.contiguous() : a.clone();
        auto batch_size = A.size(0);
        auto m = A.size(1);
        auto lda = A.stride(1);
        auto d_A = A.data<float>();
        auto s = at::empty({batch_size, m}, a.type());
        auto d_s = s.data<float>();
        auto L = at::empty({batch_size, m, m}, a.type());
        auto d_L = L.data<float>();
        auto U = at::empty({batch_size, m, m}, a.type());
        auto d_U = U.data<float>();
        auto info_ptr = unique_cuda_ptr<int>(batch_size);

        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
        cublasFillMode_t uplo = use_upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

        /*
        auto status_buffer = cusolverDnSgesvdjBatched_bufferSize(
            handle_ptr.get(),
            jobz,
            cusolverEigMode_t jobz,
            int m,
            int n,
            const float *A,
            int lda,
            const float *S,
            const float *U,
            int ldu,
            const float *V,
            int ldv,
            int *lwork,
            gesvdjInfo_t params,
            int batchSize);
        assert(CUSOLVER_STATUS_SUCCESS == status_buffer);
        */

        return std::make_tuple(L, s, U);
    }

    // batch_potrf

    // batch_potrs
} // namespace torch_cusolver


// generate wrappers
// FIXME do not use legacy preprocessor macro
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cusolver_batch_eigh", &torch_cusolver::batch_symmetric_eigenvalue_solve,
          "cusolver based batched eigh implementation");
    m.def("cusolver_generalized_eigh", &torch_cusolver::generalized_symmetric_eigenvalue_solve,
          "cusolver based generalized eigh implementation");
}
