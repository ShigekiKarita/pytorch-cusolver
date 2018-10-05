#include <torch/torch.h>

#include <assert.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include <ATen/CUDAStream.h>
// #include <helper_cuda.h>
// #include <helper_cusolver.h>



at::Tensor d_sigmoid(at::Tensor z) {
    auto s = at::sigmoid(z);
    return (1 - s) * s;
}


// a is (batch, m, m)
// see also https://docs.nvidia.com/cuda/cusolver/index.html#batchsyevj-example1
std::tuple<at::Tensor, at::Tensor> cusolver_batch_eigh(at::Tensor a, bool in_place=false, bool use_lower=true,
                               double tol=1e-7, int max_sweeps=15, bool sort_eig=false) {
    auto device_id = a.get_device(); // at::current_device(); // a.device().index();
    auto w = at::empty({a.size(0), a.size(2)}, a.type());


    // TODO use singleton handler instead of ondemand handle
    // TODO check cutorch or ATen does not handle cusolver
    // https://github.com/torch/cutorch/blob/master/lib/THC/THCGeneral.h.in
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAContext.h
    cusolverDnHandle_t handle;
    cusolverStatus_t status;
    status = cusolverDnCreate(&handle);
    assert(status == CUSOLVER_STATUS_SUCCESS);
    // TODO use non blocking stream
    cudaStream_t stream = at::globalContext().getCurrentCUDAStreamOnDevice(device_id);
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    ////////// create param /////////
    syevjInfo_t syevj_params;
    status = cusolverDnCreateSyevjInfo(&syevj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    /* default value of tolerance is machine zero */
    status = cusolverDnXsyevjSetTolerance(
        syevj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    /* default value of max. sweeps is 100 */
    status = cusolverDnXsyevjSetMaxSweeps(
        syevj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    /* disable sorting */
    status = cusolverDnXsyevjSetSortEig(
        syevj_params,
        sort_eig);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    auto V = in_place ? a.contiguous() : a.clone();
    auto d_V = V.data<float>();
    auto batch_size = a.size(0);
    auto m = a.size(2);
    auto lda = a.size(2);
    auto d_W = w.data<float>();
    auto uplo = use_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    int lwork;

    //// query working space of syevjBatched ////
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

    if (d_info ) { cudaFree(d_info); }
    if (d_work ) { cudaFree(d_work); }
    if (handle) { cusolverDnDestroy(handle); }
    return std::make_tuple(w, V);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test", &d_sigmoid, "pybind11 test");
    m.def("cusolver_batch_eigh", &cusolver_batch_eigh, "cusolver based batched eigh");
}
