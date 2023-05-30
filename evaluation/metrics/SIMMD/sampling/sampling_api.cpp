#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "sampling_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_points_wrapper", &gather_points_wrapper_fast, "gather_points_wrapper_fast");
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast, "gather_points_grad_wrapper_fast");

    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
}
