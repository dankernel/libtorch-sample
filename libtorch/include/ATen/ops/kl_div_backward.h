#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/kl_div_backward_ops.h>

namespace at {


// aten::kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor
TORCH_API inline at::Tensor kl_div_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction=at::Reduction::Mean, bool log_target=false) {
    return at::_ops::kl_div_backward::call(grad_output, self, target, reduction, log_target);
}

}
