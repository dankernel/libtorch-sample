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



#include <ATen/ops/_thnn_differentiable_gru_cell_backward_ops.h>

namespace at {


// aten::_thnn_differentiable_gru_cell_backward(Tensor grad_hy, Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias, Tensor? hidden_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
TORCH_API inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_differentiable_gru_cell_backward(const at::Tensor & grad_hy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
    return at::_ops::_thnn_differentiable_gru_cell_backward::call(grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

}
