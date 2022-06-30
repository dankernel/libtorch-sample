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



#include <ATen/ops/_test_ambiguous_defaults_ops.h>

namespace at {


// aten::_test_ambiguous_defaults.a(Tensor dummy, int a=1, int b=1) -> Tensor
TORCH_API inline at::Tensor _test_ambiguous_defaults(const at::Tensor & dummy, int64_t a=1, int64_t b=1) {
    return at::_ops::_test_ambiguous_defaults_a::call(dummy, a, b);
}

// aten::_test_ambiguous_defaults.b(Tensor dummy, int a=2, str b="2") -> Tensor
TORCH_API inline at::Tensor _test_ambiguous_defaults(const at::Tensor & dummy, int64_t a, c10::string_view b) {
    return at::_ops::_test_ambiguous_defaults_b::call(dummy, a, b);
}

}
