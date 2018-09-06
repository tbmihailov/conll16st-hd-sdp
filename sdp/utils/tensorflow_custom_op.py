import tensorflow as tf


from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs


def tensor_to_filter_3(elems, dtype=None, parallel_iterations=10, back_prop=True,
           swap_memory=False, name=None):


    """map on the list of tensors unpacked from `elems` on dimension 0.
This map operator repeatedly applies the callable `fn` to a sequence of
elements from first to last. The elements are made of the tensors unpacked
from `elems`. `dtype` is the data type of the return value of `fn`. Users
must provide `dtype` if it is different from the data type of `elems`.
Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
of the result tensor is `[len(values)] + fn(values[0]).shape`.
Args:
  fn: The callable to be performed.
  elems: A tensor to be unpacked to apply `fn`.
  dtype: (optional) The output type of `fn`.
  parallel_iterations: (optional) The number of iterations allowed to run
                       in parallel.
  back_prop: (optional) True enables back propagation.
  swap_memory: (optional) True enables GPU-CPU memory swapping.
  name: (optional) Name prefix for the returned tensors.
Returns:
  A tensor that packs the results of applying `fn` to the list of tensors
  unpacked from `elems`, from first to last.
Raises:
  TypeError: if `fn` is not callable.
Example:
  ```python
  elems = [1, 2, 3, 4, 5, 6]
  squares = map_fn(lambda x: x * x, elems)
  # squares == [1, 4, 9, 16, 25, 36]
  ```
"""
    filter_size = 3
    with vs.variable_op_scope([elems], name, "tensor_to_filter") as varscope:
        # Any get_variable calls fn will cache the first call locally
        # and not issue repeated network I/O requests for each iteration.
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        dtype = dtype if dtype else elems.dtype

        # Convert elems to tensor array.
        n = array_ops.shape(elems)[0]
        elems_ta = tensor_array_ops.TensorArray(dtype=dtype, size=n-filter_size,
                                                dynamic_size=False)

        elems_ta = elems_ta.unpack(elems)

        i = constant_op.constant(0)
        acc_ta = tensor_array_ops.TensorArray(dtype=dtype, size=n-3,
                                              dynamic_size=False)


        def compute(i, ta):
            ta = ta.write(i, elems_ta.read(i)*3)

            return [i + 1, ta]


        _, r_a = control_flow_ops.while_loop(
            lambda i, a: i < n - filter_size, compute, [i, acc_ta])
        result = r_a.pack()
        elems_dims = ops.convert_to_tensor(elems).get_shape().dims
        result_dims = result.get_shape().dims
        # if elems_dims and result_dims:
        #     result.set_shape([elems_dims[0]] + result_dims[1:])
        return result




