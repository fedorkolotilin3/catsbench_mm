import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def exp(x):
    if x.dtype == tl.float64:
        return libdevice.exp(x)
    else:
        return tl.exp(x)

@triton.jit
def exp2(x):
    if x.dtype == tl.float64:
        return libdevice.exp2(x)
    else:
        return tl.exp2(x)

@triton.jit
def log(x):
    if x.dtype == tl.float64:
        return libdevice.log(x)
    else:
        return tl.log(x)

@triton.jit
def log2(x):
    if x.dtype == tl.float64:
        return libdevice.log2(x)
    else:
        return tl.log2(x)
