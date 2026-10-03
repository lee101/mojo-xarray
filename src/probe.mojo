from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime V = SIMD[DType.float64, W]


def _nz(v: V) -> V:
    return min(v, 0.0) + max(v, 0.0)


@export("mx_probe_select")
def probe_select(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=src_addr)
    var dst = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=dst_addr)
    var k = 0
    while k + W <= n:
        var v = src.unsafe_load[width=W](k)
        var c = _nz(v)
        var m = V(1.0)
        if _nz(V(v == v)).reduce_add() > 0.0:
            m = V(0.0)
        dst.unsafe_store(k, c * m + v * (1.0 - m))
        k += W
    return


@export("mx_probe_ops")
def probe_ops(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=src_addr)
    var dst = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=dst_addr)
    var k = 0
    while k + W <= n:
        var v = src.unsafe_load[width=W](k)
        var e = SIMD[DType.int64, W](v == v)
        var f = SIMD[DType.float64, W](e)
        dst.unsafe_store(k, f.reduce_add())
        k += W
    return


@export("mx_probe_copy")
def probe_copy(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=src_addr)
    var dst = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=dst_addr)
    var k = 0
    while k + W <= n:
        dst.unsafe_store(k, src.unsafe_load[width=W](k))
        k += W
    return


@export("mx_probe_lane")
def probe_lane(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=src_addr)
    var dst = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=dst_addr)
    var k = 0
    while k + W <= n:
        var v = src.unsafe_load[width=W](k)
        dst.unsafe_store(k, _nz(v))
        k += W
    return
