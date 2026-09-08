"""Validate topology_engine.so against the Python yamada.py reference."""
import ctypes, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.environ.get('TOPOLOGY_ENGINE_SO', os.path.join(HERE, 'pipeline', 'topology_engine.so')))
lib.compute_yamada.restype = ctypes.c_int
lib.compute_yamada.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.c_double, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
for name in ('get_poly_size', 'get_offset', 'get_max_crossings', 'get_overflow_flag', 'has_openmp'):
    getattr(lib, name).restype = ctypes.c_int
for name in ('get_last_states', 'get_last_hits', 'get_last_misses'):
    getattr(lib, name).restype = ctypes.c_longlong
POLY_SIZE, OFFSET = lib.get_poly_size(), lib.get_offset()

sys.path.insert(0, '/mnt/altnas/work/dimos/Yamada')
os.chdir('/mnt/altnas/work/dimos/Yamada')
from yamada import yamada, normalize, p_str


def c_yamada(strands, tol=1e-6, tries=60, seed=42, max_crossings=20):
    pts = np.ascontiguousarray(np.vstack(strands), dtype=np.float64)
    lens = np.array([len(s) for s in strands], dtype=np.int32)
    raw = (ctypes.c_int * POLY_SIZE)()
    out = (ctypes.c_int * POLY_SIZE)()
    nx, nv = ctypes.c_int(0), ctypes.c_int(0)
    t0 = time.time()
    ret = lib.compute_yamada(pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                             lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                             len(strands), tol, tries, seed, max_crossings, raw, out,
                             ctypes.byref(nx), ctypes.byref(nv))
    dt = time.time() - t0
    poly = {i - OFFSET: out[i] for i in range(POLY_SIZE) if out[i]}
    return ret, nx.value, poly, dt


def theta(n, fy2, fz2, fy3, fz3):
    t = np.linspace(0, 1, n)
    s1 = np.column_stack([t, np.zeros(n), np.zeros(n)])
    s2 = np.column_stack([t, fy2(t), fz2(t)])
    s3 = np.column_stack([t, fy3(t), fz3(t)])
    for s in (s2, s3):
        s[0] = s1[0]; s[-1] = s1[-1]
    return [s1, s2, s3]


cases = {
    'planar':   theta(10, lambda t: 0.5*np.sin(np.pi*t), lambda t: 0.3*np.sin(np.pi*t),
                          lambda t: -0.5*np.sin(np.pi*t), lambda t: -0.3*np.sin(np.pi*t)),
    '5-cross':  theta(20, lambda t: 0.3*np.sin(3*np.pi*t), lambda t: 0.2*np.cos(2*np.pi*t),
                          lambda t: -0.3*np.sin(2*np.pi*t), lambda t: 0.2*np.sin(3*np.pi*t)),
    'medium':   theta(30, lambda t: 0.2*np.sin(5*np.pi*t), lambda t: 0.15*np.cos(3*np.pi*t),
                          lambda t: -0.2*np.sin(4*np.pi*t), lambda t: 0.15*np.sin(5*np.pi*t)),
}

print(f'OpenMP: {lib.has_openmp()}   OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS", "(default)")}')
print(f'{"case":<10}{"py_nx":>6}{"c_nx":>5}{"py_s":>9}{"c_s":>9}{"speedup":>9}{"hit%":>7}{"match"}')
for name, strands in cases.items():
    t0 = time.time()
    py_poly, info = yamada(strands, tol=1e-6, tries=60, seed=42, return_info=True)
    t_py = time.time() - t0
    py_norm = dict(normalize(py_poly))
    ret, nx, c_poly, t_c = c_yamada(strands)
    hits, misses = lib.get_last_hits(), lib.get_last_misses()
    hit = 100.0 * hits / max(1, hits + misses)
    ok = (c_poly == py_norm) and ret >= 0
    print(f'{name:<10}{info["crossings"]:>6}{nx:>5}{t_py:>9.3f}{t_c:>9.4f}{t_py/max(t_c,1e-6):>8.0f}x{hit:>6.1f}%  {"OK" if ok else "MISMATCH"}')
    if not ok:
        print(f'   py: {p_str(py_norm)}')
        print(f'   c : {c_poly}')
    if lib.get_overflow_flag():
        print('   WARNING: exponent overflow flag set')

ret, nx, _, _ = c_yamada(cases['5-cross'], max_crossings=3)
print(f'\ncap test (max_crossings=3 on 5-cross): ret={ret}  {"OK" if ret == -2 else "UNEXPECTED"}')
