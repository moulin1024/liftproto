"""Dynamic harness: build a legacy F77 file as a shared library and call any
unit from Python with arbitrary inputs.  COMMON blocks are exposed as numpy
views (storage association, no padding).  Array arguments are surrounded by
NaN canaries so out-of-bounds reads/writes are detected."""
import ctypes, hashlib, os, subprocess
import numpy as np
from . import f77

NP = {'INTEGER': np.int32, 'DOUBLE PRECISION': np.float64, 'REAL': np.float32, 'LOGICAL': np.int32}
CT = {'INTEGER': ctypes.c_int32, 'DOUBLE PRECISION': ctypes.c_double, 'REAL': ctypes.c_float,
      'LOGICAL': ctypes.c_int32}
PAD = 64


def _canary(arg_index, n):
    """Tagged quiet-NaN payloads, unique per (argument, side, slot): copying one
    argument's canary into another's pad is therefore still detected."""
    side = np.repeat(np.array([0, 1], dtype=np.uint64), PAD)
    slot = np.tile(np.arange(PAD, dtype=np.uint64), 2)
    return (np.uint64(0x7ff8_de00_0000_0000) | (np.uint64(arg_index) << np.uint64(32))
            | (side << np.uint64(16)) | slot)

BASE_FLAGS = ['-std=legacy', '-fPIC', '-shared', '-ffp-contract=off', '-fno-fast-math']


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def build(src, out, opt='-O2', extra=()):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    src, out = os.path.abspath(src), os.path.abspath(out)
    cmd = ['gfortran', *BASE_FLAGS, opt, *extra, src, '-o', out]
    subprocess.run(cmd, check=True, capture_output=True,
                   cwd=os.path.dirname(os.path.abspath(out)))
    return {'cmd': ' '.join(cmd[:-3] + [os.path.basename(src)]), 'src_sha256': sha256(src)}


def build_cov(src, outdir):
    """-O0 + gcov instrumentation; object compiled separately so gcno/gcda are
    named after the source."""
    os.makedirs(outdir, exist_ok=True)
    src = os.path.abspath(src)
    base = os.path.splitext(os.path.basename(src))[0]
    subprocess.run(['gfortran', '-std=legacy', '-fPIC', '-O0', '-ffp-contract=off', '--coverage',
                    '-c', src, '-o', f'{base}.o'], check=True, cwd=outdir, capture_output=True)
    subprocess.run(['gfortran', '-shared', '--coverage', f'{base}.o', '-o', f'lib{base}_cov.so'],
                   check=True, cwd=outdir, capture_output=True)
    for f in os.listdir(outdir):
        if f.endswith('.gcda'):
            os.remove(os.path.join(outdir, f))
    return os.path.join(outdir, f'lib{base}_cov.so')


def gcov_partial(src, outdir, units):
    """Lines with an untaken branch or never executed, attributed to units."""
    src = os.path.abspath(src)
    subprocess.run(['gcov', '-b', '-c', '-o', '.', src], check=True, cwd=outdir, capture_output=True)
    gfile = os.path.join(outdir, os.path.basename(src) + '.gcov')
    res, cur = {}, None
    for line in open(gfile):
        parts = line.split(':', 2)
        if len(parts) == 3 and parts[1].strip().isdigit():
            cnt, ln, text = parts[0].strip(), int(parts[1]), parts[2].rstrip('\n')
            cur = (ln, text)
            if cnt == '#####':
                res.setdefault(cur, set()).add('never executed')
        elif line.startswith('branch') and cur is not None:
            if 'never executed' in line or 'taken 0' in line.split('(')[0]:
                if line.split()[3:4] == ['0'] or 'never executed' in line:
                    res.setdefault(cur, set()).add('branch not taken')
    out = {}
    for (ln, text), why in sorted(res.items()):
        for u in units.values():
            if u.line_start <= ln <= u.line_end and u.kind != 'PROGRAM':
                out.setdefault(u.name, []).append({'line': ln, 'src': text.strip(), 'why': sorted(why)})
    return out


class FLib:
    def __init__(self, so, src):
        self.so, self.src = so, src
        self.lib = ctypes.CDLL(os.path.abspath(so))
        self.units = f77.parse(src)
        self._commons = {}
        for u in self.units.values():
            for blk, names in u.commons.items():
                if blk not in self._commons:
                    self._commons[blk] = self._map_common(u, blk, names)

    # ---- COMMON blocks -------------------------------------------------
    def _map_common(self, u, blk, names):
        layout, off = [], 0
        for n in names:
            v = u.var(n)
            dt = np.dtype(NP[v.type])
            shape = tuple(u.eval_dim(d) for d in v.dims)
            layout.append((n, dt, shape, off))
            off += dt.itemsize * int(np.prod(shape, dtype=int))
        raw = (ctypes.c_char * off).in_dll(self.lib, blk.lower() + '_')
        buf = np.frombuffer(raw, dtype=np.uint8)
        views = {n: np.ndarray(shape, dt, buffer=buf, offset=o) for n, dt, shape, o in layout}
        return {'raw': buf, 'views': views, 'layout': layout}

    def common(self, blk):
        return self._commons[blk]['views']

    def common_items(self):
        for blk, c in self._commons.items():
            for n, dt, shape, _ in c['layout']:
                yield blk, n, dt, shape

    def snapshot(self):
        return {b: c['raw'].copy() for b, c in self._commons.items()}

    def restore(self, snap):
        for b, raw in snap.items():
            self._commons[b]['raw'][:] = raw

    # ---- calling ---------------------------------------------------------
    def call(self, uname, args, check_reads=False):
        """With check_reads, also detect out-of-bounds READS: the call is repeated with
        the pads of one array at a time filled with finite decoys; any change of any
        output means the routine read past the declared length (differential canaries)."""
        if not check_reads:
            return self._call(uname, args)
        snap = self.snapshot()
        u = self.units[uname]
        alt = {}
        for k, a in enumerate(x for x in u.args if u.var(x).dims):
            self.restore(snap)
            alt[a] = self._call(uname, args, decoy=a)[0]
        self.restore(snap)
        outs, oob = self._call(uname, args)
        for a, o in alt.items():
            for key in outs:
                x = np.atleast_1d(np.asarray(outs[key], dtype=np.float64))
                y = np.atleast_1d(np.asarray(o[key], dtype=np.float64))
                if x.shape != y.shape or not np.array_equal(x.view(np.uint64), y.view(np.uint64)):
                    oob.append(f'{a} (read past end)'); break
        return outs, oob

    def _call(self, uname, args, decoy=None):
        """args: name -> scalar or 1-D numpy array (copied).  Returns
        (outs, oob) where outs maps every argument name (post-call value) and
        '__result__' for functions; oob lists arrays whose canaries changed."""
        u = self.units[uname]
        fn = getattr(self.lib, uname.lower() + '_')
        cargs, keep = [], {}
        for a in u.args:
            v = u.var(a)
            if v.dims:
                arr = np.asarray(args[a], dtype=NP[v.type])
                buf = np.empty(arr.size + 2 * PAD, dtype=NP[v.type])
                pat = _canary(len(keep), arr.size)          # distinct per argument and side
                buf.view(np.uint64)[:PAD], buf.view(np.uint64)[PAD + arr.size:] = pat[:PAD], pat[PAD:]
                if decoy == a:
                    buf[:PAD] = 1.25e7; buf[PAD + arr.size:] = -3.5e7
                    pat = buf.view(np.uint64)[np.r_[0:PAD, PAD + arr.size:2 * PAD + arr.size]].copy()
                buf[PAD:PAD + arr.size] = arr
                keep[a] = ('arr', buf, arr.size, pat)
                cargs.append(ctypes.c_void_p(buf.ctypes.data + PAD * buf.itemsize))
            else:
                py = int if v.type in ('INTEGER', 'LOGICAL') else float
                c = CT[v.type](py(args[a]))
                keep[a] = ('sc', c, None)
                cargs.append(ctypes.byref(c))
        if u.kind == 'FUNCTION':
            fn.restype = CT[u.var(u.name).type]
        else:
            fn.restype = None
        res = fn(*cargs)
        outs, oob = {}, []
        for a, (k, obj, n, *pat) in keep.items():
            if k == 'arr':
                outs[a] = obj[PAD:PAD + n].copy()
                pads = np.concatenate([obj[:PAD], obj[PAD + n:]])
                if not np.array_equal(pads.view(np.uint64), pat[0]):
                    oob.append(a)
            else:
                outs[a] = obj.value
        if u.kind == 'FUNCTION':
            outs['__result__'] = res
        return outs, oob
