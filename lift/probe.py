"""Deterministic dynamic probes: write/read sets, polynomial degree via
finite differences (Schwartz-Zippel style random directions), exact
homogeneity via power-of-two scaling, and element-level dependency relations."""
import numpy as np

EPS = np.finfo(np.float64).eps


def _ev(expr, env):
    return int(eval(str(expr), {}, dict(env)))


class Ctx:
    """A probing context for one unit at one problem size.

    spec (agent-proposed, *untrusted*; canaries check the array lengths):
      arg_len    : arg -> length expression in N, NX, NY
      int_args   : integer arg -> value expression
      common_int : 'BLK.NAME' -> value expression
      common_len : 'BLK.NAME' -> active length expression (float arrays)
      setup      : optional (unit, {arg: expr}) used to fill COMMON instead of random
    """

    def __init__(self, lib, uname, spec, nx, ny):
        self.lib, self.u, self.spec = lib, lib.units[uname], spec
        self.env = {'NX': nx, 'NY': ny, 'N': nx * ny}
        self.N = nx * ny
        self.float_args, self.float_commons = [], []
        for a in self.u.args:
            v = self.u.var(a)
            if v.type != 'INTEGER':
                self.float_args.append(a)
        for blk, n, dt, shape in lib.common_items():
            key = f'{blk}.{n}'
            if dt == np.float64 and blk in self.u.commons and not spec.get('setup'):
                if shape and key not in spec.get('common_len', {}):
                    continue                     # COMMON array this unit does not read
                self.float_commons.append(key)

    # ---- inputs -------------------------------------------------------------
    def shape_of(self, key):
        if '.' in key:
            ln = self.spec.get('common_len', {}).get(key)
            return () if ln is None else (_ev(ln, self.env),)
        if self.u.var(key).dims:
            return (_ev(self.spec['arg_len'][key], self.env),)
        return ()

    def keys(self):
        return self.float_args + self.float_commons

    def draw(self, rng, key=None):
        def one(k):
            sh = self.shape_of(k)
            return rng.choice([-1.0, 1.0], size=sh) * rng.uniform(1.0, 2.0, size=sh) if sh else \
                float(rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 2.0))
        if key is not None:
            return one(key)
        d = {k: one(k) for k in self.keys()}
        d.update(self.spec.get('fixed', {}))
        return d

    # ---- execution ------------------------------------------------------------
    def run(self, inputs, check_reads=False):
        lib = self.lib
        for blk in lib._commons:
            lib._commons[blk]['raw'][:] = 0
        for key, expr in self.spec.get('common_int', {}).items():
            blk, n = key.split('.')
            lib.common(blk)[n][...] = _ev(expr, self.env)
        if self.spec.get('setup'):
            su, sargs = self.spec['setup']
            lib.call(su, {a: _ev(e, self.env) for a, e in sargs.items()})
        for key in self.float_commons:
            blk, n = key.split('.')
            v = inputs[key]
            if np.ndim(v):
                lib.common(blk)[n][:len(v)] = v
            else:
                lib.common(blk)[n][...] = v
        args = {}
        for a in self.u.args:
            if a in inputs:
                args[a] = inputs[a]
            elif self.u.var(a).type == 'INTEGER':
                args[a] = _ev(self.spec['int_args'][a], self.env)
            else:
                raise KeyError(a)
        outs, oob = lib.call(self.u.name, args, check_reads=check_reads)
        for key in self.float_commons:
            blk, n = key.split('.')
            c = lib.common(blk)[n]
            outs[key] = c[:self.shape_of(key)[0]].copy() if self.shape_of(key) else float(c)
        for key in self.spec.get('common_out', []):
            blk, n = key.split('.')
            c = lib.common(blk)[n]
            outs[n] = c[:self.N].copy() if c.ndim else c.item()
        return outs, oob

    # ---- probes -----------------------------------------------------------------
    def write_set(self, rng):
        inp = self.draw(rng)
        outs, oob = self.run(inp)
        w = []
        for k, v in outs.items():
            if k == '__result__':
                w.append(k); continue
            before = inp.get(k)
            if before is None:          # integer arg
                if v != _ev(self.spec['int_args'][k], self.env):
                    w.append(k)
            elif not _beq(before, v):
                w.append(k)
        return w, oob

    def F(self, inputs, outkeys):
        outs, _ = self.run(inputs)
        return np.concatenate([np.atleast_1d(np.asarray(outs[k], dtype=np.float64)) for k in outkeys])

    def read_set(self, rng, outkeys):
        base = self.draw(rng)
        f0 = self.F(base, outkeys)
        r = []
        for k in self.keys():
            x = dict(base); x[k] = self.draw(rng, k)
            if not _beq(self.F(x, outkeys), f0):
                r.append(k)
        return r

    def degree(self, rng, key, outkeys, dmax=3, trials=3):
        """Smallest d such that every (d+1)-th mixed finite difference along random
        directions in `key` vanishes (to roundoff); None if > dmax."""
        for d in range(dmax + 1):
            ok = True
            for _ in range(trials):
                base = self.draw(rng)
                hs = [self.draw(rng, key) for _ in range(d + 1)]
                acc, scale = 0.0, 0.0
                for mask in range(1 << (d + 1)):
                    x = dict(base)
                    x[key] = base[key] + sum((hs[i] for i in range(d + 1) if mask >> i & 1),
                                             start=np.zeros_like(np.asarray(base[key])))
                    f = self.F(x, outkeys)
                    sgn = (-1) ** (d + 1 - bin(mask).count('1'))
                    acc = acc + sgn * f
                    scale = max(scale, np.max(np.abs(f)))
                if np.max(np.abs(acc)) > 1e-11 * max(scale, 1e-300):
                    ok = False; break
            if ok:
                return d
        return None

    def homogeneity(self, rng, key, outkeys, kmax=3):
        """Exact (bitwise) homogeneity exponent k with F(2v) = 2^k F(v), or None.
        Scaling by powers of two is exact in IEEE arithmetic, so this is a
        bitwise test, not a tolerance test."""
        base = self.draw(rng)
        f0 = self.F(base, outkeys)
        x = dict(base); x[key] = 2.0 * np.asarray(base[key]) if np.ndim(base[key]) else 2.0 * base[key]
        f1 = self.F(x, outkeys)
        for k in range(-kmax, kmax + 1):
            if _beq(f1, f0 * 2.0 ** k):
                return k
        return None

    def dependency(self, rng, kin, kout):
        base = self.draw(rng)
        f0 = np.atleast_1d(np.asarray(self.run(base)[0][kout], dtype=np.float64))
        n_in = self.shape_of(kin)[0]
        R = np.zeros((f0.size, n_in), dtype=bool)
        for j in range(n_in):
            x = dict(base); x[kin] = np.array(base[kin]); x[kin][j] += 1.0
            f = np.atleast_1d(np.asarray(self.run(x)[0][kout], dtype=np.float64))
            R[:, j] = f.view(np.uint64) != f0.view(np.uint64)
        return R

    def jacobian_exact(self, kin, kout, others):
        """For an input in which the map is linear: column j = F(e_j) with all
        other inputs fixed.  With a zero base this is exact in floating point
        (products with 0 and additions of 0 are exact)."""
        n_in = self.shape_of(kin)[0]
        cols = []
        for j in range(n_in):
            x = dict(others); x[kin] = np.zeros(n_in); x[kin][j] = 1.0
            cols.append(np.atleast_1d(np.asarray(self.run(x)[0][kout], dtype=np.float64)))
        return np.stack(cols, axis=1)


def _beq(a, b):
    a = np.atleast_1d(np.asarray(a, dtype=np.float64)); b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))
