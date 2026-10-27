"""Integer polynomials over named symbols (index arithmetic), with z3 export."""
import z3


class Poly:
    __slots__ = ('t',)

    def __init__(self, t=None):
        self.t = {k: v for k, v in (t or {}).items() if v != 0}

    @staticmethod
    def const(c): return Poly({(): int(c)})
    @staticmethod
    def sym(s): return Poly({(s,): 1})

    def __add__(a, b):
        b = _P(b); t = dict(a.t)
        for k, v in b.t.items(): t[k] = t.get(k, 0) + v
        return Poly(t)
    __radd__ = __add__
    def __neg__(a): return Poly({k: -v for k, v in a.t.items()})
    def __sub__(a, b): return a + (-_P(b))
    def __rsub__(a, b): return _P(b) - a

    def __mul__(a, b):
        b = _P(b); t = {}
        for k1, v1 in a.t.items():
            for k2, v2 in b.t.items():
                k = tuple(sorted(k1 + k2)); t[k] = t.get(k, 0) + v1 * v2
        return Poly(t)
    __rmul__ = __mul__

    def __eq__(a, b): return isinstance(b, (Poly, int)) and (a - _P(b)).t == {}
    def __hash__(self): return hash(tuple(sorted(self.t.items())))

    def is_const(self): return all(k == () for k in self.t)
    def cval(self): return self.t.get((), 0)
    def syms(self): return {s for k in self.t for s in k}

    def coeff(self, mono): return self.t.get(tuple(sorted(mono)), 0)

    def subs(self, m):
        r = Poly()
        for k, v in self.t.items():
            term = Poly.const(v)
            for s in k:
                term = term * (m[s] if s in m else Poly.sym(s))
            r = r + term
        return r

    def __str__(self):
        if not self.t: return '0'
        parts = []
        for k, v in sorted(self.t.items(), key=lambda kv: (-len(kv[0]), kv[0])):
            mono = '*'.join(k)
            if not k: s = str(abs(v))
            elif abs(v) == 1: s = mono
            else: s = f'{abs(v)}*{mono}'
            parts.append(('-' if v < 0 else '+', s))
        out = ('-' if parts[0][0] == '-' else '') + parts[0][1]
        for sg, s in parts[1:]:
            out += f' {sg} {s}'
        return out
    __repr__ = __str__

    def z3(self, env):
        r = z3.IntVal(0)
        for k, v in self.t.items():
            term = z3.IntVal(v)
            for s in k:
                if s not in env: env[s] = z3.Int(s)
                term = term * env[s]
            r = r + term
        return r


def _P(x): return x if isinstance(x, Poly) else Poly.const(x)
