"""Opaque primitives: an IR App to an opaque unit executes the legacy code itself
(through the harness), so the surrounding lifted structure can still be
replayed bitwise.  Its *contract* is checked separately."""
from .probe import _ev


class Native:
    def __init__(self, lib, name, spec):
        self.lib, self.name, self.spec = lib, name, spec
        u = lib.units[name]
        self.u = u
        self.params = [a for a in u.args if u.var(a).type != 'INTEGER']
        self.outs = {'__result__': None} if u.kind == 'FUNCTION' else {}

    def native(self, vals, env):
        args, it = {}, iter(vals)
        e = {k: env[k] for k in ('N', 'NX', 'NY') if k in env}
        for a in self.u.args:
            if self.u.var(a).type == 'INTEGER':
                args[a] = _ev(self.spec['int_args'][a], e)
            else:
                args[a] = next(it)
        outs, oob = self.lib.call(self.name, args)
        if oob:
            raise RuntimeError(f'{self.name}: out-of-bounds access {oob}')
        return outs['__result__'] if self.u.kind == 'FUNCTION' else outs[self.spec['out']]
