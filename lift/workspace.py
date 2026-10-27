"""Workspace: the persistent state of one lifting job, the evidence-carrying fact
store, the worklist of open obligations, and the agent-facing operations.

Trust model
  * static analysis, dynamic probes, proofs and the fact store are TRUSTED code;
  * everything an LLM agent submits (specs, terms, primitive contracts, templates,
    probes, annotations) is UNTRUSTED data: it is parsed by the DSL / validated by
    regexes, never executed, and only enters the structure after the checker
    has produced evidence for it.
"""
import hashlib, json, os, re, subprocess, sys, time
import numpy as np

from . import harness, ir, verify, dsl
from .probe import Ctx, _ev
from .native import Native
from .static.analysis import analyze
from .static import proofs
from .static.lifter import key as tkey, nodes

SPEC_RE = re.compile(r'^[0-9NXY+\-*() ]{1,40}$')
VISIBLE = [(7, 5), (6, 8)]
EVIDENCE_RANK = {'static-proof': 3, 'dynamic-bitwise': 2, 'dynamic-test': 2, 'static-structural': 2,
                 'agent-claim': 0}


class AgentError(ValueError):
    pass


def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (set, tuple)): return list(o)
    return str(o)


class Workspace:
    # ------------------------------------------------------------ persistence
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.state = json.load(open(os.path.join(self.path, 'state.json')))
        self._lib = None

    def save(self):
        tmp = os.path.join(self.path, 'state.json.tmp')
        json.dump(self.state, open(tmp, 'w'), indent=1, default=_j)
        os.replace(tmp, os.path.join(self.path, 'state.json'))

    def log(self, actor, tool, args, result):
        with open(os.path.join(self.path, 'log.jsonl'), 'a') as f:
            f.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'actor': actor, 'tool': tool,
                                'args': args, 'result': result}, default=_j) + '\n')

    @property
    def lib(self):
        if self._lib is None:
            self._lib = harness.FLib(self.state['lib'], self.state['src'])
        return self._lib

    def sizes(self):
        r = np.random.default_rng(self.state['secret_seed'])
        return VISIBLE + [tuple(int(v) for v in r.integers(9, 17, size=2))]

    def unit_names(self):
        return list(self.state['units'])

    # --------------------------------------------------------------- creation
    @classmethod
    def create(cls, path, src, main=None, secret_seed=None):
        os.makedirs(path, exist_ok=True)
        path = os.path.abspath(path)
        src, main = os.path.abspath(src), (os.path.abspath(main) if main else None)
        so = os.path.join(path, 'build', 'lib.so')
        binfo = harness.build(src, so)
        A = analyze(src, main)
        st = {'src': src, 'main': main, 'lib': so, 'build': binfo,
              'secret_seed': int(secret_seed if secret_seed is not None else int.from_bytes(os.urandom(4), 'little')),
              'units': {}, 'proposals': [], 'annotations': [], 'templates': {}, 'targeted': {},
              'static': {'order': A['order'], 'callgraph': A['callgraph'], 'contracts': A['contracts'],
                         'branches': A['branches'], 'loops': A['loops'], 'proof_log': A['prover'].log,
                         'main_calls': A['main_calls']}}
        for n in A['order']:
            s = A['summaries'][n]
            st['units'][n] = {
                'kind': s.kind,
                'static': {'kind': s.kind, 'term': ir.pp_def(s.term) if s.term else None,
                           'reason': s.reason, 'reason_specialized': getattr(s, 'reason_specialized', None),
                           'specialized': getattr(s, 'specialized', None),
                           'footprint': {k: str(v) for k, v in s.footprint.items()},
                           'reads_common': s.reads_common, 'writes_common': s.writes_common},
                'spec': A['specs'][n], 'spec_source': 'static',
                'term': ir.pp_def(s.term) if s.term else None, 'term_source': 'static' if s.term else None,
                'primitive': None,
                'lines': [A['units'][n].line_start, A['units'][n].line_end]}
        open(os.path.join(path, 'log.jsonl'), 'w').close()
        json.dump(st, open(os.path.join(path, 'state.json'), 'w'), indent=1, default=_j)
        ws = cls(path)
        ws.static_proofs()
        ws.refresh()
        ws.log('system', 'create', {'src': src, 'main': main}, ws.status())
        return ws

    # ---------------------------------------------------------------- helpers
    def fn_names(self, extra=()):
        return list(self.state['units']) + list(extra)

    def term(self, n):
        t = self.state['units'][n]['term']
        return dsl.parse_def(t, self.fn_names()) if t else None

    def spec(self, n):
        return self.state['units'][n]['spec']

    def spec_complete(self, n):
        sp = self.spec(n)
        return all(v is not None for v in sp['arg_len'].values())

    def fns(self):
        out = {}
        for n, u in self.state['units'].items():
            if u['term']:
                out[n] = self.term(n)
            elif self.spec_complete(n):
                out[n] = Native(self.lib, n, self.spec(n))     # opaque/primitive: run legacy code
        return out

    # ----------------------------------------------------------- static proofs
    def static_proofs(self):
        S = self.state['units']
        res = {'operators': {}, 'preconditioners': {}, 'spd': {}, 'degree': {}}
        for n, u in S.items():
            if u['kind'] != 'kernel' or not u['term']: continue
            d = self.term(n)
            for p in d.params:
                if self.lib.units[n].var(p).type == 'INTEGER': continue
                try:
                    deg, coll = proofs.degree(d, p)
                    vals = [v for k, v in deg.items() if not (self.lib.units[n].var(k).type == 'INTEGER' if k != '__result__' else False)]
                    dv = None if any(v is None for v in vals) else max(vals)
                    res['degree'].setdefault(n, {})[p] = {'degree': dv, 'per_output': deg, 'ieee_only_branches': coll}
                except Exception as ex:
                    res['degree'].setdefault(n, {})[p] = {'degree': None, 'error': str(ex)}
            units = self.lib.units
            arrays = [a for a in d.params if units[n].var(a).dims]
            for p in arrays:
                try:
                    grid = tuple(x for x in units[n].dims if x != p)
                    sym = proofs.symmetry(d, x=p, grid_arrays=grid)
                except Exception:
                    continue
                if any(s != (0, 0) for s in sym['shifts']):
                    res['operators'][n] = {'param': p, 'symmetric': sym['symmetric'], 'pairs': sym['pairs'],
                                           'diagonal': sym['diagonal'], 'diag_unconditional': sym['diag_unconditional'],
                                           'shifts': sym['shifts']}
                    self._sym_cache = getattr(self, '_sym_cache', {}); self._sym_cache[n] = sym
        for op, info in res['operators'].items():
            sym = self._sym_cache[op]
            for n, u in S.items():
                if u['kind'] != 'kernel' or not u['term'] or n == op: continue
                d = self.term(n)
                if len(d.params) == 1:
                    ok, dg = proofs.jacobi(sym, d, x=d.params[0])
                    if ok: res['preconditioners'][n] = {'operator': op, 'kind': 'Jacobi', 'diagonal': dg}
            reads = set(S[op]['static']['reads_common'])
            for n, u in S.items():
                if set(u['static']['writes_common']) >= (reads - {'NX', 'NY'}) and u['term'] and n != op:
                    try:
                        res['spd'][op] = dict(proofs.spd(sym, self.term(n)), coefficient_source=n)
                    except Exception as ex:
                        res['spd'][op] = {'SPD': False, 'error': str(ex), 'coefficient_source': n}
        self.state['static']['proofs'] = res
        self.save()

    # ------------------------------------------------------------- dynamics
    def missing_callees(self, n, fns):
        cg = self.state['static']['callgraph']
        out, todo = [], list(cg.get(n, []))
        while todo:
            c = todo.pop()
            if c not in fns: out.append(c)
            todo += cg.get(c, [])
        return sorted(set(out))

    def driver(self):
        for n, u in self.state['units'].items():
            if u['kind'] == 'orchestration': return n
        return None

    def capture_production(self):
        """Run the lifted driver once (IR, bitwise-equal to the legacy run) and record
        the arguments of every call: realistic inputs for V2 probes."""
        drv = self.driver()
        fns = self.fns()
        if not drv or drv not in fns or self.missing_callees(drv, fns): return {}
        sp = self.spec(drv)
        if not self.spec_complete(drv): return {}
        cap = {}
        nx, ny = VISIBLE[0]
        ctx = Ctx(self.lib, drv, sp, nx, ny)
        inp = ctx.draw(np.random.default_rng(5))
        inp['B'] = np.ones(nx * ny) if 'B' in inp else None
        inp = {k: v for k, v in inp.items() if v is not None}
        ctx.run(inp)
        def hook(fn, vals, env):
            d = fns[fn]
            lst = cap.setdefault(fn, [])
            if len(lst) < 6:
                lst.append({'size': [nx, ny], 'args': {p: np.array(v, copy=True) if np.ndim(v) else v
                                                       for p, v in zip(d.params, vals)},
                            'setup': sp.get('setup')})
        ir.HOOK['app'] = hook
        try:
            verify.run_ir(ctx, fns[drv], fns, inp)
        finally:
            ir.HOOK['app'] = None
        return cap

    def verify_all(self):
        fns, prod = self.fns(), self.capture_production()
        res = {}
        for k, n in enumerate(self.state['units']):
            u = self.state['units'][n]
            if not self.spec_complete(n):
                res[n] = {'status': 'NOT RUN (spec incomplete)'}; continue
            miss = self.missing_callees(n, fns)
            if u['term'] and miss:
                res[n] = {'status': f'BLOCKED (callees without term/spec: {", ".join(miss)})'}
            elif u['term']:
                res[n] = verify.ladder(self.lib, n, self.spec(n), fns[n], fns, self.sizes(), 1000 + k,
                                       production=prod.get(n))
            elif u.get('primitive'):
                res[n] = self.check_primitive(n, u['primitive'], prod.get(n))
            else:
                res[n] = {'status': 'NO TERM (opaque)'}
        return res, prod

    def dynamic_facts(self):
        out = {}
        for k, n in enumerate(self.state['units']):
            if not self.spec_complete(n): continue
            try:
                f, _ = verify.facts(self.lib, n, self.spec(n), *VISIBLE[0], np.random.default_rng(77 + k))
                out[n] = f
            except Exception as ex:
                out[n] = {'error': str(ex)}
        return out

    def operator_dynamic(self):
        out = {}
        sp_drv = self.spec(self.driver()) if self.driver() else {}
        for op, info in self.state['static']['proofs']['operators'].items():
            if not self.spec_complete(op): continue
            rec = {'sizes': [], 'symmetric_exact': True, 'diag_equals': {}, 'dominant_to_roundoff': True}
            for (nx, ny) in self.sizes():
                spec = dict(self.spec(op))
                if sp_drv.get('setup'): spec['setup'] = sp_drv['setup']
                ctx = Ctx(self.lib, op, spec, nx, ny)
                others = ctx.draw(np.random.default_rng(nx * ny))
                outk = [k for k in ctx.spec['arg_len'] if k != info['param']][0]
                A = ctx.jacobian_exact(info['param'], outk, others)
                rec['sizes'].append([nx, ny])
                rec['symmetric_exact'] &= bool(np.array_equal(A, A.T))
                g = verify.ir_globals(ctx)
                for dname in info['diagonal']:
                    if dname in g:
                        dv = np.asarray(g[dname])[:nx * ny]
                        ok = bool(np.array_equal(np.diag(A).view(np.uint64), dv.view(np.uint64)))
                        rec['diag_equals'][dname] = rec['diag_equals'].get(dname, True) and ok
                diag, off = np.diag(A), np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
                rel = (diag - off) / np.maximum(np.abs(diag), 1e-300)
                rec['dominant_to_roundoff'] &= bool(np.all(rel >= -16 * np.finfo(float).eps) and np.all(diag > 0))
            out[op] = rec
        return out

    def coverage(self):
        covdir = os.path.join(self.path, 'cov')
        r = subprocess.run([sys.executable, '-m', 'lift.covrun', self.path], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if r.returncode != 0:
            return {'error': r.stderr[-2000:]}
        gaps = harness.gcov_partial(self.state['src'], covdir, self.lib.units)
        loops = {(l['unit'], l['line']): l for l in self.state['static']['loops']}
        brs = {(b['unit'], b['line']) for b in self.state['static']['branches']}
        out = {}
        for unit, items in gaps.items():
            for it in items:
                closed = None
                lp = loops.get((unit, it['line']))
                if lp and lp['zero_trip'].startswith('infeasible') and it['src'].startswith('DO'):
                    closed = 'static proof: the zero-trip branch of this DO loop is infeasible on every path'
                elif (unit, it['line']) in brs:
                    closed = 'static proof: branch infeasible'
                else:
                    for j in self.state.get('justifications', []):
                        if j['unit'] == unit and j['line'] == it['line']:
                            closed = 'agent claim (unverified): ' + j['argument']
                it = dict(it, closed_by=closed)
                out.setdefault(unit, []).append(it)
        return out

    def refresh(self):
        ver, prod = self.verify_all()
        self.state['verification'] = ver
        self.state['dynamic'] = self.dynamic_facts()
        self.state['operator_dynamic'] = self.operator_dynamic()
        self.save()                                   # the coverage replay runs in a child process
        self.state['coverage'] = self.coverage()
        self.state['facts'] = self.build_facts()
        self.state['worklist'] = self.build_worklist()
        self.save()

    # ------------------------------------------------------------ primitives
    def check_primitive(self, n, prim, production=None):
        """contract: unit(params) ~= relation(params) within tol; properties checked exactly."""
        fns = {k: v for k, v in self.fns().items()}
        rel = dsl.parse_def(prim['relation'], self.fn_names())
        spec = self.spec(n)
        rec = {'sizes': [], 'max_rel_err': 0.0, 'n_probes': 0, 'properties': {}, 'oob': []}
        rng = np.random.default_rng(4242)
        ins = []
        for (nx, ny) in self.sizes():
            ctx = Ctx(self.lib, n, spec, nx, ny)
            rec['sizes'].append([nx, ny])
            for _ in range(8): ins.append((ctx, ctx.draw(rng)))
        for p in production or []:
            ctx = Ctx(self.lib, n, spec, *p['size'])
            d = ctx.draw(rng); d.update({k: v for k, v in p['args'].items() if k in d}); ins.append((ctx, d))
        outk = list(rel.outs)
        nonneg = True
        for ctx, inp in ins:
            fo, oob = ctx.run(inp)
            rec['oob'] += oob
            io = verify.run_ir(ctx, rel, fns, inp)
            bit, err = verify.compare(fo, io, outk)
            rec['max_rel_err'] = max(rec['max_rel_err'], err); rec['n_probes'] += 1
            nonneg &= all(np.all(np.asarray(fo[k]) >= 0) for k in outk)
        for prop in prim.get('properties', []):
            if prop == 'nonnegative':
                rec['properties'][prop] = bool(nonneg)
            m = re.fullmatch(r'homogeneous:(-?\d)', prop)
            if m:
                ok = True
                for (nx, ny) in self.sizes():
                    ctx = Ctx(self.lib, n, spec, nx, ny)
                    for p in rel.params:
                        for _ in range(4):
                            ok &= ctx.homogeneity(rng, p, ['__result__' if '__result__' in outk else outk[0]]) == int(m.group(1))
                rec['properties'][prop] = bool(ok)
        tol = float(prim.get('tol', 1e-13))
        rec['tol'] = tol
        ok = (not rec['oob']) and rec['max_rel_err'] <= tol and all(rec['properties'].values())
        rec['status'] = 'CONTRACT HOLDS (R-equivalent within tol; native code used for replay)' if ok else 'CONTRACT REFUTED'
        return rec

    # ----------------------------------------------------------------- facts
    def build_facts(self):
        S, V, D = self.state['units'], self.state['verification'], self.state.get('dynamic', {})
        P = self.state['static']['proofs']
        facts = []

        def add(subject, claim, evidence):
            ranks = [e for e in evidence if e.get('ok') is not None]
            bad = [e for e in ranks if e['ok'] is False]
            good = [e for e in ranks if e['ok'] is True]
            kinds = {e['kind'] for e in good}
            if bad and good: status = 'CONFLICT'
            elif bad: status = 'REFUTED'
            elif {'static-proof', 'static-structural'} & kinds and kinds & {'dynamic-bitwise', 'dynamic-test'}:
                status = 'ESTABLISHED (static + dynamic)'
            elif kinds & {'static-proof', 'static-structural'}: status = 'PROVED (static only)'
            elif kinds & {'dynamic-bitwise', 'dynamic-test'}: status = 'TESTED (dynamic only)'
            elif all(e['kind'] == 'agent-claim' for e in evidence): status = 'CLAIMED (agent, unverified)'
            else: status = 'OPEN'
            fid = hashlib.sha1(f'{subject}|{claim}'.encode()).hexdigest()[:8]
            facts.append({'id': fid, 'subject': subject, 'claim': claim, 'evidence': evidence, 'status': status})

        for n, u in S.items():
            v = V.get(n, {})
            if u['term']:
                ev = []
                if u['term_source'] == 'static':
                    ev.append({'kind': 'static-proof', 'ok': True, 'detail': f'lifted by the static analyser ({u["kind"]} mode)'})
                else:
                    ev.append({'kind': 'agent-claim', 'ok': None, 'detail': 'term proposed by the agent'})
                st = v.get('status', '')
                ran = st.startswith(('BITWISE', 'R-EQ', 'REFUTED'))
                ev.append({'kind': 'dynamic-bitwise', 'ok': (st == 'BITWISE-EQUIVALENT') if ran else None,
                           'detail': f'{st}; {v.get("n_probes")} probes at sizes {v.get("sizes")}; '
                                     f'V1 rel.err {v.get("V1_max_rel_err")}, V2 rel.err {v.get("V2_max_rel_err")}'})
                add(n, 'semantics == lifted term (order-preserving, bitwise)', ev)
            elif u.get('primitive'):
                add(n, f'opaque primitive with contract {u["primitive"]["relation"].splitlines()[-1].strip()} '
                       f'and {u["primitive"].get("properties", [])}',
                    [{'kind': 'static-proof', 'ok': None, 'detail': f'static lifting failed: {u["static"]["reason_specialized"] or u["static"]["reason"]}'},
                     {'kind': 'agent-claim', 'ok': None, 'detail': u['primitive'].get('rationale', '')},
                     {'kind': 'dynamic-test', 'ok': v.get('status', '').startswith('CONTRACT HOLDS'),
                      'detail': f'{v.get("status")}; max rel.err {v.get("max_rel_err")} (tol {v.get("tol")}); '
                                f'properties {v.get("properties")}; {v.get("n_probes")} probes'}])
            # interface
            d = D.get(n, {})
            if u['term'] and 'write_set' in d:
                t = self.term(n)
                s_out = {o for o in t.outs}
                d_out = {w.split('.')[-1] for w in d['write_set']}
                s_in = set(t.params) | set(u['static']['reads_common'])
                s_in = {x for x in s_in if self.lib.units[n].var(x).type != 'INTEGER'}
                d_in = {r.split('.')[-1] for r in d['read_set']}
                ok = (s_out == d_out or u['kind'] == 'orchestration') and s_in >= d_in
                add(n, f'reads {sorted(s_in)} -> writes {sorted(s_out)}',
                    [{'kind': 'static-structural', 'ok': True, 'detail': 'from lifted term'},
                     {'kind': 'dynamic-test', 'ok': ok, 'detail': f'probed write set {sorted(d_out)}, read set {sorted(d_in)}'}])
            # degree
            for p, sd in P['degree'].get(n, {}).items():
                dd = d.get('degree', {}).get(p, 'n/a') if d else 'n/a'
                ok = (dd == sd['degree']) if dd != 'n/a' else None
                note = f' (branch {sd["ieee_only_branches"]} is IEEE-only: equal over R)' if sd.get('ieee_only_branches') else ''
                add(n, f'polynomial degree in {p} = {sd["degree"]}{note}',
                    [{'kind': 'static-proof', 'ok': sd['degree'] is not None, 'detail': 'abstract interpretation'},
                     {'kind': 'dynamic-test', 'ok': ok, 'detail': f'finite differences: {dd}'}])
            # footprint
            if u['spec_source'] == 'static' and u['static']['footprint'] and v.get('status', '').startswith(('BITWISE', 'R-EQ')):
                lu = self.lib.units[n]
                fpl = {k: x for k, x in u['static']['footprint'].items() if k in lu.args or k in lu.dims}
                add(n, 'array footprints ' + ', '.join(f'{k}[1..{x}]' for k, x in fpl.items()),
                    [{'kind': 'static-proof', 'ok': True, 'detail': 'loop-range analysis + z3'},
                     {'kind': 'dynamic-test', 'ok': not v.get('oob'), 'detail': 'tagged-NaN canaries untouched'}])
            elif u['spec_source'] == 'agent':
                add(n, 'array lengths ' + json.dumps(u['spec']['arg_len']),
                    [{'kind': 'agent-claim', 'ok': None, 'detail': 'spec proposed by agent'},
                     {'kind': 'dynamic-test', 'ok': not v.get('oob') if v else None, 'detail': 'tagged-NaN canaries'}])
        OD = self.state.get('operator_dynamic', {})
        for op, info in P['operators'].items():
            od = OD.get(op, {})
            add(op, f'linear stencil operator in {info["param"]}, offsets {info["shifts"]}',
                [{'kind': 'static-proof', 'ok': True, 'detail': 'exact linear decomposition of the lifted term'},
                 {'kind': 'dynamic-test', 'ok': ('X->Y' in D.get(op, {}).get('shape', {})) or None,
                  'detail': str(D.get(op, {}).get('shape'))}])
            add(op, 'symmetric, IEEE-exact (c_s(p) == c_-s(p+s) modulo commutativity), all grid sizes',
                [{'kind': 'static-proof', 'ok': info['symmetric'], 'detail': '; '.join(
                    f'{p["coef"]} <-> {p["partner"]}' for p in info['pairs'])},
                 {'kind': 'dynamic-test', 'ok': od.get('symmetric_exact'), 'detail': f'exact Jacobian at sizes {od.get("sizes")}'}])
            sp = P['spd'].get(op)
            if sp:
                add(op, f'SPD over R for every grid size, with coefficients from {sp.get("coefficient_source")}',
                    [{'kind': 'static-proof', 'ok': sp.get('SPD'), 'detail': json.dumps({k: v for k, v in sp.items() if k != 'coefficient_source'})},
                     {'kind': 'dynamic-test', 'ok': od.get('dominant_to_roundoff'), 'detail': 'Gershgorin on exact Jacobian (to roundoff)'}])
        for n, pc in P['preconditioners'].items():
            od = OD.get(pc['operator'], {})
            add(n, f'Jacobi preconditioner of {pc["operator"]}: z = r / diag({pc["operator"]}) with diag = {pc["diagonal"]}',
                [{'kind': 'static-proof', 'ok': True, 'detail': 'diagonal coefficient of the operator term is unconditional'},
                 {'kind': 'dynamic-test', 'ok': od.get('diag_equals', {}).get(pc['diagonal']),
                  'detail': 'Jacobian diagonal == field bitwise'}])
        for c in self.state['static']['contracts']:
            add(c['unit'], f'interface contract {c["claim"]}', [{'kind': 'static-proof', 'ok': True, 'detail': c['why']},
                {'kind': 'dynamic-test', 'ok': ('holds' in c.get('main_call_site', '')) or None,
                 'detail': c.get('main_call_site', 'library-internal call sites')}])
        for l in self.state['static']['proof_log']:
            if l['kind'] == 'partition':
                drv = l['unit']
                add(drv, l['claim'], [{'kind': 'static-proof', 'ok': l['status'] == 'proved', 'detail': l['detail']},
                                      {'kind': 'dynamic-bitwise', 'ok': (V.get(drv, {}).get('status') == 'BITWISE-EQUIVALENT')
                                       if V.get(drv, {}).get('status', '').startswith(('BITWISE', 'R-EQ', 'REFUTED')) else None,
                                       'detail': 'driver replay compares W = concatenation bitwise'}])
            if l['kind'] == 'data-dependent-path':
                add(l['unit'], l['claim'], [{'kind': 'static-proof', 'ok': True, 'detail': l['detail']}])
        for n, t in self.state['templates'].items():
            add(n, f'implements {t["name"]} with ' + ', '.join(f'{k}={v}' for k, v in t['bindings'].items()),
                [{'kind': 'agent-claim', 'ok': None, 'detail': t.get('rationale', '')},
                 {'kind': 'static-structural', 'ok': t['match'], 'detail': t['detail']}])
        for a in self.state['annotations']:
            add(a['subject'], f'{a["name"]}: {a["meaning"]}', [{'kind': 'agent-claim', 'ok': None, 'detail': a.get('rationale', '')}])
        for j in self.state.get('justifications', []):
            add(j['unit'], f'line {j["line"]} has an unreachable branch', [{'kind': 'agent-claim', 'ok': None, 'detail': j['argument']}])
        return facts

    # -------------------------------------------------------------- worklist
    def build_worklist(self):
        S, items = self.state['units'], []

        def add(kind, subject, prio, text, hints=None):
            iid = hashlib.sha1(f'{kind}|{subject}|{text}'.encode()).hexdigest()[:6]
            items.append({'id': iid, 'kind': kind, 'subject': subject, 'priority': prio, 'task': text, 'hints': hints or {}})

        for f in self.state['facts']:
            if f['status'] in ('CONFLICT', 'REFUTED'):
                add('resolve_conflict', f['subject'], 1, f'{f["status"]}: {f["claim"]}', {'evidence': f['evidence']})
        for n, u in S.items():
            if u['term'] or u.get('primitive'): continue
            if not self.spec_complete(n):
                add('propose_spec', n, 1, f'{n} is opaque to static analysis and its array lengths are unknown. '
                    'Propose arg_len (expressions in N, NX, NY) and int_args; canaries will check them.',
                    {'static_reason': u['static']['reason_specialized'] or u['static']['reason'],
                     'current_spec': u['spec'], 'interface': self.lib.units[n].args})
            else:
                d = self.state.get('dynamic', {}).get(n, {})
                add('lift_or_primitive', n, 2, f'{n} could not be lifted statically. Propose a term '
                    '(propose_term) or register it as an opaque primitive with a checkable contract (register_primitive).',
                    {'static_reason': u['static']['reason_specialized'] or u['static']['reason'],
                     'dynamic_facts': d, 'source_lines': u['lines']})
        ver = self.state.get('verification', {})
        for unit, gaps in (self.state.get('coverage') or {}).items():
            if unit == 'error': continue
            vs = ver.get(unit, {}).get('status', '')
            if vs.startswith(('NOT RUN', 'BLOCKED', 'NO TERM')):
                continue                     # unit not exercised yet: the blocking item comes first
            just = {(j['unit'], j['line']): j for j in self.state.get('justifications', [])}
            for g in gaps:
                if (unit, g['line']) in just: continue
                if g['closed_by']: continue
                add('targeted_probe', unit, 2, f'line {g["line"]} `{g["src"]}`: {", ".join(g["why"])} by any probe. '
                    'Propose a targeted probe (run_probe) that exercises it, or explain why it is unreachable.',
                    {'line': g['line']})
        drv = self.driver()
        if drv and drv not in self.state['templates']:
            add('identify_algorithm', drv, 3, f'{drv} is the top-level driver. Identify the algorithm and submit a '
                'template with bindings (propose_template); the checker verifies the structural match.',
                {'term': S[drv]['term']})
        for op in self.state['static']['proofs']['operators']:
            if not any(a['subject'] == op for a in self.state['annotations']):
                add('annotate', op, 4, f'Name the continuous operator that {op} discretises (annotate). Optional; '
                    'recorded as an agent claim.', {'term': S[op]['term']})
        return sorted(items, key=lambda x: (x['priority'], x['subject']))

    # ======================================================= agent operations
    def _record(self, tool, args, outcome):
        self.state['proposals'].append({'tool': tool, 'args': args, 'outcome': outcome, 't': time.strftime('%H:%M:%S')})

    def propose_spec(self, unit, arg_len, int_args=None, rationale=''):
        u = self._unit(unit)
        if u['spec_source'] == 'static' and self.spec_complete(unit):
            raise AgentError(f'{unit} already has a static spec')
        sp = json.loads(json.dumps(u['spec']))
        for k, v in list(arg_len.items()) + list((int_args or {}).items()):
            if not SPEC_RE.match(str(v)): raise AgentError(f'bad expression {v!r} (allowed: digits, N, NX, NY, + - * ())')
            if k not in self.lib.units[unit].args: raise AgentError(f'{k} is not an argument of {unit}')
        sp['arg_len'].update({k: str(v) for k, v in arg_len.items()})
        sp['int_args'].update({k: str(v) for k, v in (int_args or {}).items()})
        u['spec'], u['spec_source'] = sp, 'agent'
        # immediate canary check on all sizes
        oob = []
        for (nx, ny) in self.sizes():
            ctx = Ctx(self.lib, unit, sp, nx, ny)
            for s in range(3):
                _, o = ctx.run(ctx.draw(np.random.default_rng(s)), check_reads=True)
                oob += o
        ok = not oob
        if not ok:
            u['spec_source'] = 'static'; u['spec']['arg_len'] = {k: None for k in sp['arg_len']}
        self._record('propose_spec', {'unit': unit, 'arg_len': arg_len, 'int_args': int_args, 'rationale': rationale},
                     'accepted' if ok else f'rejected: out-of-bounds access in {sorted(set(oob))}')
        self.refresh()
        return {'accepted': ok, 'out_of_bounds': sorted(set(oob)), 'dynamic_facts': self.state['dynamic'].get(unit)}

    def propose_term(self, unit, term_text, rationale=''):
        u = self._unit(unit)
        try:
            t = dsl.parse_def(term_text, self.fn_names())
        except dsl.DSLError as ex:
            raise AgentError(f'DSL: {ex}')
        if t.name != unit: raise AgentError(f'term is named {t.name}, expected {unit}')
        if u['term_source'] == 'static':
            raise AgentError(f'{unit} is already lifted statically (proved); an agent term cannot replace it')
        if not self.spec_complete(unit): raise AgentError('propose a spec first')
        fns = self.fns(); fns[unit] = t
        prod = self.capture_production()
        r = verify.ladder(self.lib, unit, self.spec(unit), t, fns, self.sizes(), 31, production=prod.get(unit))
        accepted = r['status'] == 'BITWISE-EQUIVALENT'
        if accepted:
            u['term'], u['term_source'], u['kind'] = ir.pp_def(t), 'agent', 'kernel(agent)'
        self._record('propose_term', {'unit': unit, 'term': term_text, 'rationale': rationale},
                     f'{"accepted" if accepted else "rejected"}: {r["status"]}')
        self.refresh()
        hint = None
        if r['status'].startswith('R-EQUIVALENT'):
            hint = ('Equal over R but not bitwise: the legacy code uses a different floating-point evaluation '
                    'order. Either reproduce the exact order, or register_primitive with this term as the contract.')
        return {'accepted': accepted, 'verification': r, 'hint': hint}

    def register_primitive(self, unit, relation_text, properties=(), tol=1e-13, rationale=''):
        u = self._unit(unit)
        if u['term']: raise AgentError(f'{unit} already has a term')
        if not self.spec_complete(unit): raise AgentError('propose a spec first')
        try:
            rel = dsl.parse_def(relation_text, self.fn_names())
        except dsl.DSLError as ex:
            raise AgentError(f'DSL: {ex}')
        if any(isinstance(x, ir.App) and x.fn == unit for e in rel.outs.values() for x in nodes(e)):
            raise AgentError('the contract may not call the primitive itself')
        if not (0 < float(tol) <= 1e-8): raise AgentError('tol must be in (0, 1e-8]')
        for p in properties:
            if not re.fullmatch(r'nonnegative|homogeneous:-?\d', p): raise AgentError(f'unknown property {p!r}')
        prim = {'relation': ir.pp_def(rel), 'properties': list(properties), 'tol': float(tol), 'rationale': rationale}
        r = self.check_primitive(unit, prim, self.capture_production().get(unit))
        ok = r['status'].startswith('CONTRACT HOLDS')
        if ok:
            u['primitive'], u['kind'] = prim, 'primitive'
        self._record('register_primitive', {'unit': unit, 'relation': relation_text, 'properties': list(properties),
                                            'tol': tol, 'rationale': rationale}, r['status'])
        self.refresh()
        return {'accepted': ok, 'check': r}

    def _probe_args(self, unit, inputs, size):
        lu, sp = self.lib.units[unit], self.spec(unit)
        env = {'NX': size[0], 'NY': size[1], 'N': size[0] * size[1]} if size else \
              {'N': int(inputs.get('N', 0)), 'NX': 0, 'NY': 0}
        args = {}
        for a in lu.args:
            v = inputs.get(a)
            if lu.var(a).type == 'INTEGER':
                if v is None: v = _ev(sp['int_args'][a], env)
                if not isinstance(v, int) or abs(v) > 10 ** 6: raise AgentError(f'{a}: integer expected')
            elif lu.var(a).dims:
                if isinstance(v, dict) and set(v) == {'fill'}:
                    n = _ev(sp['arg_len'][a], env) if sp['arg_len'].get(a) else env['N']
                    v = np.full(n, float(v['fill']))
                elif isinstance(v, list) and len(v) <= 10 ** 5:
                    v = np.array([float(x) for x in v])
                else:
                    raise AgentError(f'{a}: list of numbers or {{"fill": x}} expected')
            else:
                if v is None: raise AgentError(f'missing argument {a}')
                v = float(v)
            args[a] = v
        return args, env

    def exec_probe(self, unit, inputs, size=None):
        u = self.state['units'][unit]
        args, env = self._probe_args(unit, inputs, size)
        fns = self.fns()
        ref = self.term(unit) if u['term'] else (dsl.parse_def(u['primitive']['relation'], self.fn_names())
                                                if u.get('primitive') else None)
        if ref is None: raise AgentError(f'{unit} has neither term nor primitive contract yet')
        if size:
            ctx = Ctx(self.lib, unit, self.spec(unit), *size)
            base = ctx.draw(np.random.default_rng(0)); base.update(args)
            fo, oob = ctx.run(base)
            io = verify.run_ir(ctx, ref, fns, base)
        else:
            fo, oob = self.lib.call(unit, args)
            io = ir.run(ref, {p: args[p] for p in ref.params}, fns, {'NX': 0, 'NY': 0, 'N': env['N'], '__globals__': {}})
        return fo, io, oob, ref

    def run_probe(self, unit, inputs, size=None, rationale=''):
        u = self._unit(unit)
        if size is not None and not (isinstance(size, list) and len(size) == 2 and all(isinstance(x, int) and 1 <= x <= 64 for x in size)):
            raise AgentError('size must be [NX, NY] with 1 <= NX, NY <= 64')
        fo, io, oob, ref = self.exec_probe(unit, inputs, size)
        bit, err = verify.compare(fo, io, list(ref.outs))
        is_contract = not u['term']
        ok = (err <= u['primitive']['tol']) if is_contract else bit
        res = {'unit': unit, 'oob': oob, 'bitwise': bit, 'rel_err': err,
               'judged_against': 'primitive contract (R-equivalence)' if is_contract else 'lifted term (bitwise)',
               'consistent': bool(ok and not oob),
               'legacy': {k: verify._f(fo[k]).tolist()[:12] for k in ref.outs},
               'lifted': {k: verify._f(io[k]).tolist()[:12] for k in ref.outs}}
        self.state['targeted'].setdefault(unit, []).append({'inputs': inputs, 'size': size, 'rationale': rationale,
                                                            'bitwise': bit, 'rel_err': err, 'consistent': res['consistent']})
        self._record('run_probe', {'unit': unit, 'inputs': {k: (v if not isinstance(v, list) or len(v) < 10 else f'<{len(v)} values>')
                                                            for k, v in inputs.items()}, 'size': size, 'rationale': rationale},
                     f'consistent={res["consistent"]} bitwise={bit} rel_err={err:.3g}')
        self.refresh()
        return res

    def justify_unreachable(self, unit, line, argument):
        self._unit(unit)
        if len(argument) > 1000: raise AgentError('argument too long')
        self.state.setdefault('justifications', []).append({'unit': unit, 'line': int(line), 'argument': argument})
        self._record('justify_unreachable', {'unit': unit, 'line': line, 'argument': argument},
                     'recorded as agent claim (not a proof)')
        self.state['facts'] = self.build_facts(); self.state['worklist'] = self.build_worklist(); self.save()
        return {'recorded': True, 'status': 'CLAIMED (agent, unverified)'}

    def propose_template(self, unit, template_text, bindings, name, rationale=''):
        self._unit(unit)
        for k, v in bindings.items():
            if not re.fullmatch(r'[A-Za-z_]\w*', k) or v not in self.state['units']:
                raise AgentError(f'bad binding {k}={v}')
        try:
            t = dsl.parse_def(template_text, self.fn_names(bindings))
        except dsl.DSLError as ex:
            raise AgentError(f'DSL: {ex}')
        from .static.lifter import rebuild
        def subst(e):
            return rebuild(e, lambda x: ir.App(bindings[x.fn], x.args) if isinstance(x, ir.App) and x.fn in bindings else None)
        def sub_body(b):
            return [ir.Let(l.name, subst(l.e)) if isinstance(l, ir.Let) else ir.While(subst(l.cond), sub_body(l.body)) for l in b]
        t = ir.Def(unit, t.params, sub_body(t.body), {k: subst(v) for k, v in t.outs.items()})
        ok, detail = alpha_equal(t, self.term(unit))
        self.state['templates'][unit] = {'name': name, 'bindings': bindings, 'template': template_text,
                                         'match': ok, 'detail': detail, 'rationale': rationale}
        if not ok: del self.state['templates'][unit]
        self._record('propose_template', {'unit': unit, 'name': name, 'bindings': bindings, 'rationale': rationale},
                     f'match={ok}: {detail}')
        self.refresh()
        return {'match': ok, 'detail': detail}

    def annotate(self, subject, name, meaning, rationale=''):
        if subject not in self.state['units']: raise AgentError(f'unknown unit {subject}')
        if len(name) > 120 or len(meaning) > 600: raise AgentError('too long')
        self.state['annotations'] = [a for a in self.state['annotations'] if a['subject'] != subject]
        self.state['annotations'].append({'subject': subject, 'name': name, 'meaning': meaning, 'rationale': rationale})
        self._record('annotate', {'subject': subject, 'name': name, 'meaning': meaning}, 'recorded as agent claim')
        self.state['facts'] = self.build_facts(); self.state['worklist'] = self.build_worklist(); self.save()
        return {'recorded': True}

    def _unit(self, unit):
        if unit not in self.state['units']: raise AgentError(f'unknown unit {unit}')
        return self.state['units'][unit]

    # ------------------------------------------------------------- read views
    def status(self):
        cnt = {}
        for f in self.state['facts']: cnt[f['status']] = cnt.get(f['status'], 0) + 1
        return {'units': {n: {'kind': u['kind'], 'term_source': u['term_source'],
                              'verification': self.state['verification'].get(n, {}).get('status')}
                          for n, u in self.state['units'].items()},
                'facts': cnt, 'open_worklist_items': len(self.state['worklist'])}

    def show(self, unit):
        u = self._unit(unit)
        return {'unit': unit, 'kind': u['kind'], 'term': u['term'], 'term_source': u['term_source'],
                'primitive': u.get('primitive'), 'static_reason': u['static']['reason_specialized'] or u['static']['reason'],
                'spec': u['spec'], 'spec_source': u['spec_source'],
                'source': self.source(unit),
                'verification': self.state['verification'].get(unit),
                'dynamic_facts': self.state.get('dynamic', {}).get(unit),
                'facts': [f for f in self.state['facts'] if f['subject'] == unit]}

    def source(self, unit):
        a, b = self.state['units'][unit]['lines']
        lines = open(self.state['src']).read().splitlines()
        return '\n'.join(f'{i:4d} {lines[i - 1]}' for i in range(a, b + 1))


# ------------------------------------------------------------- alpha-equality
def alpha_equal(t, u):
    """Structural equality of two Defs up to consistent renaming of bound names."""
    m = dict(zip(t.params, u.params))
    if len(t.params) != len(u.params): return False, f'parameter count {len(t.params)} vs {len(u.params)}'

    def eq(a, b):
        if type(a) is not type(b): return False
        if isinstance(a, ir.Var): return m.get(a.name, a.name) == b.name
        if isinstance(a, ir.Lit): return float(a.v) == float(b.v)
        if isinstance(a, ir.Idx): return a.axis == b.axis
        if isinstance(a, ir.App):
            return a.fn == b.fn and len(a.args) == len(b.args) and all(eq(x, y) for x, y in zip(a.args, b.args))
        for f in ('op', 'di', 'dj'):
            if getattr(a, f, None) != getattr(b, f, None): return False
        from .static.lifter import _children
        ca, cb = _children(a), _children(b)
        return len(ca) == len(cb) and all(eq(x, y) for x, y in zip(ca, cb))

    def body(ba, bb, path):
        if len(ba) != len(bb): return False, f'{path}: {len(ba)} vs {len(bb)} statements'
        for k, (x, y) in enumerate(zip(ba, bb)):
            if type(x) is not type(y): return False, f'{path}[{k}]: let vs while'
            if isinstance(x, ir.Let):
                if not eq(x.e, y.e): return False, f'{path}[{k}]: let {x.name} = {ir.pp(x.e)}  vs  let {y.name} = {ir.pp(y.e)}'
                if x.name in m and m[x.name] != y.name: return False, f'{path}[{k}]: inconsistent renaming of {x.name}'
                m[x.name] = y.name
            else:
                if not eq(x.cond, y.cond): return False, f'{path}[{k}]: loop condition differs'
                ok, d = body(x.body, y.body, f'{path}[{k}].body')
                if not ok: return False, d
        return True, ''
    ok, d = body(t.body, u.body, 'body')
    if not ok: return False, d
    for k, e in t.outs.items():
        if k not in u.outs: return False, f'output {k} missing'
        if not eq(e, u.outs[k]): return False, f'output {k} differs'
    return True, f'alpha-equivalent ({len(m)} names matched)'


# ----------------------------------------------------------- structure document
def structure_md(ws):
    st = ws.state
    S, F, P = st['units'], st['facts'], st['static']['proofs']
    L = []
    w = L.append
    nf = {}
    for f in F: nf[f['status']] = nf.get(f['status'], 0) + 1
    w(f'# Extracted structure: `{os.path.basename(st["src"])}`\n')
    w(f'Source sha256 `{st["build"]["src_sha256"]}` · build `{st["build"]["cmd"]}` · '
      f'{len(F)} facts: ' + ', '.join(f'{v} {k}' for k, v in sorted(nf.items())) +
      f' · open worklist items: {len(st["worklist"])}\n')
    w('Evidence levels: **static-proof** (all sizes, z3/abstract interpretation) · **dynamic** '
      '(bitwise replay / canaries / exact Jacobians on 2 visible + 1 checker-secret size) · '
      '**agent-claim** (LLM proposal, recorded but not trusted unless backed by one of the above).\n')
    drv = ws.driver()
    w('## 1. Algorithm\n')
    if drv in st['templates']:
        t = st['templates'][drv]
        w(f'`{drv}` **is {t["name"]}** with ' + ', '.join(f'`{k}` = `{v}`' for k, v in t['bindings'].items()) +
          f'. Structural match: {t["detail"]}.\n')
    if drv:
        w(f'Lifted driver (`{drv}`):\n\n```\n{S[drv]["term"]}\n```\n')
    w('## 2. Operators and their mathematical properties\n')
    for op, info in P['operators'].items():
        ann = [a for a in st['annotations'] if a['subject'] == op]
        w(f'### `{op}` — linear stencil operator in `{info["param"]}`' + (f': {ann[0]["name"]}' if ann else '') + '\n')
        if ann: w(f'*Agent interpretation:* {ann[0]["meaning"]}\n')
        w('| offset | coefficient (exact, as computed) |\n|---|---|')
        sym = getattr(ws, '_sym_cache', {}).get(op)
        w(f'| (0,0) | `{", ".join(info["diagonal"])}` |')
        for p in info['pairs']:
            w(f'| {tuple(p["shift"])} / {tuple(-x for x in p["shift"])} | `{p["coef"]}` / `{p["partner"]}` |')
        w('')
        for f in F:
            if f['subject'] == op and f['claim'].startswith(('symmetric', 'SPD')):
                w(f'- **{f["claim"]}** — {f["status"]}')
        for n, pc in P['preconditioners'].items():
            if pc['operator'] == op:
                w(f'- **`{n}` is the Jacobi preconditioner of `{op}`** (diag = `{pc["diagonal"]}`)')
        w('')
    w('## 3. Units\n')
    w('| unit | kind | lifted form | verification |\n|---|---|---|---|')
    for n, u in S.items():
        v = st['verification'].get(n, {})
        if u['term']:
            t = u['term'].splitlines()
            form = ' '.join(x.strip() for x in t[1:]) if len(t) <= 3 else f'{len(t) - 2} lets, see below'
            form = f'`{form[:150]}`'
        elif u.get('primitive'):
            form = f'primitive ≈ `{u["primitive"]["relation"].splitlines()[-1].strip()}` {u["primitive"]["properties"]}'
        else:
            form = f'opaque: {u["static"]["reason_specialized"] or u["static"]["reason"]}'
        src = {'static': 'static lift', 'agent': 'agent term'}.get(u['term_source'], u['kind'])
        w(f'| `{n}` | {src} | {form} | {v.get("status", "")} |')
    w('\nFull lifted terms:\n')
    for n, u in S.items():
        if u['term'] and n != drv:
            w(f'```\n{u["term"]}\n```')
    w('## 4. Contracts, assumptions and IEEE-only behaviour\n')
    for f in F:
        if 'contract' in f['claim'] or 'IEEE' in f['claim'] or f['claim'].startswith('W ='):
            w(f'- `{f["subject"]}`: {f["claim"]} — **{f["status"]}**')
    w('\n## 5. Coverage\n')
    cov = st.get('coverage', {})
    closed_s, closed_a, open_ = [], [], []
    for unit, gaps in cov.items():
        if unit == 'error': continue
        for g in gaps:
            cb = g['closed_by'] or ''
            (closed_s if cb.startswith('static') else closed_a if cb.startswith('agent') else open_).append((unit, g))
    w(f'- branches closed by **static proof** (infeasible on every path): ' +
      (', '.join(f'`{u}`:{g["line"]}' for u, g in closed_s) or 'none'))
    w(f'- branches the agent argues are unreachable (**unverified claim**): ' +
      (', '.join(f'`{u}`:{g["line"]}' for u, g in closed_a) or 'none'))
    tp = {u: len(v) for u, v in st['targeted'].items()}
    w(f'- targeted probes run by the agent: {tp or "none"}')
    w(f'- still uncovered: ' + (', '.join(f'`{u}`:{g["line"]} `{g["src"]}`' for u, g in open_) or '**none**'))
    w('\n## 6. All facts\n')
    w('| subject | claim | status | evidence |\n|---|---|---|---|')
    for f in F:
        ev = '; '.join(f'{e["kind"]}:{"✔" if e["ok"] else ("✘" if e["ok"] is False else "·")}' for e in f['evidence'])
        w(f'| `{f["subject"]}` | {f["claim"].replace("|", "/")[:160]} | {f["status"]} | {ev} |')
    w('\n## 7. Agent contributions\n')
    for p in st['proposals']:
        a = {k: v for k, v in p['args'].items() if k not in ('term', 'relation')}
        w(f'- `{p["tool"]}` {json.dumps(a)[:220]} → **{p["outcome"][:200]}**')
    w('\n## 8. Remaining worklist\n')
    for it in st['worklist']:
        w(f'- [{it["kind"]}] `{it["subject"]}`: {it["task"]}')
    if not st['worklist']: w('- empty')
    return '\n'.join(L) + '\n'
