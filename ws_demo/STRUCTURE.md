# Extracted structure: `pcg.f`

Source sha256 `903b2a6d22356fa1` · build `gfortran -std=legacy -fPIC -shared -ffp-contract=off -fno-fast-math -O2 pcg.f` · 48 facts: 3 CLAIMED (agent, unverified), 40 ESTABLISHED (static + dynamic), 3 PROVED (static only), 2 TESTED (dynamic only) · open worklist items: 0

Evidence levels: **static-proof** (all sizes, z3/abstract interpretation) · **dynamic** (bitwise replay / canaries / exact Jacobians on 2 visible + 1 checker-secret size) · **agent-claim** (LLM proposal, recorded but not trusted unless backed by one of the above).

## 1. Algorithm

`CGSOLV` **is preconditioned conjugate gradient (Hestenes-Stiefel, Saad Alg. 9.1)** with `A` = `AMUL`, `Minv` = `PCOND`, `dot` = `DDOT`, `nrm` = `DNRM2`, `axpy` = `DAXPY`, `copy` = `DCOPY`, `xpay` = `XPAY`. Structural match: alpha-equivalent (15 names matched).

Lifted driver (`CGSOLV`):

```
CGSOLV = λ(B, X, TOL, MAXIT).
    let Q = AMUL(X)
    let R = B - Q
    let Z = PCOND(R)
    let P = DCOPY(Z)
    let RHO = DDOT(R, Z)
    let BNRM = DNRM2(B)
    let ITER = 0
    let RNORM = DNRM2(R)
    while_loop  cond: not (RNORM <= TOL * BNRM) and not (ITER >= MAXIT)
        let ITER = ITER + 1
        let Q = AMUL(P)
        let ALPHA = RHO / DDOT(P, Q)
        let X = DAXPY(ALPHA, P, X)
        let R = DAXPY(-ALPHA, Q, R)
        let Z = PCOND(R)
        let RHONEW = DDOT(R, Z)
        let BETA = RHONEW / RHO
        let RHO = RHONEW
        let P = XPAY(Z, BETA, P)
        let RNORM = DNRM2(R)
    in  {X = X, W = [R ++ Z ++ P ++ Q], ITER = ITER, RNORM = RNORM}
```

## 2. Operators and their mathematical properties

### `AMUL` — linear stencil operator in `X`: -div(k grad u) with homogeneous Dirichlet BC

*Agent interpretation:* Standard 5-point finite-difference/finite-volume discretisation of -d/dx(k du/dx) - d/dy(k du/dy) on the unit square, mesh h = 1/(N+1); face conductivities are arithmetic means k_{i+1/2,j} = (k_ij + k_{i+1,j})/2 (coefficient HX2I*0.5*(CK+CK[i+1,j]) = k_{i+1/2}/h^2); boundary faces use the cell value (ghost k = k_ij), u = 0 outside the grid.

| offset | coefficient (exact, as computed) |
|---|---|
| (0,0) | `DG` |
| (1, 0) / (-1, 0) | `-(HX2I * 0.5 * (CK + CK[i+1,j]))` / `-(HX2I * 0.5 * (CK + CK[i-1,j]))` |
| (0, 1) / (0, -1) | `-(HY2I * 0.5 * (CK + CK[i,j+1]))` / `-(HY2I * 0.5 * (CK + CK[i,j-1]))` |

- **symmetric, IEEE-exact (c_s(p) == c_-s(p+s) modulo commutativity), all grid sizes** — ESTABLISHED (static + dynamic)
- **SPD over R for every grid size, with coefficients from SETUP** — ESTABLISHED (static + dynamic)
- **`PCOND` is the Jacobi preconditioner of `AMUL`** (diag = `DG`)

## 3. Units

| unit | kind | lifted form | verification |
|---|---|---|---|
| `AMUL` | static lift | `4 lets, see below` | BITWISE-EQUIVALENT |
| `DAXPY` | static lift | `in  {DY = (DA == 0 ? DY : DY + DA * DX)}` | BITWISE-EQUIVALENT |
| `DCOPY` | static lift | `in  {DY = DX}` | BITWISE-EQUIVALENT |
| `DDOT` | static lift | `in  {__result__ = foldl(+, 0, DX * DY)}` | BITWISE-EQUIVALENT |
| `DNRM2` | primitive | primitive ≈ `in  {__result__ = sqrt(foldl(+, 0, X * X))}` ['nonnegative', 'homogeneous:1'] | CONTRACT HOLDS (R-equivalent within tol; native code used for replay) |
| `PCOND` | static lift | `in  {Z = R / DG}` | BITWISE-EQUIVALENT |
| `XPAY` | static lift | `in  {P = Z + BETA * P}` | BITWISE-EQUIVALENT |
| `CGSOLV` | static lift | `20 lets, see below` | BITWISE-EQUIVALENT |
| `SETUP` | static lift | `5 lets, see below` | BITWISE-EQUIVALENT |

Full lifted terms:

```
AMUL = λ(X).
    let t1 = DG * X
    let t2 = (i == 1 ? t1 : t1 - HX2I * 0.5 * (CK + CK[i-1,j]) * X[i-1,j])
    let t3 = (i == NX ? t2 : t2 - HX2I * 0.5 * (CK + CK[i+1,j]) * X[i+1,j])
    let t4 = (j == 1 ? t3 : t3 - HY2I * 0.5 * (CK + CK[i,j-1]) * X[i,j-1])
    in  {Y = (j == NY ? t4 : t4 - HY2I * 0.5 * (CK + CK[i,j+1]) * X[i,j+1])}
```
```
DAXPY = λ(DA, DX, DY).
    in  {DY = (DA == 0 ? DY : DY + DA * DX)}
```
```
DCOPY = λ(DX).
    in  {DY = DX}
```
```
DDOT = λ(DX, DY).
    in  {__result__ = foldl(+, 0, DX * DY)}
```
```
PCOND = λ(R).
    in  {Z = R / DG}
```
```
XPAY = λ(Z, BETA, P).
    in  {P = Z + BETA * P}
```
```
SETUP = λ(NXI, NYI).
    let CK = 1 + 0.5 * sin(2 * (4 * atan(1)) * i * (1 / (1 + NXI))) * cos(4 * atan(1) * j * (1 / (1 + NYI)))
    let t1 = 1 / (1 + NXI)
    let t2 = 1 / (t1 * t1)
    let t3 = 1 / (1 + NYI)
    let t4 = 1 / (t3 * t3)
    in  {NX = NXI, NY = NYI, HX2I = t2, HY2I = t4, CK = CK, DG = t2 * 0.5 * (CK + (i > 1 ? CK[i-1,j] : CK)) + t2 * 0.5 * (CK + (i < NXI ? CK[i+1,j] : CK)) + t4 * 0.5 * (CK + (j > 1 ? CK[i,j-1] : CK)) + t4 * 0.5 * (CK + (j < NYI ? CK[i,j+1] : CK))}
```
## 4. Contracts, assumptions and IEEE-only behaviour

- `DAXPY`: polynomial degree in DA = 1 (branch ['DA == 0'] is IEEE-only: equal over R) — **ESTABLISHED (static + dynamic)**
- `DNRM2`: opaque primitive with contract in  {__result__ = sqrt(foldl(+, 0, X * X))} and ['nonnegative', 'homogeneous:1'] — **TESTED (dynamic only)**
- `AMUL`: symmetric, IEEE-exact (c_s(p) == c_-s(p+s) modulo commutativity), all grid sizes — **ESTABLISHED (static + dynamic)**
- `CGSOLV`: interface contract NX*NY == N — **ESTABLISHED (static + dynamic)**
- `DNRM2`: interface contract INCX == 1 — **PROVED (static only)**
- `DAXPY`: DY: separate IEEE semantics when DA == 0 — **PROVED (static only)**
- `CGSOLV`: W = R[1..N] ++ Z[N + 1..2*N] ++ P[2*N + 1..3*N] ++ Q[3*N + 1..4*N]  (disjoint, tiles [1..4*N]) — **ESTABLISHED (static + dynamic)**

## 5. Coverage

- branches closed by **static proof** (infeasible on every path): `DDOT`:98, `DAXPY`:120
- branches the agent argues are unreachable (**unverified claim**): `DNRM2`:202
- targeted probes run by the agent: {'DAXPY': 3, 'DDOT': 2, 'DNRM2': 3, 'CGSOLV': 1}
- still uncovered: **none**

## 6. All facts

| subject | claim | status | evidence |
|---|---|---|---|
| `AMUL` | semantics == lifted term (order-preserving, bitwise) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `AMUL` | reads ['CK', 'DG', 'HX2I', 'HY2I', 'X'] -> writes ['Y'] | ESTABLISHED (static + dynamic) | static-structural:✔; dynamic-test:✔ |
| `AMUL` | polynomial degree in X = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `AMUL` | array footprints Y[1..NX*NY], DG[1..NX*NY], X[1..NX*NY], CK[1..NX*NY] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DAXPY` | semantics == lifted term (order-preserving, bitwise) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `DAXPY` | reads ['DA', 'DX', 'DY'] -> writes ['DY'] | ESTABLISHED (static + dynamic) | static-structural:✔; dynamic-test:✔ |
| `DAXPY` | polynomial degree in DA = 1 (branch ['DA == 0'] is IEEE-only: equal over R) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DAXPY` | polynomial degree in DX = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DAXPY` | polynomial degree in DY = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DAXPY` | array footprints DY[1..N], DX[1..N] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DCOPY` | semantics == lifted term (order-preserving, bitwise) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `DCOPY` | reads ['DX'] -> writes ['DY'] | ESTABLISHED (static + dynamic) | static-structural:✔; dynamic-test:✔ |
| `DCOPY` | polynomial degree in DX = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DCOPY` | array footprints DY[1..N], DX[1..N] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DDOT` | semantics == lifted term (order-preserving, bitwise) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `DDOT` | reads ['DX', 'DY'] -> writes ['__result__'] | ESTABLISHED (static + dynamic) | static-structural:✔; dynamic-test:✔ |
| `DDOT` | polynomial degree in DX = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DDOT` | polynomial degree in DY = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DDOT` | array footprints DX[1..N], DY[1..N] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DNRM2` | opaque primitive with contract in  {__result__ = sqrt(foldl(+, 0, X * X))} and ['nonnegative', 'homogeneous:1'] | TESTED (dynamic only) | static-proof:·; agent-claim:·; dynamic-test:✔ |
| `DNRM2` | array lengths {"X": "N"} | TESTED (dynamic only) | agent-claim:·; dynamic-test:✔ |
| `PCOND` | semantics == lifted term (order-preserving, bitwise) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `PCOND` | reads ['DG', 'R'] -> writes ['Z'] | ESTABLISHED (static + dynamic) | static-structural:✔; dynamic-test:✔ |
| `PCOND` | polynomial degree in R = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `PCOND` | array footprints Z[1..N], R[1..N], DG[1..N] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `XPAY` | semantics == lifted term (order-preserving, bitwise) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `XPAY` | reads ['BETA', 'P', 'Z'] -> writes ['P'] | ESTABLISHED (static + dynamic) | static-structural:✔; dynamic-test:✔ |
| `XPAY` | polynomial degree in Z = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `XPAY` | polynomial degree in BETA = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `XPAY` | polynomial degree in P = 1 | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `XPAY` | array footprints P[1..N], Z[1..N] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `CGSOLV` | semantics == lifted term (order-preserving, bitwise) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `CGSOLV` | reads ['B', 'TOL', 'X'] -> writes ['ITER', 'RNORM', 'W', 'X'] | ESTABLISHED (static + dynamic) | static-structural:✔; dynamic-test:✔ |
| `CGSOLV` | array footprints W[1..4*N], X[1..N], B[1..N] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `SETUP` | semantics == lifted term (order-preserving, bitwise) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `SETUP` | array footprints CK[1..NXI*NYI], DG[1..NXI*NYI] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `AMUL` | linear stencil operator in X, offsets [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]] | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `AMUL` | symmetric, IEEE-exact (c_s(p) == c_-s(p+s) modulo commutativity), all grid sizes | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `AMUL` | SPD over R for every grid size, with coefficients from SETUP | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `PCOND` | Jacobi preconditioner of AMUL: z = r / diag(AMUL) with diag = DG | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `CGSOLV` | interface contract NX*NY == N | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-test:✔ |
| `DNRM2` | interface contract INCX == 1 | PROVED (static only) | static-proof:✔; dynamic-test:· |
| `DAXPY` | DY: separate IEEE semantics when DA == 0 | PROVED (static only) | static-proof:✔ |
| `CGSOLV` | W = R[1..N] ++ Z[N + 1..2*N] ++ P[2*N + 1..3*N] ++ Q[3*N + 1..4*N]  (disjoint, tiles [1..4*N]) | ESTABLISHED (static + dynamic) | static-proof:✔; dynamic-bitwise:✔ |
| `CGSOLV` | implements preconditioned conjugate gradient (Hestenes-Stiefel, Saad Alg. 9.1) with A=AMUL, Minv=PCOND, dot=DDOT, nrm=DNRM2, axpy=DAXPY, copy=DCOPY, xpay=XPAY | PROVED (static only) | agent-claim:·; static-structural:✔ |
| `AMUL` | -div(k grad u) with homogeneous Dirichlet BC: Standard 5-point finite-difference/finite-volume discretisation of -d/dx(k du/dx) - d/dy(k du/dy) on the unit squa | CLAIMED (agent, unverified) | agent-claim:· |
| `SETUP` | conductivity field and mesh factors: k(x,y) = 1 + 0.5 sin(2 pi x) cos(pi y) at x = i h_x, y = j h_y; HX2I = 1/h_x^2, HY2I = 1/h_y^2; DG = diagonal of the operat | CLAIMED (agent, unverified) | agent-claim:· |
| `DNRM2` | line 202 has an unreachable branch | CLAIMED (agent, unverified) | agent-claim:· |

## 7. Agent contributions

- `propose_spec` {"unit": "DNRM2", "arg_len": {"X": "N"}, "int_args": {"INCX": "1"}, "rationale": "DO 10 IX = 1,1+(N-1)*INCX,INCX reads X(1..1+(N-1)*INCX); INCX == 1 at every call site (static contract), so X(1..N)."} → **accepted**
- `propose_term` {"unit": "DNRM2", "rationale": "Scaled sum of squares = Euclidean norm; try the direct form."} → **rejected: R-EQUIVALENT (float-different)**
- `register_primitive` {"unit": "DNRM2", "properties": ["nonnegative", "homogeneous:1"], "tol": 1e-14, "rationale": "Reference-BLAS DNRM2 (Hammarling): overflow-safe scaled sum of squares with state (SCALE, SSQ). The DSL has no stateful fold t → **CONTRACT HOLDS (R-equivalent within tol; native code used for replay)**
- `run_probe` {"unit": "DAXPY", "inputs": {"N": 6, "DA": 0.0, "DX": [1, 2, 3, 4, 5, 6], "DY": [-0.0, 1, -0.0, 2, 3, -0.0]}, "size": null, "rationale": "DA==0 early return; signed zeros in DY distinguish return-unchanged from DY + 0*DX → **consistent=True bitwise=True rel_err=0**
- `run_probe` {"unit": "DAXPY", "inputs": {"N": 3, "DA": 0.5, "DX": [1, 2, 3], "DY": [1, 1, 1]}, "size": null, "rationale": "N < 4: only the remainder loop runs"} → **consistent=True bitwise=True rel_err=0**
- `run_probe` {"unit": "DAXPY", "inputs": {"N": 0, "DA": 0.5, "DX": [], "DY": []}, "size": null, "rationale": "N <= 0 early return"} → **consistent=True bitwise=True rel_err=0**
- `run_probe` {"unit": "DDOT", "inputs": {"N": 0, "DX": [], "DY": []}, "size": null, "rationale": "N <= 0 early return"} → **consistent=True bitwise=True rel_err=0**
- `run_probe` {"unit": "DDOT", "inputs": {"N": 3, "DX": [1, 2, 3], "DY": [0.1, 0.2, 0.3]}, "size": null, "rationale": "N < 5: remainder loop only, jump to 60"} → **consistent=True bitwise=True rel_err=0**
- `run_probe` {"unit": "DNRM2", "inputs": {"N": 0, "X": []}, "size": null, "rationale": "N < 1 branch"} → **consistent=True bitwise=True rel_err=0**
- `run_probe` {"unit": "DNRM2", "inputs": {"N": 1, "X": [-3.0]}, "size": null, "rationale": "N == 1 branch returns ABS(X(1))"} → **consistent=True bitwise=True rel_err=0**
- `run_probe` {"unit": "DNRM2", "inputs": {"N": 5, "X": [0.0, 3.0, 0.0, -4.0, 0.0]}, "size": null, "rationale": "zeros in X skip the update (X(IX) == 0 branch)"} → **consistent=True bitwise=True rel_err=0**
- `run_probe` {"unit": "CGSOLV", "inputs": {"B": {"fill": 1.0}, "X": {"fill": 0.0}, "W": {"fill": 0.0}, "TOL": 1e-10, "MAXIT": 3, "RNORM": 0.0}, "size": [7, 5], "rationale": "exit through ITER >= MAXIT instead of convergence"} → **consistent=True bitwise=True rel_err=0**
- `justify_unreachable` {"unit": "DNRM2", "line": 202, "argument": "The DO loop is only reached in the ELSE branch, i.e. N >= 2 and INCX >= 1, so the upper bound 1+(N-1)*INCX >= 2 > 1 = lower bound: the zero-trip branch cannot be taken."} → **recorded as agent claim (not a proof)**
- `propose_template` {"unit": "CGSOLV", "name": "preconditioned conjugate gradient (Hestenes-Stiefel, Saad Alg. 9.1)", "bindings": {"A": "AMUL", "Minv": "PCOND", "dot": "DDOT", "nrm": "DNRM2", "axpy": "DAXPY", "copy": "DCOPY", "xpay": "XPAY" → **match=False: body: 8 vs 9 statements**
- `propose_template` {"unit": "CGSOLV", "name": "preconditioned conjugate gradient (Hestenes-Stiefel, Saad Alg. 9.1)", "bindings": {"A": "AMUL", "Minv": "PCOND", "dot": "DDOT", "nrm": "DNRM2", "axpy": "DAXPY", "copy": "DCOPY", "xpay": "XPAY" → **match=True: alpha-equivalent (15 names matched)**
- `annotate` {"subject": "AMUL", "name": "-div(k grad u) with homogeneous Dirichlet BC", "meaning": "Standard 5-point finite-difference/finite-volume discretisation of -d/dx(k du/dx) - d/dy(k du/dy) on the unit square, mesh h = 1/(N+ → **recorded as agent claim**
- `annotate` {"subject": "SETUP", "name": "conductivity field and mesh factors", "meaning": "k(x,y) = 1 + 0.5 sin(2 pi x) cos(pi y) at x = i h_x, y = j h_y; HX2I = 1/h_x^2, HY2I = 1/h_y^2; DG = diagonal of the operator."} → **recorded as agent claim**

## 8. Remaining worklist

- empty
