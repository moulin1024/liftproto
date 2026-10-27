# Agent guide: lifting legacy scientific code into checked structure

You are the *proposer* in a propose–verify loop. A deterministic checker (static
analysis + z3 + dynamic probes on the real binary) owns all evidence. You never
edit the legacy code, the checker, the workspace state, or the reference data.
Your only channel is the tool API (`liftctl.py WS TOOL '{json}'` or the MCP tools).

## Loop

1. `worklist` and do the highest-priority item first. Use `show <unit>` to see the
   source, the static analyser's reason, the dynamic facts, and the current facts.
2. Act on it with exactly one proposal tool, and give a short `rationale`.
3. Read the checker's answer. A rejection carries a diagnostic, so fix the proposal or
   change strategy. Do not resubmit the same thing.
4. Repeat until the worklist is empty (priority 4 items are optional). Finish with
   `structure`.

## What each item asks of you

| item | what to do |
|---|---|
| `propose_spec` | Read the unit's source and give array lengths as expressions in `N`, `NX`, `NY`, plus integer args such as `INCX` (use the call-site constants listed in contracts). Tagged-NaN canaries reject lengths that are too short. |
| `lift_or_primitive` | Prefer `propose_term` if you can reproduce the exact floating-point evaluation order. Otherwise use `register_primitive` with a mathematically equivalent relation and the properties you believe hold. The legacy code then stays the executable reference ("native"), and your relation becomes a checked contract. |
| `targeted_probe` | Build inputs that exercise the named line (`run_probe`; grid units need `"size": [NX, NY]`). Think about IEEE corner cases (signed zeros, N = 0, N = 1, N below the unroll factor, zeros in data). If you believe the branch is unreachable, use `justify_unreachable`. That is recorded as an unverified claim, never as a proof. |
| `identify_algorithm` | Write the textbook algorithm as a DSL template with placeholder calls (for example `A(P)`, `Minv(R)`, `dot(R, Z)`) and bind the placeholders to units. The match is exact alpha-equivalence, so mirror the lifted driver's statement order. |
| `resolve_conflict` | Static and dynamic evidence disagree. Investigate with `show` and `run_probe`, then report. Never try to make a conflict disappear by weakening a claim. |
| `annotate` | Give the continuous or physical meaning. It is recorded as an *agent claim* and is never trusted as evidence. |

## Rules

- Proposals are data in a restricted DSL (`dsl_help`). Nothing you send is executed.
- Bitwise equivalence is the bar for a *term*. R-equivalence only qualifies an opaque *primitive's contract*.
- The checker also tests on a problem size you never see. Do not tailor proposals to the visible sizes.
- Report honestly. If you cannot do an item, leave it open and say why in the rationale of your last action.
