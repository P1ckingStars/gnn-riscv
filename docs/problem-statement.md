# Problem statement: GNN-based spec → RV64GC assembly synthesis

*Status: v2 draft (2026-05-17). Living document — update as decisions firm up.*

## 1. Problem

Given a formal specification `S = (inputs, outputs, pre, post)` over fixed-width
integers, produce an RV64GC scalar assembly sequence `A` (the candidate function `f`)
such that:

1. **Spec satisfaction**: ``∀ inputs. pre(inputs) → post(inputs, f(inputs))`` (§3).
2. **Cost minimization**: among satisfying candidates, prefer the `A` minimizing a
   stated cost function — instruction count, with an in-order RV64 cycle estimate as a
   secondary tiebreaker (§4).

The research claim is that a **GNN-based generative model of the compute graph** is a
better inductive bias than seq2seq emission of asm tokens, because programs are DAGs
and a generative-GNN decoder can exploit graph structure during search.

## 2. The spec language

A first-order-logic DSL over fixed-width-integer bitvectors. Two layers:

**Term layer** — pure arithmetic/bitwise expressions; carry a type ``Ty ∈ {i8,i16,i32,i64}``:

```
Ty   ::= i8 | i16 | i32 | i64
Expr ::= Const(value, Ty)
       | Var(name, Ty)
       | Bin(op, Expr, Expr)              -- op ∈ {add,sub,mul,sdiv,udiv,srem,urem,
                                          --       and,or,xor,shl,lshr,ashr}
       | Un(op, Expr)                     -- op ∈ {neg,not,sext,zext,trunc}
       | Select(cond:Expr, then:Expr, else_:Expr)
```

**Formula layer** — propositional + first-order formulas over terms:

```
Formula ::= true | false
          | Cmp(op, Expr, Expr)              -- op ∈ {eq,ne,slt,sle,sgt,sge,
                                             --        ult,ule,ugt,uge}
          | Not(Formula)
          | And(Formula, Formula) | Or(Formula, Formula)
          | Implies(Formula, Formula) | Iff(Formula, Formula)
          | Forall(x:Ty, Formula) | Exists(x:Ty, Formula)
          | LetTerm(name, Ty, Expr, Formula)
```

A **Spec** is a Hoare triple:

```
Spec ::= (inputs: [Param], outputs: [Param], pre: Formula, post: Formula)
```

with v1 supporting single-output specs. ``pre`` defaults to ``true``. The candidate ``f``
is correct iff ``pre(inputs) → post(inputs, f(inputs))`` for all inputs.

**Functional special case.** When ``post`` has the form ``r == body``, the spec degenerates
to a pure functional spec — ``Spec.functional(inputs, body)`` is the convenience
constructor. The expression DSL alone (the term layer) is recovered as this special case.

**Term constraints.**

- Binary ops require operand-type equality (e.g. `add` of i32 + i32 → i32). Shifts are
  the special case: amount ranges in `[0, width)` to avoid UB; the generator forces
  shift amounts to a const in range.
- Divisions and remainders are UB on zero divisor. The IO sampler avoids such inputs via
  `pre`; the SMT verifier conjoins the UB-free predicate with `pre`.
- `sext`/`zext` require result wider than arg; `trunc` requires result narrower.

**Surface syntax.** See ``spec/parser.py`` and ``examples/specs/``. Tokens:

- Comparison: ``== != <s <=s >s >=s <u <=u >u >=u``
- Term ops: ``+ - * & | ^ ~`` plus signedness-tagged ``/s /u %s %u >>s >>u <<``
- Connectives: ``~ & | -> <->``; keywords ``true false``
- Quantifiers: ``forall x: i32. F``, ``exists x: i32. F``
- Built-ins (term-forming): ``sext(e, ty)``, ``zext(e, ty)``, ``trunc(e, ty)``,
  ``select(cond_term, then, else_)`` (cond is treated as nonzero/zero — for Formula
  conditions, use implications instead).

## 3. Spec satisfaction (verification)

Define `[[A]] : Z* → Z` as the input/output function of the candidate asm `A`, with
inputs masked to declared parameter widths and outputs masked to the declared output
width.

Candidate `A` **satisfies** spec `S` iff:

```
∀ x ∈ Z*. pre(x) ∧ no_UB(pre, post, x) → post(x, [[A]](x))
```

Two verifiers implement this in tandem:

- **`eval.verify_io`** (fast filter). Sample N = 1000 inputs from `sample_inputs(spec)`
  (already Pre-filtered with random + boundary values), assemble `A` with
  `riscv64-linux-gnu-as`, link a generated harness, run on `qemu-riscv64-static`.
  For each row, check `satisfies(spec, input, [observed_output])`. Used during training
  and search.
- **`eval.verify_smt`** (soundness gate). Translate spec and asm into Z3 bitvector
  formulas. Query:

  ```
  pre(x) ∧ no_UB(x) ∧ ¬post(x, f(x))     -- UNSAT → A satisfies S
                                          -- SAT   → counterexample
                                          -- UNKNOWN → timed out
  ```

  Quantifiers in spec (`Forall`/`Exists`) are emitted as Z3 quantifiers natively.
  Only the integer subset of RV64GC is modeled in v1; SMT verification of asm using
  ops outside the supported set raises ``UnsupportedAsm`` and the caller falls back to
  `verify_io` only.

## 4. Cost

For ranking satisfying candidates:

```
cost(A) = (instruction_count(A), estimated_cycles(A))
```

ordered lexicographically. `estimated_cycles` uses a single-issue in-order RV64 latency
table summed over the static schedule — no bypass, no cache, no branch prediction.

## 5. Dataset

For v1:

- Sample functional specs from `spec.generator.sample_spec` with bounded AST depth
  (≤ 5) and `n_params ∈ [1, 4]`. (Relational sampling is reserved for milestone 2 —
  the supervised GNN needs functional teacher signals to bootstrap.)
- For each sampled spec `S`:
  - Reject if trivially constant-foldable or constant across probes.
  - Generate 1000 random + boundary input tuples avoiding `pre`-violations and UB; cache
    as `data/v1/<spec-id>/io.json`.
  - Compile its functional body via `spec.lower_to_c → riscv64-linux-gnu-gcc -O2 -S` and
    `clang --target=riscv64-linux-gnu -O2 -S`; cache both as the reference asm baselines.
- Split by **spec-template family** (AST skeleton with leaves abstracted) to prevent
  the model from memorizing rewrites of the same skeleton.

Target dataset size for v1: 50k specs (functional). Relational specs from hand-written
``.spec`` files seed a held-out qualitative-evaluation set.

## 6. Evaluation metrics

On the held-out split:

| Metric | Definition |
|---|---|
| `assembles_rate` | candidate parses + assembles. |
| `io_satisfies_rate` | passes `verify_io` (Post holds on N sampled inputs satisfying Pre). |
| `smt_satisfies_rate` | `verify_smt` returns `equivalent=True` within timeout. |
| `beats_gcc_O2_rate` | (smt-satisfies AND `instruction_count < instruction_count(gcc_O2)`); for relational specs where gcc has no reference, this slot is N/A. |
| `cycle_ratio_geomean` | geometric mean of `cycles(A)/cycles(gcc_O2)` over the SMT-satisfying functional set. |

Greedy decoding and sampled-with-verifier (K = 16) are reported separately.

## 7. Baselines (must be in milestone 1)

1. **`gcc -O2`** and **`clang -O2`** for functional specs — upper bound on imitation,
   and the perf bar.
2. **Seq2seq baseline**: same spec encoder, transformer decoder emitting asm tokens
   directly. Isolates whether graph-structured generation actually helps.
3. **Random asm + verifier**: lower bound — pure search, no learning.
4. **Enumerative SyGuS-style synthesizer** (optional, milestone 2 cross-check) for the
   relational specs gcc can't produce.

If the GNN generator does not beat the seq2seq baseline on functional `beats_gcc_O2_rate`
by end of milestone 2, the structural-prior claim does not hold and the architecture needs
revisiting.

## 8. Out of scope for v1

- Natural-language input.
- Control flow in candidate code (straight-line basic blocks only).
- Memory, function calls, multiple returns.
- RISC-V vector extension (RVV), floating point, atomics, custom ISAs.
- Learned lowering (graph → asm stays deterministic in v1).
- Execution-feedback RL on the generator (deferred to milestone 2).
- Hardware-measured cycle counts (estimated only in v1).
- User-defined predicates and axioms in the spec DSL — built-in predicates only
  (`Cmp` family). Reserved as a future extension hook.

## 9. Open decisions

- Generative-GNN family: autoregressive node-addition vs diffusion vs autoflow.
  Decision target: end of workstream A.
- Relational-spec generation: how to sample non-trivial postconditions that aren't
  trivially `r == f(x)`. Decision target: late milestone 1.
- SMT-verification coverage of RV64GC: which subset is in scope for v1's formal gate.
  Decision target: when expanding past the current MVP.
