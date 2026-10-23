# Exception handling in IRStructurizer

Status: design and working prototype on branch `eh-tryop`. The prototype lifts
`EnterNode`-based try/catch to a structured `TryOp`, lowers it back to `IRCode`,
and has an `exceptions=:trap` mode for consumers without exceptions.

## Summary

- A try becomes a **`TryOp`** with two regions. The **body** is the protected code.
  The **handler** is the catch block, and it runs with the try's handler popped. The
  op sits at the `EnterNode`'s SSA index. Its results are the values of the single
  continuation, read with `getfield` as for `IfOp`.
- **`:leave` disappears.** Leaving the body pops the handler. Normal completion is a
  `YieldOp`. An exit from a nested region (an `IfOp` arm) is a new terminator,
  **`LeaveOp`**, which works like `BreakOp` but targets the innermost enclosing try
  body. This is the structured `:leave`, and it lowers to a Wasm `br` out of a
  `try_table`.
- **PhiC values become handler block arguments.** `TryOp.upsilons[k]` lists the
  `UpsilonNode` statements that feed argument `k`, and those statements stay where
  they are. A backend gives each argument a local, so an upsilon is a `local.set`
  and the handler entry reads it with `local.get`.
- **`:the_exception` and `:pop_exception` stay as statements.** They operate on
  Julia's exception stack, which is runtime state rather than control flow.
  `pop_exception %t` names the `TryOp` whose entry depth it restores.
- **Regions come from a handler-stack dataflow** (Julia's `compute_trycatch`).
  Normalization then gives every try a single continuation. Every mux it, or any
  other pass, builds joins only edges that hold the same handlers. A loop that
  contains a try contains all of it. So a throwing statement never changes which
  handler it unwinds to, and LLVM's "unwind destination mismatch" cannot happen.
- **`exceptions=:trap`** drops handlers at ingest (a throw traps). The result is
  plain SCF, except for `EnterNode`s that carry a ScopedValue scope, which stay as
  handler-less `TryOp`s.

## 1. What Julia's IR gives us

| construct | meaning |
|---|---|
| `%e = EnterNode(catch_dest[, scope])` | Block terminator. It pushes a handler and falls through into the protected region, with an implicit edge to block `catch_dest` (`compute_basic_blocks` also gives the catch block a virtual predecessor `0`). `catch_dest == 0` is a scope-only enter: the optimizer proved the body nothrow but kept the scope (`Compiler/src/optimize.jl`). |
| `Expr(:leave, %e...)` | Pops one handler per non-`nothing` argument. It ends its block (`ir.jl`, `basic_blocks_starts`). A `return` out of nested trys pops several handlers at once. |
| `Expr(:pop_exception, %e)` | Restores the exception stack to its state at enter `%e`. It runs on every normal exit from a catch block, and can run after an enclosing try was already left (`leave %outer` comes before `pop_exception %inner`). |
| `Expr(:the_exception)` | Reads the current exception. |
| `%u = UpsilonNode(v)` / `PhiCNode(%u...)` | A value live into a catch block. Each assignment in the protected region (and one before the enter) feeds an upsilon. The catch block's `PhiCNode` reads whichever of its upsilons ran last. |

The handler stack is balanced: every block has one stack of active handlers on
entry. `Compiler.compute_trycatch` (`inferencestate.jl`) computes it and asserts
"unbalanced try/catch" otherwise. Everything below builds on that invariant.

```
 1 ─ %1 = enter #4                         %1 = try -> Tuple{Int64}
 2 ─ %2 = invoke g(_2)                     ├ body:
 └──      leave %1                         │   %2 = invoke g(_2)
 3 ─      return %2                        │   yield %2
 4 ┄      pop_exception %1                 ├ catch:
 └──      return -1                        │   pop_exception %1
                                           └   return -1
                                           %7 = getfield(%1, 1)
                                           return %7
```

## 2. Prior art

**Structured IRs that have a try construct**

- **ClangIR `cir.try`** (`clang/include/clang/CIR/Dialect/IR/CIROps.td`). It has a
  `try_region` plus variadic `handler_regions`. Each handler receives a
  `!cir.eh_token` block argument, and calls inside the try are only marked. To
  lower, `CIRTryOpFlattening` (`clang/lib/CIR/Dialect/Transforms/FlattenCFG.cpp`)
  first flattens everything nested, because an unwind edge may not cross a region
  boundary. It then rewrites calls to `cir.try_call` with normal and unwind
  successors. This is the closest relative of our `TryOp`: handlers as regions,
  with the exception payload as a block argument.
- **ILSpy's ILAst** (`ICSharpCode.Decompiler/IL/Instructions/TryInstruction.cs`)
  has `TryCatch`/`TryFinally`/`TryFault` over single-entry `BlockContainer`s.
  `Branch` stays inside a container and **`Leave` exits it**, the same model as
  `LeaveOp`. `BlockBuilder.CreateContainerStructure` builds trys before loop
  detection and treats them as opaque. CIL guarantees nested ranges.
- **Binaryen** (`src/wasm.h`) has legacy `Try` (catch bodies as child expressions,
  payload through a `Pop` that must come first, `src/ir/eh-utils.h`) and `TryTable`
  (`catchTags`/`catchDests`/`catchRefs`: handlers are branch targets).
  `TranslateEH.cpp` rewrites one into the other with nested blocks and trampolines.
  `StripEH.cpp` turns `throw` into `unreachable` and a try into its body. ReReloop
  rejects EH.
- **The Wasm exnref proposal**
  ([Exceptions.md](https://github.com/WebAssembly/exception-handling/blob/main/proposals/exception-handling/Exceptions.md),
  rationale in [issue #280](https://github.com/WebAssembly/exception-handling/issues/280)).
  `try_table` catch clauses name labels of enclosing blocks: `catch` pushes the tag
  payload, `catch_ref` the payload and an `exnref`, `catch_all_ref` an `exnref`
  alone, and `throw_ref` rethrows. A handler is ordinary code after a block's
  `end`, so the protected region only has to be a properly nested block.

**CFG-level representations (handler as an edge)**

- **MLIR**: SCF and `cf` have no EH, and `CFGToSCF.cpp` ignores unwinding. The
  `llvm` dialect has `llvm.invoke` (normal and unwind successors with operands,
  checked in `InvokeOp::verify`), `llvm.landingpad` and `llvm.resume`. `emitc`,
  `wasmssa` (`WasmSSAOps.td`) and Flang's FIR have no EH ops.
- **Swift SIL `try_apply`** (`docs/SIL/Instructions.md`) has `normal bb(%r)` and
  `error bb(%e)` successors. The error is a block argument, with no unwinding.
- **Cranelift `try_call`** (`cranelift/codegen/meta/src/shared/instructions.rs`,
  `ir/exception_table.rs`). An exception table maps a tag to a block call. The
  results and the exception payload arrive as edge-created block arguments
  (`BlockArg::TryCallRet`/`TryCallExn`).
- **Graal** has `WithExceptionNode`/`InvokeWithExceptionNode` with an exception
  successor and `ExceptionObjectNode`. Web Image puts one `try_table` around each
  throwing call (`WasmIRWalker.lowerWithExceptionExnRef`), so ranges never nest.
- **V8 and SpiderMonkey** keep side tables (Ignition `HandlerTable`;
  `TryNoteKind`). Their optimizers turn `try_table` back into per-call
  exception edges (Turboshaft `CheckForException`, Liftoff `EmitLandingPad`).

**Recovering structure (where non-nesting is fought)**

- **LLVM's WebAssembly backend** (`llvm/lib/Target/WebAssembly/`).
  `WebAssemblyExceptionInfo` treats an EH pad plus what it dominates as a scope,
  nested like loops. `WebAssemblyCFGStackify::placeTryTableMarker` places `block` +
  `try_table` over the linearized CFG. Because the CFG is linearized first, a call
  can end up inside the try of a pad it does not unwind to (a *call-unwind
  mismatch*, `fixCallUnwindMismatches`), and an exception a pad does not catch can
  reach a pad that does (a *catch-unwind mismatch*, `fixCatchUnwindMismatches`).
  The fixes wrap the code in an extra `try_table (catch_all_ref $tramp)` whose
  trampoline `throw_ref`s at the right depth (legacy EH: `try ... delegate`).
- **TeaVM** (`core/src/main/java/org/teavm/`) keeps a per-block try/catch list.
  Its decompiler (`ast/decompilation/Decompiler`) closes trys at structured-block
  ends and reopens them, so a range that crosses loops becomes several trys that
  share a handler. The handler is a `break` target, and the WasmGC backend emits
  `try_table`.
- **Vineflower, CFR and Procyon** normalize exception tables before structuring:
  they merge or split ranges until each has one entry
  (`ExceptionDeobfuscator.splitExceptionRange`, CFR `ExceptionAggregator`), drop
  ranges that contain their own handler, and undo javac's duplicated `finally`.
- **dart2wasm, Kotlin/Wasm and J2Wasm** start from source structure. A finally
  block is either duplicated per exit (Kotlin `FinallyBlocksLowering`, dart2wasm
  `visitTryFinally`) or run once and left through a continuation index and a switch
  (J2CL `ImplementFinallyViaControlFlow`, the dart2wasm state machine). Julia
  lowering already duplicates finally, so the switch form is just a multi-exit
  region to the structurizer.

**What we take from this**

1. Compute handler membership per block first, as a dataflow (like V8's handler
   stack, Wasmtime's `HandlerState` and TeaVM's per-block lists), and derive
   regions from it. Do not place regions over a linearized CFG and repair them
   afterwards, as LLVM must.
2. Build try regions before loop and branch normalization, and treat them as
   atomic in the enclosing structure (ILSpy, Procyon, Vineflower).
3. A structured exit out of a try is an ordinary multi-level exit: `Leave` in
   ILSpy, `br` past `try_table` in Wasm.
4. Values reach a handler through mutable state (locals, spills, Julia's upsilons),
   not SSA edges. The handler's own inputs can still be block arguments, as in
   CIR, SIL and Cranelift.
5. Dropping EH is easy: a try becomes its body and the handler is dead (`StripEH`).

## 3. The design

### 3.1 `TryOp`

```julia
mutable struct TryOp <: ControlFlowOp
    body::Block                      # protected region
    handler::Union{Block, Nothing}   # catch region; `nothing` for a scope-only enter
    upsilons::Vector{Vector{Int}}    # per handler arg: SSA indices of its UpsilonNodes
    scope::IRValue                   # EnterNode scope (ScopedValues), or `nothing`
end
```

- The op's SSA index is the `EnterNode`'s, so `pop_exception %t` keeps naming it.
  Its type is `Tuple{...}` of the continuation's live block arguments, extracted
  with `getfield` statements as for `IfOp`. It is `Tuple{}` for a continuation
  without values and `Nothing` for a try whose continuation is outside its region
  or absent.
- **Body.** It ends in `YieldOp(values)` for normal completion, or in
  `ReturnNode()` when it ends in a throw. Nested regions leave it with `LeaveOp`.
  It never holds a `BreakOp`, `ContinueOp` or value `return`: those always come
  after the try's continuation (§4.3).
- **Handler.** It runs with this try's handler popped and the exception pushed. It
  ends in `YieldOp`, `ReturnNode(v)`, `ReturnNode()` (after `rethrow`), or
  `LeaveOp` when it sits in an outer try's body.
- `blocks(op)` is `(body, handler)`, or `(body,)` for a scope-only try. Generic
  walkers (`walk`, `eachblock`, `uses`, `replace_uses!`, `promote_loops!`) see
  both regions.

### 3.2 `LeaveOp` and region exits

`LeaveOp(values)` exits the innermost enclosing `TryOp` *body* and continues after
the op with `values` as its results. It can cross `IfOp` arms and the handler of an
inner `TryOp` (whose handler is already popped). It never crosses a loop or another
try body. `validate_terminators` checks this.

| exit | structured form | Julia IR (unstructurize) | Wasm |
|---|---|---|---|
| body completes | `yield v` in the body | `leave %t; goto merge` | `br $cont` |
| exit from a nested arm | `leave v` | `leave %t; goto merge` | `br $cont` (deeper label) |
| handler completes | `yield v` in the handler | `goto merge` | fall through / `br $cont` |
| `return` / `break` / `continue` out of a try | `yield` a discriminator; the return or loop edge follows the op | `leave`, then the original edge | `br $cont`, then `return`/`br` |
| throw inside the body | the throwing call, then `return` (unreachable) | unchanged | unchanged (caught by `try_table`) |

`return`, `break` and `continue` out of a try are not terminators inside it.
Normalization routes every exit of the try to one continuation (§4.2), and the
`IfOp` dispatch after the `TryOp` performs the original transfer. This matches
how the package already gives loops a single exit, and keeps `TryOp` opaque to
loop promotion.

Example (`try; if x > 0; if x > 5; return 3x; end; y = g(x) else y = h(x) end; y + 1
catch; -7 end`):

```
%1 = try -> Tuple{Int64, Int64, Int64}
├ body:
│   %24 = if (0 < x) -> Tuple{Int64}
│   ├ then:
│   │   %22 = if (5 < x) -> Tuple{Int64}
│   │   ├ then:  %6 = mul_int(x, 3); leave 0, %6, undef      # exits the try
│   │   ├ else:  %9 = g(x); yield %9
│   │   yield getfield(%22, 1)
│   ├ else:  %11 = h(x); yield %11
│   %13 = add_int(getfield(%24, 1), 1)
│   yield 1, undef, %13
├ catch:
│   pop_exception %1
└   return -7
if getfield(%1, 1) === 0 → return getfield(%1, 2) else → return getfield(%1, 3)
```

### 3.3 Values into the handler

Each `PhiCNode` of the catch block becomes a handler block argument, and
`upsilons[k]` lists the `UpsilonNode` statements that feed it. Those statements
stay in place, in the body or just before the op, and are ordinary uses of their
operand (`walk_uses!`, validation and substitution all treat them so). This keeps
the handler SSA-clean: no reference into the body. It also tells a backend exactly
what to do:

- each handler argument `k` is a local `$u_k`;
- `UpsilonNode(v)` listed for argument `k` becomes `local.set $u_k v`. The reverse
  map, from upsilon to `(TryOp, k)`, comes from one walk over the `TryOp`s;
- handler entry reads `local.get $u_k`.

This is also how Julia's LLVM codegen implements upsilon/phic (an alloca per
PhiC), and what Liftoff, Graal and Cranelift do with locals or spills.

### 3.4 The exception stack

`Expr(:the_exception)` and `Expr(:pop_exception, %t)` stay as statements. Their
operand is the exception stack, which is also read by `rethrow()` and
`current_exceptions()`, runtime calls the structurizer cannot see. So the
exception is not made a handler argument. A backend models the stack directly:
push on handler entry, save the depth at each `TryOp` that a `pop_exception`
names, and restore it at `pop_exception %t`.

`pop_exception`'s operand is a token, exempt from SSA scoping. Julia emits
`leave %outer` *before* `pop_exception %inner` when a catch returns out of an
enclosing try. After the lift, that pop sits after the outer `TryOp`, where the
inner enter no longer dominates it in lowered IR. `retarget_pop_exceptions!`
rewrites such a pop to the outermost enclosing try whose enter saw the same
exception-stack state (the second component of `compute_trycatch`). Both name the
same state, and the outer enter dominates.

### 3.5 Scoped values

`EnterNode(dest, scope)` (from `@with`) becomes a `TryOp` with `scope` set.
`EnterNode(0, scope)` becomes a `TryOp` with `handler === nothing`: entering sets
the scope and leaving restores it. `:trap` keeps these, because dropping them
would change what `ScopedValue` reads return.

### 3.6 Lowering to Wasm (exnref)

```wat
;; %t = try -> (T...)   body / handler(%a_1..%a_n) / upsilons
block $cont (result T...)
  block $h (result (ref $jl_exn) exnref)        ;; one tag for Julia objects
    try_table (catch_ref $jl_tag $h)            ;; (+ catch_all_ref for foreign throws)
      ;; body; UpsilonNode(v) feeding a_k:  local.set $u_k v
      ;; yield v / leave v:                 br $cont v
    end
    unreachable                                 ;; body never falls through
  end
  ;; handler: push the exception on the Julia exception stack, keep the exnref
  ;; for rethrow (throw_ref), %a_k = local.get $u_k
  ;; yield v: falls through to $cont; return v: return; rethrow(): throw_ref
end
;; getfield(%t, i) are the block results
```

The handler is outside the `try_table` but inside `$cont`, which is the Julia
semantics: the handler runs with the try's handler popped. Since `LeaveOp` never
crosses a loop, every `br` it becomes is a forward branch to an enclosing block.

### 3.7 Consumers that trap on throw

- `StructuredIRCode(ir; exceptions=:trap)` / `code_structured(...; exceptions=:trap)`.
  At ingest, an `EnterNode` without a scope becomes a plain fallthrough, its
  `:leave`s and all `UpsilonNode`s are dropped, and the now-unreachable catch
  blocks are pruned. The joins that lose a predecessor have their single-edge
  phis folded. The result is the same SCF the package produced before EH support,
  with no `TryOp` except scope-only ones. A throw propagates out of the function,
  which a trapping target turns into a trap.
- Dropping handlers after the fact also works. Treat `TryOp` as a block that runs
  its body, with a `LeaveOp` as `br` to its end, and ignore `handler` and
  `upsilons`. A consumer without multi-level exits should use `:trap` instead.

## 4. Recovering the regions

Implementation: `src/structurize/exceptions.jl`, plus small hooks in ingest,
normalize, the walk and unstructurize.

### 4.1 Handler stacks and the two regions

`ingest` turns `EnterNode` into an `MEnter` terminator: a body edge, the catch block
id, and the scope. It then makes every `:leave` a block of its own that pops one
handler (`split_leaves!`, innermost first, naming the enter it pops).
`handler_stacks` is the forward dataflow over `MEnter` pushes and `:leave` pops,
and it rejects unbalanced input. For a try entered at block B with stack σ:

- **R (body)**: blocks whose entry stack contains the try. Every edge out of R
  starts at a leave block and lands on a block with stack σ.
- **C (handler)**: blocks dominated by the catch block whose stack extends σ and
  that run before the catch's `pop_exception`, minus leave blocks that pop a
  handler in σ. A block runs before the pop when its exception state (the second
  component of `compute_trycatch`, `exception_states`) is this catch's, or that of
  a catch nested in it. The leave blocks belong to the enclosing body: a handler
  that exits an outer try first reaches its leave block, which is then a
  continuation of *this* try.

  Dominance alone is not enough when the body cannot complete normally (it always
  throws, or loops forever as in `Base.wait_forever`). The catch block then
  dominates all the code after the handler, including each continuation that
  normalization builds. The handler would absorb every mux built for its exits,
  and normalization would never converge.

Because stacks are balanced, the regions of different trys nest properly. An inner
try's R and C lie inside the outer R (or C), and a natural loop with header in R
lies inside R. Try-vs-try overlap, the TeaVM/Vineflower problem, cannot arise in
Julia IR. What remains is try-vs-loop and try-vs-branch interaction, handled below.

### 4.2 Normalization (`normalize_one_try!`, first in the `normalize_cf!` fixpoint)

1. **Single continuation.** If the edges out of R ∪ C reach more than one block, or
   reach a loop header, route them through one `EdgeMux` K. The dispatch to the
   original targets (the return block, the loop latch, the after-try code) runs
   after the `TryOp`, outside every handler of the try. All these edges hold
   stack σ, so the mux is valid.
2. **Reduce form.** A value defined in R ∪ C and read in a block K dominates becomes
   a new block argument of K, and those reads are rewritten (`reduce_try!`). This
   is the try analogue of `reduce_loop!`. Afterwards nothing escapes the `TryOp`
   except through its results.

Since every later mux, whether a latch, continuation or pre-header, joins edges
whose sources share a handler stack, no pass can merge paths with different
handlers. `verify_try_structure` checks the invariants after the fixpoint and
throws `UnstructuredControlFlowError` instead of lifting a shape it cannot
represent.

### 4.3 Loops

- **Loop inside a try** needs no special handling. Its blocks lie in R, its exits
  stay in R, and its own latch is in R.
- **Try inside a loop.** `natural_loops_m` grows every loop that contains B by all
  of R ∪ C (`extend_loops_over_trys!`). Otherwise two things go wrong. A throwing
  dead end in R cannot reach the back edge, so it is not in the natural loop, and
  the latch would route it out of the loop, out of the try, past the handler. And
  a handler that leaves the loop would make the exceptional edge itself a loop exit
  that the latch would redirect. With the try inside, the loop's exits leave from
  the try's continuation K, which is in the loop and holds no handler of the try.
- **`break`/`continue`/`return` out of a try** go to K (§4.2 rule 1), and the
  post-try dispatch performs them. `LeaveOp` therefore never crosses a loop, and
  `promote_loops!` is unaffected: there are no loop exits inside a `TryOp`.
- A value `return` inside a handler stays where it is when no outer try is active,
  even inside a loop. `return` is a structured function exit. This differs from
  non-EH code, where a return in a loop is routed through the latch.

### 4.4 Branches inside a try

A branch in R may have one arm leave the try while its other arm continues in R.
That edge is not a continuation edge of the branch: the edge changes the handler
stack, so muxing it with an in-region edge would be wrong. `branch_continuation`
skips any edge into a block with a shallower handler stack than the branch (the
`depth` keyword), the same way it skips loop exits, in both normalization and the
lift. In the lift, `resolve_try_exit!` turns that edge into a `LeaveOp` carrying
the edge's operands into K. This also covers a throwing merge next to an exit,
which a funnel-style "try latch" could not: dead ends never reach the latch.

### 4.5 Overlap and nesting corner cases

- **Multi-handler `:leave`** (`return` out of nested trys) is split into one
  leave block per handler, so each region boundary is a single edge.
- **A handler that leaves an outer try** (`catch; return ... end` inside another
  try): the outer leave block is excluded from C (§4.1). It becomes the inner
  try's continuation, and the outer exit happens after the inner `TryOp`, as a
  `LeaveOp` or `YieldOp` of the outer body. The trailing `pop_exception %inner` is
  retargeted (§3.4).
- **finally**: Julia lowering duplicates the finally code onto the normal path and
  the exceptional path (`catch; <finally>; rethrow()`). When the try has several
  exits it dispatches on a selector after a shared copy. Both are ordinary CFG to
  the structurizer, giving a handler that ends in `rethrow` (`ReturnNode()`) and a
  multi-exit continuation.
- **A body that cannot complete normally.** Every exit of the try is a handler
  exit, and the code after the handler is a continuation like any other (§4.1),
  even though the catch block dominates it.
- **Catch block reached by ordinary control flow** never comes out of Julia
  lowering. It is rejected.
- **Variables undefined on the catch path.** A variable never assigned before a
  throw gets a `Union{}`-typed `PhiCNode`, and phis carry it around merges and
  loops. Julia reads it only behind `throw_undef_if_not`. `undef_bottom_values!`
  turns every edge operand that carries such a value into `Undef`, and an upsilon
  of one into an undefined upsilon. The lift then never builds a `Union{}` loop
  carry or result. Relatedly, a loop-header phi whose entry value is `Undef` (a
  variable first assigned inside the loop, e.g. in a catch) becomes a carry with an
  `Undef` initial value instead of an error.

### 4.6 Why there is no unwind-destination mismatch

LLVM's mismatches come from placing try markers over a *linear layout*: code that
lands between the markers is protected whether or not it should be. Here regions
are the exact sets of blocks with the handler active, and the lift never moves a
statement out of its block. The only code normalization adds is mux dispatch
(`===` comparisons and gotos), which cannot throw. Two conditions remain, and the
dataflow guarantees them:

1. A block is never placed in a region with a different handler.
2. No mux joins edges with different handler stacks.

`verify_try_structure` re-checks both. A *catch-unwind* mismatch (an exception the
handler does not catch) cannot occur either. A Julia `catch` catches everything,
and type filtering is ordinary code in the handler followed by `rethrow`.

## 5. Alternatives considered

- **Handler as a branch target** (exnref-style `TryOp(body)` with an exceptional
  exit, the handler structured by the ordinary IfOp/mux machinery after it). Both
  research surveys found this the most common choice (TeaVM, Graal, dart2wasm,
  LLVM). It removes handler-vs-outer-try nesting (§4.5) for free, but every try
  then yields a "caught" discriminator and the handler becomes an `IfOp` arm
  dispatched after it. That is noisier for the common `try … catch … end`, and it
  hides the handler's extent from consumers that want to drop it. The region form
  still lowers to the branch-target Wasm form directly (§3.6). If needed, it can be
  produced from a `TryOp` by moving the handler after the op.
- **A "try latch" instead of `LeaveOp`.** Funnel every exit of R through one block
  inside R, as the loop latch does, and keep only `YieldOp`. This needs no new
  terminator, but it fails when a throwing merge sits next to an exit: dead ends
  never reach the funnel. It also adds a discriminator for every early `return`
  inside a branch. `LeaveOp` matches ILSpy's `Leave` and Wasm's `br`.
- **Keep `PhiCNode` statements in the handler.** That is faithful to Julia, but the
  handler would reference body SSA values outside its scope. Block arguments plus
  `upsilons` say the same thing and keep the regions SSA-clean.
- **Explicit `:leave` statements inside regions.** Exits are already structural,
  and an explicit leave would let a body pop its handler midway, which no
  structured target can express.

## 6. Prototype status

**Handled** (each is round-tripped by `test/exceptions.jl` with and without loop
promotion and under `:trap`, on throwing and non-throwing inputs). The lowered code
must pass `verify_ir`, and every run must match the direct call. That covers the
result or exception (type and message), the side-effect log (cleanup that runs
once, in order), and the exception-stack depth after a return. Under `:trap` a run
may instead end in the first exception the direct call throws.

- try/catch
- catch with `e` (`the_exception`)
- try/finally
- try/catch/finally with `return` in each part
- rethrow, including conditional rethrow
- nested exceptions of one type, told apart by message
- values live into the catch (upsilon/phic)
- nested trys, three levels deep
- a try inside a catch
- sequential trys
- a `return` in the try that pops one or two handlers
- an early `return` nested in branches that merge again inside the try
- a throwing merge next to an exit
- a try inside a loop with `break`, `continue`, `return` and a throw inside the try
- `continue` from a catch
- a loop inside a try, including a throw inside the loop
- a catch returning out of a loop and an enclosing try
- finally with `break`
- `@goto` out of a try
- a try in a `while` condition
- ScopedValues `@with`, including the scope-only `EnterNode(0, scope)`

- ScopedValue reads inlined into the function (not hidden behind `@noinline`)
- a body that cannot complete normally, inside a loop and with two handler exits

The battery outside the test suite also covers a try returning a `Union`
(`p3`).

Beyond the hand-written shapes, every Base/stdlib method with a concrete
signature was swept: 93 have an `EnterNode` after optimization on Julia 1.12, and
103 on 1.13. In both modes (186 and 206 lifts), all of them lift with full
validation, unstructurize and pass `Compiler.verify_ir`. The first version of this
sweep (77 methods, 1.12) had 10 lifts fail the yield-type check, which the
narrowing `PiNode` fixes (§6, fixes). On 1.13, `Base.wait_forever` did not
converge until the handler boundary at `pop_exception` (§4.1).

**Not handled or known limitations**

- Irreducible control flow that crosses a try boundary is untested. A mux that
  would join different handler stacks is caught by `verify_try_structure`, which
  throws, rather than miscompiled.
- `pop_exception` retargeting covers the "leave outer, then pop inner" shape that
  Julia emits. A pop after leaving a try from a handler nested two catch levels
  deep is still correct as structured IR (the token semantics hold), but its
  lowered `IRCode` may fail Julia's dominance check.
- Outputs can be verbose: a try with several exit kinds gets a discriminator
  dispatch after it, as multi-exit loops already do.
- The try analyses are recomputed on every normalization step. That is cheap for
  ordinary functions and quadratic-ish in the number of trys for huge ones. Code
  without trys does no extra work.
- Only tested on Julia 1.12.

**Fixes to pre-existing code made along the way** (exposed by EH shapes, not
EH-specific)

- `normalize_one_irreducible!` read `CFGReachability.irreducible` for unreachable
  blocks. Compiler allocates that `BitVector` uninitialized and only writes it for
  reachable blocks, so any MCFG with a dead block could randomly hit "irreducible
  SCC with no entry block". Unreachable blocks are now skipped.
- `anchor_line!` ignores a self-anchor. `bind_single_pred_args!` could create one,
  which made `source_location` (and printing) loop forever.
- `merge_info` widens a `Union{}`-typed merge argument's result slot to `Any`.
  `Union{}` cannot be a `Tuple` field.
- Julia types a phi by what inference knows on each incoming edge, so a phi can be
  narrower than the value feeding it. Examples are an inlined `ScopedValue[]` read,
  a try returning `Union{Nothing,Int}` read behind a guard, and `x::Any` behind
  `x isa Int`. Julia's verifier allows this, but the yield then failed
  `validate_terminators`' type check, on `main` too. `merge_values` now narrows such
  an operand with a `PiNode` in the region that takes the edge, where the refinement
  holds.

## 7. Future work

- A `TryOp` → branch-target form helper, for consumers that want the exnref shape
  explicitly.
- Expose the reverse upsilon map (`upsilon => (TryOp, arg)`) as an API helper.
- Typed catch: if Julia ever gets typed handlers, `TryOp` would grow per-type
  handler regions, as in ClangIR. A catch-unwind mismatch then becomes possible and
  needs LLVM's trampoline fix.
- Cache the try analyses across normalization steps.
