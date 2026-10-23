#=============================================================================
 Exception handling: EnterNode / :leave / PhiC+Upsilon to TryOp / LeaveOp.

 Every shape round-trips (structurize, unstructurize, execute) on inputs that do
 and don't throw, comparing results, thrown exceptions (type and message), the
 `eh_log` side effects, and the exception-stack depth (`eh_check`). Structural
 tests pin the TryOp shape: handler block args from PhiC, LeaveOp from nested exits,
 a single continuation, the scope operand, and `exceptions=:trap`.
=============================================================================#

using IRStructurizer: TryOp, LeaveOp
using Base.ScopedValues

const EH_LOG = Any[]
@noinline eh_log(@nospecialize(x)) = (push!(EH_LOG, x); nothing)
# An exception, comparable across runs: its type and message.
eh_payload(@nospecialize(e)) = (typeof(e), sprint(showerror, e))
"""Throw `e`, logging it first. The test functions throw through this, so a log
also records the throws that a handler catches."""
@noinline eh_throw(@nospecialize(e)) = (push!(EH_LOG, (:throw, eh_payload(e))); throw(e))
@noinline eh_g(x::Int) = x > 10 ? eh_throw(ErrorException("big")) : x + 1
@noinline eh_h(x::Int) = x < 0 ? eh_throw(DomainError(x)) : x * 2
const EH_SV = ScopedValue(1)
@noinline eh_readsv() = EH_SV[]

"""The outcome of a call: `(:ok, value)` or `(:threw, payload)`, the `eh_log` calls it
made, and the exception-stack depth after a normal return (-1 after a throw). A
lost or mistargeted `pop_exception` leaves exceptions on the stack."""
function eh_outcome(f, args...)
    empty!(EH_LOG)
    depth = -1
    result = try
        v = f(args...)
        depth = length(current_exceptions())
        (:ok, v)
    catch e
        (:threw, eh_payload(e))
    end
    return (; result, log = copy(EH_LOG), depth)
end

"""Lower `sci` back to `IRCode`, check it with `verify_ir`, and compile it. This
runs outside `eh_outcome`, so a lowering failure is a test error, never an
exception that `:trap` mode would skip."""
function eh_lower(sci::StructuredIRCode)
    ir = CC.IRCode(sci)
    CC.verify_ir(ir)
    ir = CC.copy(ir)
    ir.argtypes[1] = Tuple{}
    @static if VERSION >= v"1.12-"
        ir.debuginfo.def = Symbol("unstructurized")
    end
    return Core.OpaqueClosure(ir)
end

"""
    eh_check(f, inputs; exceptions=:structured, promote=true, argtypes, optimize_until,
             internal_throws=false) -> Vector{String}

Structurize `f` (at the types of the first input tuple, or `argtypes`), lower it
back with `eh_lower`, and run it on every input tuple. Each outcome must equal the
direct call's (`eh_outcome`). Under `exceptions=:trap` a handler that would have
caught the first throw is gone, so a run may end in that throw instead: its log
must match the direct call's up to and including that throw, which `eh_throw`
logged. With `internal_throws`, the throw may also come from Julia itself (not
`eh_throw`). Returns the mismatches.
"""
function eh_check(f, inputs; exceptions=:structured, promote=true,
                  argtypes=Tuple{map(typeof, first(inputs))...},
                  optimize_until=nothing, internal_throws=false)
    ir, _ = only(code_ircode(f, argtypes; optimize_until))
    sci = StructuredIRCode(ir; exceptions, promote)
    return eh_compare(f, eh_lower(sci), inputs; exceptions, internal_throws)
end

"""Run the lowered closure `oc` and `f` on every input, as `eh_check` describes."""
function eh_compare(f, oc, inputs; exceptions=:structured, internal_throws=false)
    problems = String[]
    for args in inputs
        expected = eh_outcome(f, args...)
        got = eh_outcome(oc, args...)
        isequal(expected, got) && continue
        if exceptions === :trap && got.result[1] === :threw
            n = length(got.log)
            isequal(got.log, expected.log[1:min(n, end)]) &&
                (internal_throws || (n > 0 && isequal(got.log[n], (:throw, got.result[2])))) &&
                continue
        end
        push!(problems, "$(nameof(f))$(args): expected $(expected), got $(got)")
    end
    return problems
end

"""`eh_check` at `Tuple{Int}`, on single-integer inputs."""
eh_roundtrip(f, inputs; kwargs...) = eh_check(f, [(x,) for x in inputs]; kwargs...)

function eh_ops(blk::Block, T)
    found = Any[]
    for (_, e) in blk.body
        e.stmt isa T && push!(found, e.stmt)
        if e.stmt isa ControlFlowOp
            for b in IRStructurizer.blocks(e.stmt)
                append!(found, eh_ops(b, T))
            end
        end
    end
    return found
end
eh_ops(sci::StructuredIRCode, T) = eh_ops(sci.entry, T)

function eh_terminators(blk::Block, T)
    n = blk.terminator isa T ? 1 : 0
    for (_, e) in blk.body
        e.stmt isa ControlFlowOp || continue
        for b in IRStructurizer.blocks(e.stmt)
            n += eh_terminators(b, T)
        end
    end
    return n
end

#----------------------------------------------------------------------------- shapes

eh_basic(x) = try eh_g(x) catch; -1 end
eh_exc(x) = try eh_g(x) catch e; e end
function eh_finally(x)
    r = 0
    try
        r = eh_g(x)
    finally
        r += 100
    end
    r
end
function eh_nested(x)
    try
        a = eh_g(x)
        try
            eh_h(a - 8)
        catch
            a - 5
        end
    catch
        -1
    end
end
function eh_upsilon(x)
    y = 0
    try
        y = x + 1
        y = eh_g(y)
        y = eh_h(y - 20)
    catch
        return y
    end
    y
end
eh_rethrow(x) = try eh_g(x) catch; x > 15 && rethrow(); -x end
function eh_return_in_try(x)
    try
        y = eh_g(x)
        if y > 5
            return y
        end
        eh_h(y - 3)
    catch
        0
    end
end
function eh_mixed_exit(x)   # an exit nested in an arm whose sibling stays in the try
    try
        if x > 0
            if x > 5
                return x * 3
            end
            y = eh_g(x)
        else
            y = eh_h(x)
        end
        y + 1
    catch
        -7
    end
end
function eh_dead_merge(x)   # an exit next to a throwing merge inside the try
    try
        if x > 3
            z = 1
        elseif x > 1
            return eh_g(x)
        else
            z = 2
        end
        eh_throw(ErrorException("boom $z"))
    catch
        -9
    end
end
function eh_nested_return(x)   # `return` pops two handlers; pop_exception after leave
    try
        try
            return eh_g(x)
        catch
            return eh_h(x)
        end
    catch
        return 0
    end
end
function eh_try_in_loop(n)
    s = 0
    for i in 1:n
        try
            s += eh_g(i)
        catch
            s -= 1
        end
    end
    s
end
function eh_loop_in_try(n)
    try
        s = 0
        for i in 1:n
            s += eh_g(i)
        end
        s
    catch
        -1
    end
end
function eh_break_in_try(n)
    s = 0
    i = 0
    while true
        i += 1
        try
            s += eh_g(i)
            s > n && break
        catch
            s = -s
            break
        end
    end
    s
end
function eh_continue_from_catch(n)
    s = 0
    i = 0
    while i < n
        i += 1
        try
            s += eh_g(i * 3)
        catch
            continue
        end
        s += 1
    end
    s
end
function eh_throw_in_loop_in_try(n)
    try
        s = 0
        for i in 1:n
            i == 7 && eh_throw(ArgumentError("seven"))
            s += i
        end
        s
    catch e
        e isa ArgumentError ? -7 : -1
    end
end
function eh_throw_in_try_in_loop(n)
    s = 0
    i = 0
    while i < n
        i += 1
        try
            i == 3 && eh_throw(ArgumentError("three"))
            s += i
        catch
            s -= 100
        end
    end
    s
end
function eh_return_from_inner_catch(n)   # inner catch returns out of a loop and an outer try
    try
        s = eh_h(n)
        for i in 1:n
            try
                s += eh_g(i)
            catch
                return -s
            end
        end
        s + eh_h(n - 20)
    catch
        -1
    end
end
function eh_finally_break(n)
    s = 0
    for i in 1:n
        try
            s += eh_g(i)
            s > 10 && break
        finally
            eh_log(i)
        end
    end
    s
end
function eh_finally_return(x)
    try
        return eh_g(x)
    catch
        return -1
    finally
        x > 100 && eh_log(x)
    end
end
function eh_try_in_catch(n)
    s = 0
    for i in 1:n
        try
            s += eh_g(i)
        catch
            try
                s += eh_h(-i)
            catch
                s -= 1
            end
        end
    end
    s
end
function eh_three_levels(x)
    try
        a = eh_g(x)
        try
            b = eh_h(a - 8)
            try
                eh_g(b)
            catch
                b + 1000
            end
        catch
            a + 100
        end
    catch
        -1
    end
end
function eh_sequential(x)
    a = try eh_g(x) catch; 1 end
    b = try eh_h(x - 5) catch; 2 end
    a + b
end
function eh_goto_out(x)
    try
        y = eh_g(x)
        y > 5 && @goto out
        return y
    catch
        return -1
    end
    @label out
    return 1000
end
function eh_lazy(n)   # a variable first defined in a catch: undefined on loop entry
    local acc
    i = 0
    while i < n
        i += 1
        try
            eh_g(i)
        catch
            if !@isdefined(acc)
                acc = Int[]
            end
            push!(acc, i)
        end
    end
    @isdefined(acc) ? length(acc) : 0
end
eh_scoped(x) = @with EH_SV => x begin; eh_readsv() + eh_g(x) end
eh_scope_only(x) = @with EH_SV => x begin; x + 1 end   # nothrow body: EnterNode(0, scope)
# Nested exceptions of one type: only the message tells them apart.
function eh_same_type(x)
    try
        try
            eh_throw(ErrorException("inner $x"))
        catch
            x > 5 && eh_throw(ErrorException("outer $x"))
            rethrow()
        end
    catch e
        x > 10 && rethrow()
        (e::ErrorException).msg
    end
end

const EH_CASES = [
    (eh_basic, (1, 20)), (eh_exc, (1, 20)), (eh_finally, (1, 20)), (eh_nested, (1, 5, 20)),
    (eh_upsilon, (1, 15, 20)), (eh_rethrow, (1, 12, 20)), (eh_return_in_try, (1, 7, 20)),
    (eh_mixed_exit, (-3, 0, 2, 7, 20)), (eh_dead_merge, (0, 2, 5, 20)),
    (eh_nested_return, (1, 20, -20)), (eh_try_in_loop, (0, 3, 15)),
    (eh_loop_in_try, (0, 5, 15)), (eh_break_in_try, (3, 20, 100)),
    (eh_continue_from_catch, (0, 2, 6)), (eh_throw_in_loop_in_try, (0, 5, 9)),
    (eh_throw_in_try_in_loop, (0, 2, 5)), (eh_return_from_inner_catch, (0, 3, 15, 25)),
    (eh_finally_break, (0, 3, 15)), (eh_finally_return, (1, 20)),
    (eh_try_in_catch, (0, 5, 15)), (eh_three_levels, (1, 5, 9, 15)),
    (eh_sequential, (1, 3, 20)), (eh_goto_out, (1, 7, 20)), (eh_lazy, (0, 5, 15)),
    (eh_scoped, (3, 20)), (eh_scope_only, (3, 20)),
    (eh_same_type, (1, 7, 12)),
]

@testset "exceptions" begin

@testset "round trip: $(nameof(f))" for (f, inputs) in EH_CASES
    @test isempty(eh_roundtrip(f, inputs))
    @test isempty(eh_roundtrip(f, inputs; promote=false))
    @test isempty(eh_roundtrip(f, inputs; exceptions=:trap))
end

@testset "TryOp shape: body, handler, results" begin
    @test @filecheck begin
        @check "try -> Tuple{Int64}"
        @check "body:"
        @check "invoke"
        @check "yield"
        @check "catch:"
        @check "pop_exception"
        @check "return -1"
        @check "getfield"
        @check_not "leave"
        code_structured(eh_basic, Tuple{Int})
    end

    sci, _ = only(code_structured(eh_basic, Tuple{Int}))
    op = only(eh_ops(sci, TryOp))
    @test op.handler isa Block
    @test op.scope === nothing
    @test isempty(op.handler.args)
    # The TryOp sits at the EnterNode's SSA index, which pop_exception names.
    tidx = only(i for (i, e) in sci.entry.body if e.stmt === op)
    pops = [e.stmt for (_, e) in op.handler.body
            if e.stmt isa Expr && e.stmt.head === :pop_exception]
    @test only(pops).args[1] == SSAValue(tidx)
    # No EnterNode / :leave / PhiCNode survives the lift.
    for b in IRStructurizer.eachblock(sci), (_, e) in b.body
        @test !(e.stmt isa Core.EnterNode || e.stmt isa Core.PhiCNode)
        @test !(e.stmt isa Expr && e.stmt.head === :leave)
    end
end

@testset "PhiC becomes a handler block argument" begin
    sci, _ = only(code_structured(eh_upsilon, Tuple{Int}))
    op = only(eh_ops(sci, TryOp))
    @test length(op.handler.args) == 1
    @test op.handler.args[1].type === Int
    ups = only(op.upsilons)
    # Every listed upsilon is an UpsilonNode statement: one before the try (the
    # initial value), the rest in the body.
    defs = IRStructurizer.defs(sci)
    @test length(ups) >= 3
    @test all(u -> IRStructurizer.def(defs, SSAValue(u))[:stmt] isa Core.UpsilonNode, ups)
    @test count(u -> haskey(op.body.body, u), ups) == length(ups) - 1
    # The handler returns its argument.
    @test op.handler.terminator == ReturnNode(op.handler.args[1])
end

@testset "LeaveOp exits the try from a nested arm" begin
    sci, _ = only(code_structured(eh_mixed_exit, Tuple{Int}))
    op = only(eh_ops(sci, TryOp))
    @test eh_terminators(op.body, LeaveOp) == 1
    @test op.body.terminator isa YieldOp        # the body's own exit is a yield
    @test @filecheck begin
        @check "try"
        @check "if"
        @check "if"
        @check "leave"
        @check "catch:"
        code_structured(eh_mixed_exit, Tuple{Int})
    end

    # A dead end (throw) merged from both arms stays inside the try body.
    sci2, _ = only(code_structured(eh_dead_merge, Tuple{Int}))
    op2 = only(eh_ops(sci2, TryOp))
    @test eh_terminators(op2.body, LeaveOp) == 1
    @test op2.body.terminator == ReturnNode()   # the throw ends the body
end

@testset "multi-handler leave and pop_exception after an outer leave" begin
    ir, _ = only(code_ircode(eh_nested_return, (Int,)))
    @test any(s -> s isa Expr && s.head === :leave && length(s.args) == 2, ir.stmts.stmt)
    sci = StructuredIRCode(ir)
    outer = only(op for op in eh_ops(sci, TryOp) if !isempty(eh_ops(op.body, TryOp)))
    oidx = only(i for (i, e) in sci.entry.body if e.stmt === outer)
    # The inner catch's pop_exception runs after the outer try was left; it is
    # retargeted to the outer enter (same exception-stack state), which dominates.
    after = Any[]
    for b in IRStructurizer.eachblock(sci.entry)
        b === outer.body && continue
        any(x -> x === b, IRStructurizer.eachblock(outer.body)) && continue
        for (_, e) in b.body
            e.stmt isa Expr && e.stmt.head === :pop_exception && push!(after, e.stmt.args[1])
        end
    end
    @test all(==(SSAValue(oidx)), after)
    @test !isempty(after)
end

@testset "loops: the try stays inside the loop, exits go through the latch" begin
    sci, _ = only(code_structured(eh_try_in_loop, Tuple{Int}))
    loop = only(eh_ops(sci, IRStructurizer.ForOp))
    @test length(eh_ops(loop.body, TryOp)) == 1
    # A throw inside the try inside the loop is not hoisted out of the try.
    sci2, _ = only(code_structured(eh_throw_in_try_in_loop, Tuple{Int}))
    op = only(eh_ops(sci2, TryOp))
    @test eh_terminators(op.body, ReturnNode) >= 1
    # break out of a try: the TryOp has a single continuation, the loop's exit
    # dispatch runs after it.
    sci3, _ = only(code_structured(eh_break_in_try, Tuple{Int}))
    op3 = only(eh_ops(sci3, TryOp))
    @test eh_terminators(op3.body, BreakOp) == 0
    @test eh_terminators(op3.body, ContinueOp) == 0
end

@testset "scoped values" begin
    sci, _ = only(code_structured(eh_scoped, Tuple{Int}))
    op = only(eh_ops(sci, TryOp))
    @test op.scope isa SSAValue
    @test op.handler isa Block
    # A nothrow body keeps only the scope: EnterNode(0, scope), no handler.
    ir, _ = only(code_ircode(eh_scope_only, (Int,)))
    @test any(s -> s isa Core.EnterNode && s.catch_dest == 0, ir.stmts.stmt)
    sci2 = StructuredIRCode(ir)
    op2 = only(eh_ops(sci2, TryOp))
    @test op2.handler === nothing
    @test op2.scope isa SSAValue
    @test length(IRStructurizer.blocks(op2)) == 1
end

@testset "exceptions=:trap drops the handlers" begin
    for (f, _) in EH_CASES
        sci, _ = only(code_structured(f, Tuple{Int}; exceptions=:trap))
        ops = eh_ops(sci, TryOp)
        # Only scope enters survive, without a handler.
        @test all(op -> op.handler === nothing && op.scope !== nothing, ops)
        for b in IRStructurizer.eachblock(sci), (_, e) in b.body
            @test !(e.stmt isa Core.UpsilonNode)
            @test !(e.stmt isa Expr && e.stmt.head in (:pop_exception, :the_exception))
        end
    end
    @test length(eh_ops(first(only(code_structured(eh_scoped, Tuple{Int}; exceptions=:trap))), TryOp)) == 1
    @test_throws ArgumentError StructuredIRCode(first(only(code_ircode(eh_basic, (Int,))));
                                                 exceptions=:bogus)
end

@testset "unsupported shapes are rejected" begin
    # The catch block is also reached by ordinary control flow (after a leave).
    ir = build_ir([
        (stmts=[(Core.EnterNode(3), Any)],                       succs=[3, 2]),
        (stmts=[(Expr(:leave, SSAValue(1)), Any), (GotoNode(3), Any)], succs=[3]),
        (stmts=[(ReturnNode(1), Any)],                           succs=Int[]),
    ], Any[Any, Int])
    @test_throws UnstructuredControlFlowError StructuredIRCode(ir)

    # Unbalanced: a block reached with and without the handler active.
    ir2 = build_ir([
        (stmts=[(GotoIfNot(Argument(2), 3), Any)],               succs=[3, 2]),
        (stmts=[(Core.EnterNode(4), Any)],                       succs=[4, 3]),
        (stmts=[(ReturnNode(1), Any)],                           succs=Int[]),
        (stmts=[(ReturnNode(2), Any)],                           succs=Int[]),
    ], Any[Any, Bool])
    @test_throws UnstructuredControlFlowError StructuredIRCode(ir2)
end

@testset "unreachable blocks are not irreducible" begin
    # `CFGReachability.irreducible` is uninitialized for unreachable blocks; the
    # irreducible pass must not read it (trap-mode pruning leaves dead blocks).
    ir = build_ir([
        (stmts=[(GotoNode(4), Any)],     succs=[4]),
        (stmts=[(ReturnNode(), Any)],    succs=Int[]),
        (stmts=[(ReturnNode(), Any)],    succs=Int[]),
        (stmts=[(ReturnNode(1), Any)],   succs=Int[]),
    ], Any[Any])
    for _ in 1:20
        @test !IRStructurizer.normalize_one_irreducible!(IRStructurizer.ingest(ir))
    end
end

@testset "handler stacks and try regions" begin
    ir, _ = only(code_ircode(eh_nested, (Int,)))
    m = IRStructurizer.ingest(ir)
    stacks = IRStructurizer.handler_stacks(m)
    regions = IRStructurizer.try_info(m, stacks)
    @test length(regions) == 2
    outer, inner = sort!(collect(values(regions)); by=r -> r.depth)
    @test outer.depth == 0 && inner.depth == 1
    # The inner try (body and handler) nests in the outer body.
    @test inner.block in outer.body
    @test issubset(inner.body, outer.body) && issubset(inner.handler, outer.body)
    @test isdisjoint(outer.body, outer.handler)
    # Every :leave sits alone in its block and pops one handler.
    for b in m.blocks, s in b.body
        IRStructurizer.is_leave(s.stmt) || continue
        @test length(b.body) == 1 && length(s.stmt.args) == 1
    end
end

end  # exceptions
