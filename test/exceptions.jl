#=============================================================================
 Exception handling: EnterNode / :leave / PhiC+Upsilon to TryOp / LeaveOp.

 The shapes below are functions with try/catch. Each is lifted with
 `exceptions=:structured` and round-trips (structurize, unstructurize, execute) on
 inputs that do and don't throw, comparing results, thrown exceptions (type and
 message), the `eh_log` side effects, and the exception-stack depth (`eh_check`).
 Structural tests pin the TryOp shape: handler block args from PhiC, LeaveOp from
 nested exits, a single continuation, and the scope operand. The MCFG tests check
 what `ingest(ir; exceptions=:structured)` and `normalize_cf!` build: the handler
 stacks, the try regions, and where each `pop_exception` runs.
=============================================================================#

using IRStructurizer: TryOp, LeaveOp, ingest, handler_stacks, is_leave, MEnter, try_info,
                      try_exit_refs, edge_of, is_pop_exception
using Base.ScopedValues

"""`code_structured` with try/catch lifted to `TryOp` (`exceptions=:structured`)."""
eh_structured(f, tt) = code_structured(f, tt; exceptions=:structured)

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
made, and how many exceptions it left on the exception stack after a normal return
(-1 after a throw). A lost or mistargeted `pop_exception` leaves exceptions on the
stack.

The depth is relative to the stack before the call. Julia 1.11 itself leaves the
exception on the stack when a `finally` returns after a throw
(`try f() finally return 1 end`), and leaked exceptions stay there across calls,
so an absolute depth would also count the leaks of the calls that ran before."""
function eh_outcome(f, args...)
    empty!(EH_LOG)
    depth = -1
    before = length(current_exceptions())
    result = try
        v = f(args...)
        depth = length(current_exceptions()) - before
        (:ok, v)
    catch e
        (:threw, eh_payload(e))
    end
    return (; result, log = copy(EH_LOG), depth)
end

"""Lower `sci` back to `IRCode`, check it with `verify_ir`, and compile it. This
runs outside `eh_outcome`, so a lowering failure is a test error, never an
outcome to compare."""
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
    eh_check(f, inputs; promote=true, argtypes, optimize_until) -> Vector{String}

Structurize `f` (at the types of the first input tuple, or `argtypes`) with
`exceptions=:structured`, lower it back with `eh_lower`, and run it on every input
tuple. Each outcome must equal the direct call's (`eh_outcome`). Returns the
mismatches.
"""
function eh_check(f, inputs; promote=true, argtypes=Tuple{map(typeof, first(inputs))...},
                  optimize_until=nothing)
    ir, _ = only(code_ircode(f, argtypes; optimize_until))
    sci = StructuredIRCode(ir; exceptions=:structured, promote)
    return eh_compare(f, eh_lower(sci), inputs)
end

"""Run the lowered closure `oc` and `f` on every input, as `eh_check` describes."""
function eh_compare(f, oc, inputs)
    problems = String[]
    for args in inputs
        expected = eh_outcome(f, args...)
        got = eh_outcome(oc, args...)
        isequal(expected, got) && continue
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
# Inlined ScopedValue reads: the phi of the read is narrower than the value
# feeding it (an `Any` yield into an `Int` slot unless narrowed).
eh_scoped_inline(x) = @with EH_SV => x begin; EH_SV[] + eh_g(x) end
eh_scoped_inline_try(x) = try @with EH_SV => x begin; eh_g(EH_SV[]) end catch; EH_SV[] end
# A body that cannot complete normally: the catch dominates everything after it.
function eh_throwing_body_in_loop(n)
    for i in 1:n
        try
            eh_throw(ErrorException("x"))
        catch
        end
    end
    n
end
function eh_throwing_body_two_exits(x)
    x < -5 && @goto big
    if x > 0
        try
            eh_g(x + 20)
        catch
        end
        x > 5 && @goto big
    end
    return 0
    @label big
    return 1
end
# `Base.wait_forever`'s shape: an endless loop around a try whose body is an endless
# loop. Not run, since it never returns.
function eh_endless(x)
    while true
        try
            while true
                eh_log(eh_g(x))
            end
        catch e
            eh_log(e)
        end
    end
end
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
# An exit out of a try nested in a catch: Julia pops the catch's exception before
# it leaves the nested try, so the leave block is still inside the handler.
function eh_break_from_try_in_catch(x)
    for i in 1:3
        try
            x > i && eh_throw(ErrorException("a $i"))
        catch
            try
                eh_log(i)
                x > 5 && break
            catch
            end
        end
    end
    x
end
function eh_return_from_try_in_catch(x)
    try
        try
            x > 0 && eh_throw(ErrorException("a"))
        catch
            try
                eh_log(9)
                return x
            catch
            end
        end
    finally
        eh_log(1)
    end
    -x
end
# The first try's body always returns, so its catch dominates the second try,
# which must not join the first try's handler.
# A catch that returns out of an enclosing try and a loop: Julia leaves the outer
# try, then pops the inner catch's exception.
function eh_return_from_nested_catch_in_loop(x)
    w = 0
    while w < 3
        w += 1
        try
            try
                x > w && eh_throw(ErrorException("a $w"))
            catch
                x > 5 && return x
            end
            x > 9 && eh_throw(ErrorException("b"))
        catch
        end
    end
    x
end
# A `continue` out of a try nested in a catch: Julia pops the catch's exception,
# then leaves the nested try. The body always throws, so the catch dominates the
# loop's latch.
function eh_continue_from_try_in_catch(x)
    for i in 1:3
        try
            eh_throw(ErrorException("a $i"))
        catch
            try
                x > 4 && eh_throw(ErrorException("b $i"))
                continue
            catch
                eh_log(i)
            end
        end
        eh_log(-i)
    end
    x
end
function eh_continue_from_try_in_catch_in_try(x)
    try
        for i in 1:3
            try
                eh_throw(ErrorException("a $i"))
            catch
                try
                    x > 4 && eh_throw(ErrorException("b $i"))
                    continue
                catch
                    eh_log(i)
                end
            end
            eh_log(-i)
        end
    catch
    end
    x
end
# A `finally` that returns after a throw. Julia 1.11 leaves the exception on the
# exception stack here, in the direct call and the round trip alike.
function eh_return_from_finally(x)
    try
        eh_g(x)
    finally
        x > 5 && return -x
    end
end
# A try nested in a catch, in a loop, whose exits pop the catch's exception at
# different points: the `break` pops it in the nested catch, the `return` after
# leaving the nested try. The nested try's continuation is still in the catch.
function eh_return_or_break_from_try_in_catch(x)
    for i in 1:3
        try
            eh_log(i)
            x > 7 && eh_throw(ErrorException("a $i"))
        catch
            try
                x > -2 && eh_throw(ErrorException("b $i"))
                return x
            catch
                break
            end
        end
    end
    -x
end
# `continue` and `break` from a catch in an inner loop: the inner loop's two exits
# meet in a block with arguments right before the outer back edge.
function eh_exits_from_catch_in_nested_loops(x)
    w = 0
    while w < 3
        w += 1
        v = 0
        while v < 3
            v += 1
            try
                x > 1 && eh_throw(ErrorException("a $v"))
                x += 11
            catch
                x > 5 && continue
                x > 7 && break
                x += 1
            end
        end
    end
    x
end
# A catch returns out of a try nested in another catch, itself in a try body: the
# return leaves two trys and pops two catches' exceptions at once.
function eh_return_across_catches_and_trys(x)
    try
        try
            eh_log(10)
            x > 5 && eh_throw(ErrorException("a"))
        catch
            try
                try
                    eh_throw(ErrorException("b"))
                catch
                    eh_log(0)
                    return x
                end
            catch
                rethrow()
            end
        end
    catch
    end
    -x
end
function eh_try_after_returning_try(x)
    try
        eh_log(10)
        return x
    catch
    end
    try
        x += 1
        eh_log(5)
    catch
    end
    x
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
    (eh_scoped_inline, (3, 20)), (eh_scoped_inline_try, (3, 20)),
    (eh_throwing_body_in_loop, (0, 3)), (eh_throwing_body_two_exits, (-9, -1, 3, 9)),
    (eh_same_type, (1, 7, 12)),
    (eh_break_from_try_in_catch, (-2, 0, 1, 3, 7)),
    (eh_return_from_try_in_catch, (-2, 0, 3)),
    (eh_try_after_returning_try, (0, 3)),
    (eh_return_from_nested_catch_in_loop, (-2, 0, 3, 7, 12)),
    (eh_continue_from_try_in_catch, (-2, 3, 7)),
    (eh_continue_from_try_in_catch_in_try, (-2, 3, 7)),
    (eh_return_from_finally, (1, 7, 12, 20)),
    (eh_return_or_break_from_try_in_catch, (-5, 0, 9)),
    (eh_exits_from_catch_in_nested_loops, (-5, 0, 3, 6, 8)),
    (eh_return_across_catches_and_trys, (1, 7)),
]

@testset "exceptions" begin

@testset "round trip: $(nameof(f))" for (f, inputs) in EH_CASES
    @test isempty(eh_roundtrip(f, inputs))
    @test isempty(eh_roundtrip(f, inputs; promote=false))
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
        eh_structured(eh_basic, Tuple{Int})
    end

    sci, _ = only(eh_structured(eh_basic, Tuple{Int}))
    op = only(eh_ops(sci, TryOp))
    @test op.handler isa Block
    @test !IRStructurizer.has_scope(op)
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
    sci, _ = only(eh_structured(eh_upsilon, Tuple{Int}))
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
    sci, _ = only(eh_structured(eh_mixed_exit, Tuple{Int}))
    op = only(eh_ops(sci, TryOp))
    @test eh_terminators(op.body, LeaveOp) == 1
    @test op.body.terminator isa YieldOp        # the body's own exit is a yield
    @test @filecheck begin
        @check "try"
        @check "if"
        @check "if"
        @check "leave"
        @check "catch:"
        eh_structured(eh_mixed_exit, Tuple{Int})
    end

    # A dead end (throw) merged from both arms stays inside the try body.
    sci2, _ = only(eh_structured(eh_dead_merge, Tuple{Int}))
    op2 = only(eh_ops(sci2, TryOp))
    @test eh_terminators(op2.body, LeaveOp) == 1
    @test op2.body.terminator == ReturnNode()   # the throw ends the body
end

@testset "multi-handler leave and pop_exception after an outer leave" begin
    pops(blk) = Any[e.stmt.args[1] for b in IRStructurizer.eachblock(blk) for (_, e) in b.body
                    if e.stmt isa Expr && e.stmt.head === :pop_exception]
    tryidx(sci, op) = only(i for b in IRStructurizer.eachblock(sci) for (i, e) in b.body
                           if e.stmt === op)
    ir, _ = only(code_ircode(eh_nested_return, (Int,)))
    @test any(s -> s isa Expr && s.head === :leave && length(s.args) == 2, ir.stmts.stmt)
    # Julia leaves the outer try, then pops the inner catch's exception. The pop
    # moves before the leave, into the inner catch, where its own enter dominates it.
    sci = StructuredIRCode(ir; exceptions=:structured)
    outer = only(op for op in eh_ops(sci, TryOp) if !isempty(eh_ops(op.body, TryOp)))
    inner = only(eh_ops(outer.body, TryOp))
    @test SSAValue(tryidx(sci, inner)) in pops(inner.handler)
    @test pops(sci.entry) == [pops(outer.body); pops(outer.handler)]
    @test !(SSAValue(tryidx(sci, inner)) in pops(outer.handler))
    # The same inside a loop: the return leaves the loop too, and a pop left after
    # the leave would end up after the loop, where neither enter dominates it.
    sci = first(only(eh_structured(eh_return_from_nested_catch_in_loop, Tuple{Int})))
    loop = only(e.stmt for (_, e) in sci.entry.body if e.stmt isa Union{LoopOp, WhileOp})
    ids(v) = sort!([p.id for p in v])
    @test ids(pops(sci.entry)) == ids(reduce(vcat, map(pops, IRStructurizer.blocks(loop))))
    inner = only(op for op in eh_ops(sci, TryOp) if isempty(eh_ops(op.body, TryOp)))
    # its normal exit and the return both pop in the catch
    @test pops(inner.handler) == fill(SSAValue(tryidx(sci, inner)), 2)
end

@testset "validation: pop_exception is scoped like any use of its TryOp" begin
    sci = first(only(eh_structured(eh_try_in_loop, Tuple{Int})))
    @test validate_ssa_defs(sci)
    op = only(eh_ops(sci, TryOp))
    t = only(i for b in IRStructurizer.eachblock(sci) for (i, e) in b.body if e.stmt === op)
    # A pop after the loop names a TryOp inside it: its enter does not dominate it.
    push!(sci.entry, sci.max_ssa_idx + 1, Expr(:pop_exception, SSAValue(t)), Nothing)
    @test_throws "SSA values used but not defined: %$t" validate_ssa_defs(sci)
end

@testset "loops: the try stays inside the loop, exits go through the latch" begin
    sci, _ = only(eh_structured(eh_try_in_loop, Tuple{Int}))
    loop = only(eh_ops(sci, IRStructurizer.ForOp))
    @test length(eh_ops(loop.body, TryOp)) == 1
    # A throw inside the try inside the loop is not hoisted out of the try.
    sci2, _ = only(eh_structured(eh_throw_in_try_in_loop, Tuple{Int}))
    op = only(eh_ops(sci2, TryOp))
    @test eh_terminators(op.body, ReturnNode) >= 1
    # break out of a try: the TryOp has a single continuation, the loop's exit
    # dispatch runs after it.
    sci3, _ = only(eh_structured(eh_break_in_try, Tuple{Int}))
    op3 = only(eh_ops(sci3, TryOp))
    @test eh_terminators(op3.body, BreakOp) == 0
    @test eh_terminators(op3.body, ContinueOp) == 0
end

@testset "scoped values" begin
    sci, _ = only(eh_structured(eh_scoped, Tuple{Int}))
    op = only(eh_ops(sci, TryOp))
    @test op.scope isa SSAValue
    @test op.handler isa Block
    # A nothrow body keeps only the scope: EnterNode(0, scope), no handler.
    ir, _ = only(code_ircode(eh_scope_only, (Int,)))
    @test any(s -> s isa Core.EnterNode && s.catch_dest == 0, ir.stmts.stmt)
    sci2 = StructuredIRCode(ir; exceptions=:structured)
    op2 = only(eh_ops(sci2, TryOp))
    @test op2.handler === nothing
    @test op2.scope isa SSAValue
    @test length(IRStructurizer.blocks(op2)) == 1
end

@testset "the exceptions keyword" begin
    ir = first(only(code_ircode(eh_basic, (Int,))))
    # The default rejects try/catch, as before `TryOp` existed.
    @test_throws UnstructuredControlFlowError StructuredIRCode(ir)
    @test_throws UnstructuredControlFlowError code_structured(eh_basic, Tuple{Int})
    @test StructuredIRCode(ir; exceptions=:structured) isa StructuredIRCode
    @test_throws ArgumentError StructuredIRCode(ir; exceptions=:bogus)
end

@testset "a body that cannot complete normally" begin
    # The catch dominates the code after the try (here, the loop's back edge). The
    # handler region ends at `pop_exception`, so the try has a continuation and
    # normalization converges (it used to grow the handler by one mux per step).
    sci, _ = only(eh_structured(eh_throwing_body_in_loop, Tuple{Int}))
    op = only(eh_ops(sci, TryOp))
    @test op.body.terminator == ReturnNode()
    @test op.handler.terminator isa YieldOp
    @test only(eh_ops(sci, ForOp)) isa ForOp
    # `eh_endless` never returns, so only the lift and the lowering are checked.
    sci, _ = only(eh_structured(eh_endless, Tuple{Int}))
    @test (CC.verify_ir(CC.IRCode(sci)); true)
end

@testset "validation rejects malformed TryOps" begin
    invalid(sci) = try validate_terminators(sci); "" catch e; e.msg end
    fresh(f) = first(only(eh_structured(f, Tuple{Int})))
    sci = fresh(eh_basic)
    @test validate_terminators(sci)
    # The body yield must carry the op's one result, of its type.
    op = only(eh_ops(sci, TryOp))
    op.body.terminator = YieldOp()
    @test occursin("body yield carries 0 values", invalid(sci))
    op.body.terminator = YieldOp(Any["str"])
    @test occursin("not <: declared Int64", invalid(sci))
    # A value return inside the body would skip the leave.
    op.body.terminator = ReturnNode(42)
    @test occursin("value return inside the body", invalid(sci))
    # A LeaveOp out of a nested arm carries the op's results too.
    sci = fresh(eh_mixed_exit)
    op = only(eh_ops(sci, TryOp))
    leave = only(b for b in IRStructurizer.eachblock(op.body) if b.terminator isa LeaveOp)
    leave.terminator = LeaveOp(leave.terminator.values[1:end-1])
    @test occursin("leave carries", invalid(sci))
    # Upsilons name UpsilonNode statements.
    sci = fresh(eh_upsilon)
    op = only(eh_ops(sci, TryOp))
    notups = first(i for (i, e) in op.body.body if !(e.stmt isa Core.UpsilonNode))
    push!(op.upsilons[1], notups)
    @test occursin("is not an UpsilonNode", invalid(sci))
    # A LeaveOp exits a try body only: not from a loop inside it, nor anywhere in
    # code without a TryOp.
    sci = fresh(eh_mixed_exit)
    op = only(eh_ops(sci, TryOp))
    @test validate_terminators(sci)
    op.handler.terminator = LeaveOp()
    @test occursin("LeaveOp outside a TryOp body", invalid(sci))
    sci = StructuredIRCode(first(only(code_ircode(eh_throw_in_loop_in_try, (Int,))));
                           exceptions=:structured, promote=false)
    loop = only(eh_ops(only(eh_ops(sci, TryOp)).body, LoopOp))
    first(eh_ops(loop.body, IfOp)).then_region.terminator = LeaveOp()
    @test occursin("LeaveOp outside a TryOp body", invalid(sci))
    sci = first(only(eh_structured(x -> x > 0 ? x : -x, Tuple{Int})))
    first(e.stmt for (_, e) in sci.entry.body if e.stmt isa IfOp).then_region.terminator =
        LeaveOp()
    @test occursin("LeaveOp outside a TryOp body", invalid(sci))
end

@testset "has_eh: the try analyses only run for code with a try" begin
    m = IRStructurizer.ingest(first(only(code_ircode(x -> x > 0 ? x : -x, (Int,))));
                              exceptions=:structured)
    @test !m.has_eh && !IRStructurizer.has_enter(m)
    m = IRStructurizer.ingest(first(only(code_ircode(eh_basic, (Int,)))); exceptions=:structured)
    @test m.has_eh && IRStructurizer.has_enter(m)
    # Without a try, a branch carries no handler depths.
    ir = first(only(code_ircode(x -> x > 0 ? x : -x, (Int,))))
    m = IRStructurizer.normalize_cf(ir)
    @test IRStructurizer.StructurizeCtx(m).depth === nothing
end

@testset "ingest: the exceptions keyword" begin
    ir = first(only(code_ircode(eh_basic, (Int,))))
    @test_throws UnstructuredControlFlowError ingest(ir)
    @test_throws ArgumentError ingest(ir; exceptions=:bogus)
    m = ingest(ir; exceptions=:structured)
    @test m.has_eh && count(b -> b.term isa MEnter, m.blocks) == 1
    m = ingest(first(only(code_ircode(x -> x > 0 ? x : -x, (Int,)))); exceptions=:structured)
    @test !m.has_eh
end

@testset "handler stacks" begin
    m = ingest(first(only(code_ircode(eh_nested, (Int,)))); exceptions=:structured)
    stacks = handler_stacks(m)
    enter_blocks = [bid for (bid, b) in enumerate(m.blocks) if b.term isa MEnter]
    @test length(enter_blocks) == 2
    outer, inner = sort(enter_blocks; by=bid -> length(stacks[bid]))
    # The inner try is entered with the outer handler active; its body runs with both.
    @test stacks[outer] == Int[] && stacks[inner] == [m.blocks[outer].term.id]
    @test stacks[m.blocks[inner].term.body.target] ==
          [m.blocks[outer].term.id, m.blocks[inner].term.id]
    # The catch block keeps the stack its try was entered with.
    @test stacks[m.blocks[inner].term.catch_dest] == stacks[inner]
    # `return` out of two trys: Julia's one `:leave` popping both becomes a chain of
    # leave blocks, each alone in its block and popping the innermost handler.
    ir = first(only(code_ircode(eh_nested_return, (Int,))))
    @test any(s -> s isa Expr && s.head === :leave && length(s.args) == 2, ir.stmts.stmt)
    m = ingest(ir; exceptions=:structured)
    stacks = handler_stacks(m)
    nleaves = 0
    for (bid, b) in enumerate(m.blocks), s in b.body
        is_leave(s.stmt) || continue
        nleaves += 1
        @test length(b.body) == 1 && length(s.stmt.args) == 1
        @test s.stmt.args[1].id == stacks[bid][end]
    end
    @test nleaves >= 3
    # A block reached with and without a handler active is rejected.
    ir = build_ir([
        (stmts=[(GotoIfNot(Argument(2), 3), Any)],               succs=[3, 2]),
        (stmts=[(Core.EnterNode(4), Any)],                       succs=[4, 3]),
        (stmts=[(ReturnNode(1), Any)],                           succs=Int[]),
        (stmts=[(ReturnNode(2), Any)],                           succs=Int[]),
    ], Any[Any, Bool])
    @test_throws UnstructuredControlFlowError ingest(ir; exceptions=:structured)
end

"""The MCFG `ingest` builds for `f` at `Tuple{Int}`, with try/catch."""
eh_mcfg(f) = ingest(first(only(code_ircode(f, Tuple{Int}))); exceptions=:structured)

@testset "try regions" begin
    m = eh_mcfg(eh_nested)
    regions = try_info(m)
    @test length(regions) == 2
    outer, inner = sort!(collect(values(regions)); by=r -> r.depth)
    @test outer.depth == 0 && inner.depth == 1
    # The inner try (body and handler) nests in the outer body.
    @test inner.block in outer.body
    @test issubset(inner.body, outer.body) && issubset(inner.handler, outer.body)
    @test isdisjoint(outer.body, outer.handler)
    # Every edge out of a try's body and handler lands on a block with the stack the
    # try was entered with. That includes an exit out of a try nested in a catch,
    # which pops the catch's exception before it leaves the nested try: the leave
    # block after the pop still belongs to the catch.
    @testset "$(nameof(f))" for (f, _) in EH_CASES
        m = eh_mcfg(f)
        stacks = handler_stacks(m)
        for r in values(try_info(m)), ref in try_exit_refs(m, r)
            @test stacks[edge_of(m, ref).target] == stacks[r.block]
        end
    end
    # A body that cannot complete normally: the catch dominates the loop's latch,
    # but the handler region ends at the catch's `pop_exception`.
    m = eh_mcfg(eh_throwing_body_in_loop)
    r = only(values(try_info(m)))
    loops = IRStructurizer.natural_loops_m(m)
    @test !any(h -> h in r.handler, keys(loops))
end

"""For each `pop_exception` in `m`: the handler depth of the catch it pops (its
try's entry depth), and the depth of the block it runs in."""
function eh_pop_depths(m)
    stacks = handler_stacks(m)
    enter_depth = Dict(b.term.id => length(stacks[bid]) for (bid, b) in enumerate(m.blocks)
                       if b.term isa MEnter && stacks[bid] !== nothing)
    return [(enter_depth[s.stmt.args[1].id], length(stacks[bid]))
            for (bid, b) in enumerate(m.blocks) if stacks[bid] !== nothing
            for s in b.body if is_pop_exception(s.stmt)]
end

@testset "a pop after a leave of an enclosing try moves before it" begin
    # A catch that returns out of an enclosing try: Julia leaves the outer try,
    # then pops the inner catch's exception, one handler shallower than the catch.
    # Moved before the leave, the pop runs in its catch.
    @testset "$(nameof(f))" for f in (eh_nested_return, eh_return_from_inner_catch,
                                      eh_return_from_nested_catch_in_loop, eh_three_levels)
        @test all(((d, at),) -> at >= d, eh_pop_depths(eh_mcfg(f)))
    end
end

@testset "a pop before leaving trys nested in its catch moves after them" begin
    # A `continue` out of a try nested in a catch pops the catch's exception, then
    # leaves the nested try. Moved after the leave, every pop runs at exactly the
    # handler depth of its catch.
    @testset "$(nameof(f))" for (f, _) in EH_CASES
        @test all(((d, at),) -> at == d, eh_pop_depths(eh_mcfg(f)))
    end
end

@testset "a pop that leaves several catches is split, one per catch" begin
    # A `break` from a catch nested in another catch pops straight to the outer
    # state. Split, every pop runs in the exception state of the catch it pops.
    @testset "$(nameof(f))" for (f, _) in EH_CASES
        m = eh_mcfg(f)
        enter_block = Dict(b.term.id => bid for (bid, b) in enumerate(m.blocks)
                           if b.term isa MEnter)
        state, at_enter = IRStructurizer.exception_states(m, enter_block)
        for (bid, b) in enumerate(m.blocks)
            state[bid] < 0 && continue
            s = state[bid]
            for st in b.body
                is_pop_exception(st.stmt) || continue
                x = st.stmt.args[1].id
                @test s == x
                s = at_enter[x]
            end
        end
    end
end

@testset "values undefined on the catch path become Undef" begin
    # A variable never assigned before a throw reaches the catch through a
    # `Union{}`-typed `PhiCNode`, and phis carry it on. Nothing reads such a value
    # (Julia guards it with `throw_undef_if_not`), so no edge operand or upsilon
    # names one after ingest.
    @testset "$(nameof(f))" for (f, _) in EH_CASES
        m = eh_mcfg(f)
        bottom = Set{Int}(a for b in m.blocks for a in b.args if get(m.types, a, Any) === Union{})
        for b in m.blocks, s in b.body
            s.stmt isa Core.PhiCNode && s.type === Union{} && push!(bottom, s.id)
        end
        isbottom(v) = v isa SSAValue && v.id in bottom
        for b in m.blocks
            t = b.term
            edges = t isa IRStructurizer.MGoto ? (t.edge,) :
                    t isa IRStructurizer.MCondBr ? (t.t, t.f) : t isa MEnter ? (t.body,) : ()
            @test !any(e -> any(isbottom, e.args), edges)
            @test !any(s -> s.stmt isa Core.UpsilonNode && isdefined(s.stmt, :val) &&
                            isbottom(s.stmt.val), b.body)
        end
    end
    @test any(s -> s isa Core.PhiCNode, first(only(code_ircode(eh_lazy, (Int,)))).stmts.stmt)
end

@testset "normalize_cf! gives every try a single continuation" begin
    # Afterwards the edges out of a try's body and handler reach one block, which
    # is not a loop header, and a loop that contains a try contains all of it.
    @testset "$(nameof(f))" for (f, _) in EH_CASES
        m = eh_mcfg(f)
        IRStructurizer.normalize_cf!(m)
        loops = IRStructurizer.natural_loops_m(m)
        for r in values(try_info(m))
            targets = unique(edge_of(m, ref).target for ref in try_exit_refs(m, r))
            @test length(targets) <= 1
            @test !any(in(keys(loops)), targets)
            for body in values(loops)
                r.block in body && @test issubset(IRStructurizer.construct_blocks(r), body)
            end
        end
    end
    # A catch block also reached by ordinary control flow (after a leave) is not
    # something Julia's lowering produces, and is rejected.
    ir = build_ir([
        (stmts=[(Core.EnterNode(3), Any)],                       succs=[3, 2]),
        (stmts=[(Expr(:leave, SSAValue(1)), Any), (GotoNode(3), Any)], succs=[3]),
        (stmts=[(ReturnNode(1), Any)],                           succs=Int[]),
    ], Any[Any, Int])
    @test_throws UnstructuredControlFlowError IRStructurizer.normalize_cf!(
        ingest(ir; exceptions=:structured))
end

end  # exceptions
