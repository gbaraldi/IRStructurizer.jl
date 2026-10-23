#=============================================================================
 Control-flow shapes from JuliaLowering.jl's test suite, as round-trip tests.

 The cases come from JuliaLowering.jl's `test/` directory (exceptions, branching,
 loops, generators, closures, scopes, decls, destructuring, compat and demo). Each
 snippet there evaluates a top-level expression over free globals (`a`, `b`, `f`).
 Here each one is an ordinary function whose arguments pick the path, and the
 opaque operations are `@noinline` calls that can throw or log. A comment names the
 source of each case. Cases marked `(added)` fill in categories the suite only
 touches: irreducible gotos, trys inside and around irreducible loops, ScopedValues,
 globals in loops, exits across several trys, and bodies that never complete.

 Every case is structurized in both `exceptions` modes, with and without loop
 promotion, checked for leftover exception statements and one `TryOp` per
 `EnterNode`, and round-tripped with `eh_compare` (test/exceptions.jl): lowered
 back to `IRCode` (which must pass `verify_ir`) and run on every input, the result,
 thrown exception, `jl_log` side effects and exception-stack depth must match the
 direct call. Every throw goes through `eh_throw`, so under `exceptions=:trap` a
 run may end only in the first throw of the direct call.
 Each case also runs with `Any` arguments, and with the optimizer stopped after
 its first and second compaction, to get less tidy IR from the same source.

 Skipped, because they only make sense as lowering-internal or top-level syntax:
 - error cases (`break` outside a loop, undefined or duplicate labels, `@label` as a
   value, bad `let`/`local`/`global` declarations): they never reach `IRCode`;
 - top-level-only forms: soft/neutral/hard scope wrappers on module globals,
   `global` declarations in tail position, `const` and `global f() = ...` method
   definitions in `let`, modules, macros, `@eval`, `include`, typedefs, `@ccall`,
   `@cfunction`, `@islocal`/`@locals`;
 - forms without control flow of their own (named tuples, property access,
   updating and chained assignments, juxtaposition, quoting).
=============================================================================#

using Base.ScopedValues

#----------------------------------------------------------------------------- helpers

# `jl_f` stands for the free calls of the original snippets (`a`, `f`, `g`). It
# throws for x > 10. `jl_g` throws a different exception for x < 0. `jl_error`
# stands for `error`, and every throw goes through `eh_throw`, which logs it.
@noinline jl_f(x::Int) = x > 10 ? eh_throw(ErrorException("f: $x")) : x + 1
@noinline jl_g(x::Int) = x < 0 ? eh_throw(DomainError(x)) : 2x
jl_error(msg) = eh_throw(ErrorException(msg))
@noinline jl_log(@nospecialize(x)) = (push!(EH_LOG, x); nothing)
const JL_SV = ScopedValue(1)
const JL_SV2 = ScopedValue(10)
@noinline jl_readsv() = JL_SV[]
@noinline jl_readsv2() = JL_SV2[]
global jl_counter::Int = 0
jl_untyped = 0

# Cases whose caught exceptions Julia raises itself (an undefined variable), not
# `eh_throw`.
const JL_INTERNAL_THROWS = Set{Symbol}([:jl_destructure_throw, :jl_destructure_isdefined])

"""Structurize `f` at the signature of its inputs (or `argtypes`), check the
structured shape, and round-trip it with `eh_compare`. `optimize_until` stops the
optimizer early, as for `code_ircode`. Returns the problems found."""
function jl_roundtrip(f, inputs; exceptions=:structured, promote=true,
                      argtypes=Tuple{map(typeof, first(inputs))...}, optimize_until=nothing)
    ir, _ = only(code_ircode(f, argtypes; optimize_until))
    nenter = count(s -> s isa Core.EnterNode, ir.stmts.stmt)
    sci = StructuredIRCode(ir; exceptions, promote)
    problems = String[]
    # No unstructured exception construct survives the lift.
    for b in IRStructurizer.eachblock(sci), (_, e) in b.body
        s = e.stmt
        if s isa Core.EnterNode || s isa Core.PhiCNode || (s isa Expr && s.head === :leave)
            push!(problems, "leftover $(typeof(s)) in the structured IR")
        end
        if exceptions === :trap && (s isa Core.UpsilonNode ||
                                    (s isa Expr && s.head in (:pop_exception, :the_exception)))
            push!(problems, "exception statement $(s) survives :trap")
        end
    end
    tryops = eh_ops(sci, TryOp)
    if exceptions === :structured
        length(tryops) == nenter ||
            push!(problems, "$(length(tryops)) TryOps for $nenter EnterNodes")
    else
        all(op -> op.handler === nothing && op.scope !== nothing, tryops) ||
            push!(problems, "a TryOp with a handler survives :trap")
    end
    append!(problems, eh_compare(f, eh_lower(sci), inputs; exceptions,
                                 internal_throws = nameof(f) in JL_INTERNAL_THROWS))
    return problems
end

#=============================================================================
 try / catch / else / finally
=============================================================================#

# exceptions.jl "tail position"
jl_try_tail(x) = try jl_f(x) catch; 2 end
function jl_try_error_tail(x)
    try
        x > 0 && jl_error("hi")
        1
    catch
        2
    end
end
jl_try_catch_exc(x) = try x > 0 ? jl_error("hi") : x catch exc; exc end
jl_try_else_tail(x) = try jl_f(x) catch; 2 else 3 end
function jl_try_return_then_return(x)     # `try return 1 catch end; return 2`
    try
        return jl_f(x)
    catch
    end
    return 2
end
function jl_try_return_fallthrough(x)     # `try return 1 catch end` (nothing on catch)
    try
        return jl_f(x)
    catch
    end
end
function jl_while_catch_break(n)          # `while true; try error() catch; x = 2; break end end`
    x = -1
    i = n
    while true
        try
            i = jl_f(i)
        catch
            x = 2
            break
        end
    end
    (x, i)
end
function jl_while_try_break(n)            # `while true; try x = 2; break catch end end`
    x = -1
    while true
        try
            x = jl_f(n)
            break
        catch
            n -= 5
        end
    end
    x
end

# exceptions.jl "value position"
function jl_try_value(x)
    y = try
        jl_f(x)
    catch
        2
    end
    y
end
function jl_try_value_exc(x)
    y = try
        x > 0 && jl_error("hi")
        x
    catch exc
        exc
    end
    y
end
function jl_try_value_else(x)
    y = try
        jl_f(x)
    catch
        2
    else
        3
    end
    y
end

# exceptions.jl "not value/tail position"
function jl_try_assign(x)
    y = -1
    try
        jl_f(x)
        y = 1
    catch
        y = 2
    end
    y
end
function jl_try_assign_exc(x)
    y = -1
    try
        y = x > 0 ? jl_error("hi") : x
    catch exc
        y = exc
    end
    y
end
function jl_try_assign_else(x)
    y = -1
    try
        jl_f(x)
        y = 1
    catch
        y = 2
    else
        y = 3
    end
    y
end

# exceptions.jl "exception stack": the outer catch sees both exceptions.
function jl_exception_stack(x)
    try
        try
            jl_f(x)
        catch
            jl_error("ho")
        end
    catch
        a = []
        for e in current_exceptions()
            push!(a, e.exception)
        end
        a
    end
end
# (added) every exit from a catch pops the exception stack again.
function jl_exception_stack_balanced(x)
    try
        jl_f(x)
    catch
    end
    for i in 1:2
        try
            jl_g(x - 15)
        catch
            continue
        end
    end
    length(current_exceptions())
end

# exceptions.jl "try/finally"
function jl_finally_assign(x)
    y = -1
    try
        y = jl_f(x)
    finally
        jl_log(y)
        y = 2
    end
    y
end
function jl_try_finally_in_try(x)
    y = -1
    try
        try
            jl_f(x)
            y = 1
        finally
            y = 2
        end
    catch
    end
    y
end
function jl_nested_finally(x::Bool, c::Int)     # `nested_finally(a, x, b, c)`
    try
        try
            if x
                return jl_f(c)
            end
            c
        finally
            jl_log(1)
        end
    finally
        jl_log(2)
    end
end
jl_try_catch_finally(x) = try jl_f(x) catch; 2 finally jl_log(3) end

# compat.jl "semantics only"
function jl_catch_else_finally(x)       # `try; 1; catch e; e; else; 2; finally; 3; end`
    try
        jl_f(x)
    catch e
        e
    else
        jl_log(:else)
        2
    finally
        jl_log(:finally)
    end
end
jl_throw_catch(x) = try x > 0 && eh_throw(x); 0 catch e; e end          # `try throw(0) catch e; 1 end`
jl_try_finally_value(x) = try jl_f(x) finally jl_log(1) end           # `try 0 finally 1 end`
function jl_throw_catch_finally(x)      # `try throw(0) catch e; 1 finally 2 end`
    try
        x > 0 && eh_throw(x)
        0
    catch e
        e + 1
    finally
        jl_log(2)
    end
end
function jl_throw_catch_else(x)         # `try throw(0) catch e; 1 else 2 end`
    try
        x > 0 && eh_throw(x)
    catch e
        1
    else
        2
    end
end
function jl_throw_catch_else_finally(x) # `try throw(0) catch e; 1 else 2 finally 3 end`
    try
        x > 0 && eh_throw(x)
    catch e
        1
    else
        2
    finally
        jl_log(3)
    end
end
function jl_finally_before_catch(x)     # `try throw(0) finally 1 catch e; 2 end`
    try
        x > 0 && eh_throw(x)
        0
    finally
        jl_log(1)
    catch e
        2
    end
end
function jl_catch_rethrow_e(x)          # compat.jl's harness: `catch e; ...; rethrow(e)`
    try
        jl_f(x)
    catch e
        jl_log(:caught)
        x > 15 && rethrow(e)
        -1
    end
end

# exceptions_ir.jl
function jl_return_try_catch(x)         # "Return from inside try/catch"
    try
        jl_f(x)
        return x
    catch
        jl_g(x - 20)
        return -x
    end
end
function jl_return_try_catch_const(x)   # "... with simple return vals"
    try
        jl_f(x)
        return 10
    catch
        jl_log(:g)
        return 20
    end
end
function jl_return_multi_try(x)         # "Return from multiple try + try/catch"
    try
        try
            jl_f(x)
            return 10
        catch
            jl_g(x - 20)
            return 20
        end
    catch
    end
end
function jl_return_multi_catch(x)       # "Return from multiple catch + try/catch"
    try
        jl_f(x)
    catch
        try
            jl_g(x - 20)
            return 10
        catch
            return 20
        end
    end
end
function jl_try_else_stmt(x)            # "try/catch/else, not value/tail"
    try
        jl_f(x)
    catch
        jl_log(:b)
    else
        jl_log(:c)
    end
    x * 3
end
function jl_finally_stmt(x)             # "basic try/finally, not value/tail"
    try
        jl_f(x)
    finally
        jl_log(x)
    end
    x * 3
end
function jl_finally_break(x)            # "try/finally + break"
    while true
        try
            jl_f(x)
            break
        finally
            jl_log(x)
        end
    end
    x
end
function jl_nested_finally_return(x)    # "Nested finally blocks"
    try
        try
            if x > 5
                return jl_f(x)
            end
            jl_g(x)
        finally
            jl_log(:c)
        end
    finally
        jl_log(:d)
    end
end
jl_exc_object(x) = try jl_f(x) catch exc; jl_log(exc); 0 end          # "Access to the exception object"

# demo.jl
function jl_nested_catch_else(x::Int, y::Bool)
    z = try
        try
            jl_f(x)
            1
        catch exc
            if y
                return 2
            end
            jl_g(x - 20)
            3
        else
            4
        end
    catch
        5
    end
    z
end
function jl_let_nested_finally_return(x)
    try
        try
            return jl_f(x)
        finally
            jl_log(1)
        end
    finally
        jl_log(2)
    end
    x
end
function jl_goto_retry(n)               # `@label foo; try ... @goto foo catch ... @goto foo end`
    i = n
    j = 1
    @label foo
    try
        jl_log(i)
        i = jl_f(i)
        if i <= 2
            @goto foo
        end
    catch exc
        j = j + 1
        if j <= 2
            i -= 9
            @goto foo
        end
    end
    (i, j)
end

# destructuring.jl: no assignment happens when the right-hand side throws.
function jl_destructure_throw(x)
    local a
    p, q, r = 1, 2, 3
    side = false
    x > 0 && (a = x)
    exc = try
        f() = (side = true)
        (p, q, r) = (100, a, f())
        nothing
    catch e
        e isa UndefVarError
    end
    (p, q, r, side, exc)
end
function jl_destructure_isdefined(x)
    local p, q, a
    x > 0 && (a = x)
    try
        (p, q) = (1, a)
    catch
    end
    @isdefined(p)
end

#=============================================================================
 Nested trys, exits across trys, finally with return/break, rethrow
=============================================================================#

# (added) `continue`/`break` across two trys, finally on the way out.
function jl_continue_nested_finally(n)
    s = 0
    for i in 1:n
        try
            try
                isodd(i) && continue
                s += jl_f(i)
            finally
                jl_log(i)
            end
        catch
            s -= 1
        end
    end
    s
end
function jl_break_nested_finally(n)
    s = 0
    i = 0
    while true
        i += 1
        try
            try
                s += jl_f(i)
                s > n && break
            finally
                jl_log(-i)
            end
        catch
            break
        end
    end
    (s, i)
end
# (added) `break` and `return` from inside finally (allowed, if unusual).
function jl_break_in_finally(n)
    s = 0
    for i in 1:n
        try
            s += jl_f(i)
        finally
            i == 3 && break
        end
    end
    s
end
function jl_return_in_finally(x)
    try
        jl_f(x)
    finally
        x > 5 && return -x
    end
end
# (added) `@goto` from an inner try or its catch to a label in the outer try.
function jl_goto_between_trys(x)
    try
        try
            jl_f(x)
            x > 5 && @goto mid
            jl_log(1)
        catch
            jl_log(2)
            @goto mid
        end
        jl_log(3)
        @label mid
        jl_g(x - 15)
    catch
        -1
    end
end
# (added) a rethrow from a nested catch, caught by the outer try.
function jl_nested_rethrow(x)
    try
        try
            jl_f(x)
        catch e
            x > 15 && rethrow()
            jl_g(x - 13)
        end
    catch e
        e isa DomainError ? -2 : -1
    end
end
# (added) catch inside finally inside catch.
function jl_catch_in_finally(x)
    try
        jl_f(x)
    catch
        try
            jl_g(-x)
        finally
            try
                jl_f(x - 5)
            catch
                jl_log(:inner)
            end
        end
    end
end
# (added) values live into a catch from inside a loop inside the try.
function jl_live_into_catch_loop(n)
    s = 0
    last = 0
    try
        for i in 1:n
            last = i
            s += jl_f(i)
        end
    catch
        return (s, last, -1)
    end
    (s, last, 0)
end
# (added) the caught exception is used after the try.
function jl_exc_after_try(x)
    caught = nothing
    try
        jl_f(x)
    catch e
        caught = e
    end
    caught === nothing ? 0 : caught
end
# (added) `return` from a catch nested two catches deep inside an outer try: the
# exit leaves the outer try, then pops both exceptions.
function jl_return_two_catches_deep(x)
    try
        try
            jl_f(x)
        catch
            try
                jl_g(x - 15)
            catch
                return -length(current_exceptions())
            end
        end
    catch
        return -100
    end
    length(current_exceptions())
end
function jl_return_three_catches_deep(x)
    try
        try
            jl_f(x)
        catch
            try
                jl_g(x - 15)
            catch
                try
                    jl_g(x - 25)
                catch
                    x > 28 && return -length(current_exceptions())
                end
            end
        end
        jl_f(x - 20)
    catch
        return -100
    end
    length(current_exceptions())
end
# (added) `return`, `break` and `continue` all leaving through one finally.
function jl_finally_all_exits(n)
    for i in 1:n
        try
            try
                i == 2 && n > 4 && return i
                i == 3 && break
                isodd(i) && continue
                jl_f(i * 3)
            finally
                jl_log(i)
            end
        catch
            jl_log(-i)
        end
    end
    -1
end
# (added) a variable assigned in an inner try and its catch, read in the outer catch.
function jl_upsilon_through_catch(x)
    y = 0
    try
        try
            y = 1
            jl_f(x)
            y = 2
        catch
            y = 3
            jl_g(-x)
            y = 4
        end
        y = 5
        jl_f(x - 5)
    catch
        return y
    end
    y
end
# (added) a try whose result is a Union.
function jl_try_union(x)
    r = try
        x > 5 ? jl_f(x) : nothing
    catch
        missing
    end
    r === nothing ? -1 : r === missing ? -2 : r
end
# (added) `Base.wait_forever`'s shape: a loop around a try whose body never
# completes normally, so all code after the handler is dominated by the catch.
function jl_endless_try_body(n)
    i = 0
    k = 0
    while true
        try
            while true
                i = jl_f(i)
            end
        catch
            k += 1
            k > n && return (i, k)
            i = k
        end
    end
end
function jl_try_always_throws(n)
    s = 0
    for i in 1:n
        try
            jl_log(i)
            jl_error("always")
        catch
            s += i
        end
    end
    s
end
function jl_after_throwing_try(x)
    try
        jl_error("x")
    catch
        jl_log(1)
    end
    x > 2 ? jl_f(x) : jl_g(x)
end
function jl_throwing_try_two_exits(x)
    x < -5 && @goto big
    if x > 0
        try
            jl_log(x)
            jl_error("x")
        catch
        end
        x > 5 && @goto big
    end
    return 0
    @label big
    return 1
end
# (added) a catch body inside a comprehension, inlined into the caller's loop.
jl_comp_try(n) = [try jl_f(3x) catch; -x end for x in 1:n]
# (added) a closure capturing (and boxing) a variable assigned in a catch.
function jl_closure_catch(x)
    y = 0
    get_y() = y
    try
        jl_f(x)
    catch
        y = -1
    end
    get_y() + y
end
# (added) a try in a short-circuit while condition.
function jl_while_and_try(n)
    i = 0
    while i < n && (try jl_f(2i) > 0 catch; false end)
        i += 1
    end
    i
end

#=============================================================================
 Branching, short-circuit conditions, symbolic goto
=============================================================================#

# branching.jl "Tail position" / "Value required" / "Side effects"
jl_if_tail(a::Bool) = if a; 1 end
jl_if_else_tail(a::Bool) = if a; 1 else 2 end
jl_elseif_tail(a::Bool, b::Bool) = if a; 1 elseif b; 2 else 3 end
function jl_if_value(a::Bool, b::Bool)
    x = if a
        1
    elseif b
        2
    end
    x
end
function jl_if_side(a::Bool, b::Bool)
    x = nothing
    if a
        x = 1
    elseif b
        x = 2
    else
        x = 3
    end
    x
end
function jl_block_cond(a::Bool)         # `if begin; x = 2; a; end; x end`
    if (x = 2; a)
        x
    end
end

# branching.jl "`&&` and `||` chains"
jl_and_chain(a::Bool, b::Bool) = a && b && "hi"
jl_or_chain(a::Bool, b::Bool) = a || b || "hi"
function jl_and_or_value(a::Bool, b::Bool)
    z = a && "hi"
    w = b || "ho"
    (z, w)
end

# branching_ir.jl "Blocks compile directly to branches": `if (a; b && c) d end`
function jl_block_and_cond(a::Bool, b::Bool, c::Bool)
    if (jl_log(a); b && c)
        jl_log(:d)
    end
end
# (added) nested short-circuits in conditions, around a throwing call.
function jl_complex_cond(x)
    if (x > 3 && isodd(x)) || (x < 0 && jl_g(x) > 0)
        1
    elseif !(x == 2 || x == 4) && jl_f(x) > 5
        2
    else
        3
    end
end
function jl_cond_in_try(x)
    try
        if x > 0 && jl_f(x) > 5 || x < -5
            return 1
        end
        0
    catch
        -1
    end
end

# branching_ir.jl "symbolic goto forward/backward jump"
function jl_goto_forward(x)
    jl_log(1)
    x > 0 && @goto foo
    jl_log(2)
    @label foo
    x
end
function jl_goto_backward(n)            # branching.jl "symbolic goto/label"
    a = Int[]
    i = 1
    @label foo
    push!(a, i)
    i = i + 1
    if i <= n
        @goto foo
    end
    a
end

# branching_ir.jl "Jumping out of try and catch blocks using @goto"
function jl_goto_out_of_try_catch(x)
    try
        jl_f(x)
        x > 5 && @goto lab
        jl_log(:b)
    catch
        jl_log(:c)
        x > 15 && @goto lab
        jl_log(:d)
    end
    return 0
    @label lab
    return 1
end
# branching_ir.jl "Jumping out of nested try/catch and catch/try"
function jl_goto_out_nested(x)
    try
        try
            jl_f(x)
        catch
            jl_log(:b)
            x > 15 && @goto lab
            jl_log(:c)
        end
        jl_g(x - 30)
    catch
        try
            jl_log(:d)
            x > 25 && @goto lab
            jl_g(-x)
        catch
        end
    end
    return 0
    @label lab
    return 1
end

# (added) irreducible loops built from gotos: a loop with two entries.
function jl_irreducible(x)
    i = 0
    x > 0 && @goto b
    @label a
    i += 1
    jl_log(i)
    @label b
    i += 2
    i < x && @goto a
    i
end
# (added) a try inside an irreducible loop.
function jl_irreducible_try(x)
    i = 0
    x > 0 && @goto b
    @label a
    i += 1
    @label b
    try
        i = jl_f(i)
    catch
        return -i
    end
    i < x && @goto a
    i
end
# (added) an irreducible loop inside a try body.
function jl_irreducible_in_try(x)
    try
        i = 0
        x > 0 && @goto b
        @label a
        i = jl_f(i)
        @label b
        i += 2
        i < x && @goto a
        i
    catch
        -1
    end
end
# (added) a catch that jumps into the middle of an irreducible loop.
function jl_irreducible_catch_entry(x)
    i = 0
    try
        jl_f(x)
    catch
        @goto b
    end
    @label a
    i += 1
    @label b
    i += 2
    i < 20 && @goto a
    i
end
# (added) the body and the catch enter an irreducible loop at different labels.
function jl_irreducible_two_try_exits(x)
    i = x
    try
        i = jl_f(x)
        @goto a
    catch
        @goto b
    end
    @label a
    i += 1
    jl_log(i)
    @label b
    i += 2
    i < 30 && @goto a
    i
end
# (added) a try in each half of an irreducible loop.
function jl_irreducible_two_trys(x)
    i = 0
    x > 3 && @goto b
    @label a
    try
        i = jl_f(i)
    catch
        i = 2i
    end
    @label b
    try
        i = jl_g(i - 7) ÷ 2 + 1
    catch
        return -i
    end
    i < x && @goto a
    i
end

#=============================================================================
 Loops
=============================================================================#

# loops.jl "while loops"
function jl_while_push(n)
    a = Int[]
    i = 0
    while i < n
        i = i + 1
        push!(a, i)
    end
    a
end
function jl_while_break(n)
    a = Int[]
    i = 0
    while i < n
        i = i + 1
        if i == 3
            break
        end
        push!(a, i)
    end
    a
end
function jl_while_continue(n)
    a = Int[]
    i = 0
    while i < n
        i = i + 1
        if isodd(i)
            continue
        end
        push!(a, i)
    end
    a
end
# loops_ir.jl "While loop with short circuit condition"
function jl_while_and(n)
    i = 0
    while i < n && i != 7
        i += 1
    end
    i
end
# loops_ir.jl "While loop with with break and continue" (dead code after both)
function jl_while_break_continue(n)
    i = 0
    while i < n
        jl_log(1)
        break
        jl_log(2)
        continue
        jl_log(3)
    end
    i
end
# (added) complex while conditions: a block, `||` of calls, a try.
function jl_while_complex(n)
    i = 0
    s = 0
    while (i += 1; i < n) && (isodd(i) || i % 3 == 0 || jl_f(i) < 5)
        s += i
    end
    (i, s)
end
function jl_while_try_cond(n)
    i = 0
    while (try jl_f(i) catch; 100 end) < n
        i += 1
    end
    i
end

# loops.jl "for loops"
function jl_for_push(n)
    a = Int[]
    for i = 1:n
        push!(a, i)
    end
    a
end
function jl_for_underscore(n)
    a = Int[]
    for _ = 1:n
        push!(a, 1)
    end
    a
end
function jl_for_break(n)
    a = Int[]
    for i = 1:n
        if i == 3
            break
        end
        push!(a, i)
    end
    a
end
function jl_for_nested_break(n)          # "Break from inner nested loop"
    a = Tuple{Int,Int}[]
    for i in 1:n
        for j in 3:4
            push!(a, (i, j))
            j == 6 && break
        end
    end
    a
end
function jl_for_continue(n)
    a = Int[]
    for i = 1:n
        if isodd(i)
            continue
        end
        push!(a, i)
    end
    a
end
function jl_for_loopvar_assign(n)       # "Loop variable scope"
    a = Int[]
    for i = 1:n
        push!(a, i)
        i = 100
    end
    a
end
function jl_for_shadow(n)
    i = 100
    for i = 1:n
    end
    i
end
function jl_for_outer(n)
    i = 100
    for outer i = 1:n
        nothing
    end
    i
end
function jl_for_destructure(n)          # "Fancy for loop left hand side"
    a = Tuple{Int,Int}[]
    i = 100
    j = 200
    for (i, j) in [(1, 2), (n, 2n)]
        push!(a, (i, j))
    end
    (a, i, j)
end
function jl_for_number(n)               # compat.jl `for x in 1; x+=1; if true continue elseif false break end end`
    s = 0
    for x in n
        x += 1
        if x > 3
            s += x
            continue
        elseif x < 0
            break
        end
        s -= 1
    end
    s
end

# loops.jl "multidimensional for loops"
function jl_for2(n)
    a = Tuple{Int,Int}[]
    for i = 1:n, j = 3:4
        push!(a, (i, j))
    end
    a
end
function jl_for2_break(n)
    a = Tuple{Int,Int}[]
    for i = 1:n, j = 3:4
        push!(a, (i, j))
        break
    end
    a
end
function jl_for2_continue_outer(n)
    a = Tuple{Int,Int}[]
    for i = 1:n, j = 3:4
        if isodd(i)
            continue
        end
        push!(a, (i, j))
    end
    a
end
function jl_for2_continue_inner(n)
    a = Tuple{Int,Int}[]
    for i = 1:2, j = 1:n
        if isodd(j)
            continue
        end
        push!(a, (i, j))
    end
    a
end
function jl_for2_loopvar(n)
    a = Tuple{Int,Int}[]
    for i = 1:n, j = 3:4
        push!(a, (i, j))
        i = 100
    end
    a
end
function jl_for2_outer(n)
    i = 100
    j = 200
    for outer i = 1:n, j = 3:4
        nothing
    end
    (i, j)
end
function jl_for2_outer_inner(n)
    i = 100
    j = 200
    for i = 1:2, outer j = 3:n
        nothing
    end
    (i, j)
end
# loops_ir.jl "Syntax sugar for nested for loop" (assigning a copy of `x`)
function jl_for2_copy(n)
    s = 0
    for x in 1:n, y in 1:2
        x = 10
        s += x + y
    end
    s
end

# (added) trys and loops nested every way, with values crossing both.
function jl_try_for_try(n)
    s = 0
    try
        for i in 1:n
            try
                s += jl_f(i)
            catch
                s += jl_g(5 - i)
            end
        end
    catch
        s = -s
    end
    s
end
function jl_try_while_goto(n)           # `@goto` out of a loop and a try
    i = 0
    try
        while true
            i += 1
            for j in 1:i
                j * i > n && @goto done
            end
            jl_f(i)
        end
    catch
        return -i
    end
    @label done
    i
end

#=============================================================================
 Generators, comprehensions, closures
=============================================================================#

# generators.jl
jl_gen_sq(n) = collect(x^2 for x in 1:n)
jl_gen_filter(n) = collect(x for x in 1:n if isodd(x))
jl_gen_zip(n) = collect((y, x) for (x, y) in zip(1:n, 2:n+1) if y != 3)
jl_gen_product(n) = collect((x, y) for x in 1:n, y in 1:2)
jl_gen_flatten(n) = collect((x, y, z) for x in 1:n, y in 4:5 for z in 6:7)
jl_gen_dup(n) = collect(x for x in 1:n for x in 1:2)
jl_gen_protect(n) = collect((z = y; y = 100; z) for y in 1:n for x in 1:2)
jl_typed_comp(n) = Tuple{Int,Int}[(x, y) for x in 1:2, y in 1:n]
jl_comp3(n) = [(x, y, z) for x in 1:n for y in 4:5 for z in 6:7]
jl_typed_comp_filter(n) = Int[x + y for x in 1:n, y in 3:4 if isodd(x + y)]   # compat.jl
# (added) a throwing comprehension, and a generator inside a try.
jl_comp_throw(n) = [jl_f(x) for x in 1:n]
jl_gen_in_try(n) = try sum(jl_f(x) for x in 1:n; init=0) catch; -1 end

# closures_ir.jl / closures.jl
function jl_closure_assign(x)           # "arguments are captured into a closure and assigned"
    g() = (x = 10)
    x > 5 && g()
    x
end
function jl_closure_isdefined(x)        # "Closure captures with `isdefined`"
    function g()
        z = 3
        (@isdefined(x), @isdefined(y), @isdefined(z))
    end
    r1 = g()
    x > 0 && (y = 2)
    (r1, g(), @isdefined(y), @isdefined(x))
end
function jl_closure_typed(x)            # "Closure capturing a typed local"
    T = Float64
    y::T = x
    f() = (y = 2.0)
    x > 0 && f()
    y
end
function jl_mutual_recursion(n)         # "Mutually recursive closures"
    ra(k) = k <= 0 ? 0 : rb(k - 1) + 1
    rb(k) = k <= 0 ? 0 : ra(k - 1) + 2
    ra(n)
end
jl_do_block(n) = map(1:n) do y          # "`do` blocks"
    y > 3 ? y + 2 : y
end
jl_anon(x) = (y -> y * y)(x)            # "Anonymous function syntax with ->"
# (added) closures with control flow of their own, inlined into the caller.
function jl_closure_control(n)
    acc = 0
    add(i) = isodd(i) ? (acc += i) : (acc -= 1)
    for i in 1:n
        add(i)
    end
    acc
end
function jl_closure_try(x)
    h = y -> try jl_f(y) catch; -y end
    h(x) + h(x + 5)
end

#=============================================================================
 Scoping: let, local, global, typed declarations
=============================================================================#

function jl_let_shadow(x)               # scopes.jl `let x = x + 1; y = x end`
    y = 0
    let x = x + 1
        y = x
    end
    (x, y)
end
function jl_let_rebind(x)               # scopes_ir.jl `let x = f(x), x = g(x) end`
    let x = jl_f(x), x = jl_g(x - 5)
        x
    end
end
function jl_let_closure_outer(x)        # scopes.jl: a closure in `let` sees the outer `y`
    y = x
    let f() = y
        y2 = -x
        f() + y2
    end
end
function jl_arg_decl(x)                 # decls_ir.jl "Type decl on function argument"
    x::Int = x
    x = x / 2
    x
end
function jl_typed_local(x)              # assignments_ir.jl "declarations of typed locals"
    local y::Int = jl_f(x)
    y = y / 2
    y
end
# (added) `local` in a loop body is fresh in every iteration.
function jl_local_in_loop(n)
    s = 0
    for i in 1:n
        local t
        if isodd(i)
            t = i
        end
        s += @isdefined(t) ? t : 100
    end
    s
end
# (added) `local` in a loop body, assigned in a catch.
function jl_local_in_loop_catch(n)
    s = 0
    for i in 1:n
        local t
        try
            jl_f(3i)
        catch
            t = i
        end
        s += @isdefined(t) ? t : 0
    end
    s
end
# (added) typed and untyped globals updated in loops.
function jl_global_in_loop(n)
    global jl_counter = 0
    for i in 1:n
        jl_counter += i
    end
    jl_counter
end
function jl_global_try(n)
    global jl_untyped = 0
    for i in 1:n
        try
            jl_untyped += jl_f(i)
        catch
            jl_untyped = -jl_untyped
        end
    end
    jl_untyped
end

#=============================================================================
 ScopedValues and other EnterNode scopes (added: the suite has none)
=============================================================================#

function jl_with_nested(x)
    @with JL_SV => x begin
        a = jl_readsv()
        @with JL_SV => x + 1 begin
            a + jl_readsv() + jl_f(x)
        end
    end
end
jl_with_two(x) = @with JL_SV => x JL_SV2 => 2x begin; jl_readsv() + jl_readsv2() end
function jl_with_in_loop(n)
    s = 0
    for i in 1:n
        s += @with JL_SV => i jl_readsv()
    end
    s
end
function jl_with_in_try(x)
    try
        @with JL_SV => x begin
            jl_f(jl_readsv())
        end
    catch
        jl_readsv()
    end
end
function jl_try_in_with(x)
    @with JL_SV => x begin
        try
            jl_f(x)
        catch
            -jl_readsv()
        end
    end
end
function jl_with_return(x)
    r = @with JL_SV => x begin
        x > 5 && return jl_readsv() * 2
        jl_f(x)
    end
    r + jl_readsv()
end
function jl_with_break(n)
    s = 0
    for i in 1:n
        @with JL_SV => i begin
            s += jl_readsv()
            s > 10 && break
        end
    end
    (s, jl_readsv())
end
jl_with_function(x) = with(() -> jl_readsv() + jl_f(x), JL_SV => x)
function jl_with_in_catch(x)
    try
        jl_f(x)
    catch
        @with JL_SV => -x begin
            jl_readsv()
        end
    end
end

#=============================================================================
 Misc
=============================================================================#

function jl_preserve(x)                 # misc_ir.jl "GC.@preserve support"
    a = Ref(x)
    GC.@preserve a begin
        x > 0 ? a[] + 1 : jl_g(x)
    end
end

#----------------------------------------------------------------------------- cases

const JL_INTS = [(0,), (1,), (3,), (7,), (12,), (20,), (30,), (-3,)]
const JL_BOOLS2 = [(a, b) for a in (false, true) for b in (false, true)]
const JL_SMALL = [(0,), (1,), (3,), (6,), (12,)]

const JL_CASES = Any[
    # try/catch/else/finally
    (jl_try_tail, JL_INTS), (jl_try_error_tail, JL_INTS), (jl_try_catch_exc, JL_INTS),
    (jl_try_else_tail, JL_INTS), (jl_try_return_then_return, JL_INTS),
    (jl_try_return_fallthrough, JL_INTS), (jl_while_catch_break, JL_INTS),
    (jl_while_try_break, JL_INTS), (jl_try_value, JL_INTS), (jl_try_value_exc, JL_INTS),
    (jl_try_value_else, JL_INTS), (jl_try_assign, JL_INTS), (jl_try_assign_exc, JL_INTS),
    (jl_try_assign_else, JL_INTS), (jl_exception_stack, JL_INTS),
    (jl_exception_stack_balanced, JL_INTS), (jl_finally_assign, JL_INTS),
    (jl_try_finally_in_try, JL_INTS),
    (jl_nested_finally, [(true, 5), (true, 100), (false, 5), (false, 100)]),
    (jl_try_catch_finally, JL_INTS), (jl_catch_else_finally, JL_INTS),
    (jl_throw_catch, JL_INTS), (jl_try_finally_value, JL_INTS),
    (jl_throw_catch_finally, JL_INTS), (jl_throw_catch_else, JL_INTS),
    (jl_throw_catch_else_finally, JL_INTS), (jl_finally_before_catch, JL_INTS),
    (jl_catch_rethrow_e, JL_INTS), (jl_return_try_catch, JL_INTS),
    (jl_return_try_catch_const, JL_INTS), (jl_return_multi_try, JL_INTS),
    (jl_return_multi_catch, JL_INTS), (jl_try_else_stmt, JL_INTS),
    (jl_finally_stmt, JL_INTS), (jl_finally_break, JL_INTS),
    (jl_nested_finally_return, JL_INTS), (jl_exc_object, JL_INTS),
    (jl_nested_catch_else, [(x, y) for x in (1, 12, 25) for y in (false, true)]),
    (jl_let_nested_finally_return, JL_INTS), (jl_goto_retry, JL_INTS),
    (jl_destructure_throw, JL_INTS), (jl_destructure_isdefined, JL_INTS),
    # nested trys, exits across trys, rethrow
    (jl_continue_nested_finally, JL_SMALL), (jl_break_nested_finally, JL_SMALL),
    (jl_break_in_finally, JL_SMALL), (jl_return_in_finally, JL_INTS),
    (jl_goto_between_trys, JL_INTS), (jl_nested_rethrow, JL_INTS),
    (jl_catch_in_finally, JL_INTS), (jl_live_into_catch_loop, JL_SMALL),
    (jl_exc_after_try, JL_INTS), (jl_return_two_catches_deep, JL_INTS),
    (jl_return_three_catches_deep, JL_INTS), (jl_finally_all_exits, JL_SMALL),
    (jl_upsilon_through_catch, JL_INTS), (jl_try_union, JL_INTS),
    (jl_comp_try, JL_SMALL), (jl_closure_catch, JL_INTS), (jl_while_and_try, JL_INTS),
    (jl_endless_try_body, JL_SMALL), (jl_try_always_throws, JL_SMALL),
    (jl_after_throwing_try, JL_INTS), (jl_throwing_try_two_exits, JL_INTS),
    # branching, short-circuit, goto
    (jl_if_tail, [(false,), (true,)]), (jl_if_else_tail, [(false,), (true,)]),
    (jl_elseif_tail, JL_BOOLS2), (jl_if_value, JL_BOOLS2), (jl_if_side, JL_BOOLS2),
    (jl_block_cond, [(false,), (true,)]),
    (jl_and_chain, JL_BOOLS2), (jl_or_chain, JL_BOOLS2), (jl_and_or_value, JL_BOOLS2),
    (jl_block_and_cond, [(a, b, c) for a in (false, true) for b in (false, true) for c in (false, true)]),
    (jl_complex_cond, JL_INTS), (jl_cond_in_try, JL_INTS),
    (jl_goto_forward, JL_INTS), (jl_goto_backward, JL_SMALL),
    (jl_goto_out_of_try_catch, JL_INTS), (jl_goto_out_nested, JL_INTS),
    (jl_irreducible, JL_INTS), (jl_irreducible_try, JL_INTS),
    (jl_irreducible_in_try, JL_INTS), (jl_irreducible_catch_entry, JL_INTS),
    (jl_irreducible_two_try_exits, JL_INTS), (jl_irreducible_two_trys, JL_INTS),
    # loops
    (jl_while_push, JL_SMALL), (jl_while_break, JL_SMALL), (jl_while_continue, JL_SMALL),
    (jl_while_and, JL_INTS), (jl_while_break_continue, JL_SMALL),
    (jl_while_complex, JL_INTS), (jl_while_try_cond, JL_INTS),
    (jl_for_push, JL_SMALL), (jl_for_underscore, JL_SMALL), (jl_for_break, JL_SMALL),
    (jl_for_nested_break, JL_SMALL), (jl_for_continue, JL_SMALL),
    (jl_for_loopvar_assign, JL_SMALL), (jl_for_shadow, JL_SMALL), (jl_for_outer, JL_SMALL),
    (jl_for_destructure, JL_SMALL), (jl_for_number, JL_INTS),
    (jl_for2, JL_SMALL), (jl_for2_break, JL_SMALL), (jl_for2_continue_outer, JL_SMALL),
    (jl_for2_continue_inner, JL_SMALL), (jl_for2_loopvar, JL_SMALL),
    (jl_for2_outer, JL_SMALL), (jl_for2_outer_inner, JL_SMALL), (jl_for2_copy, JL_SMALL),
    (jl_try_for_try, JL_SMALL), (jl_try_while_goto, [(0,), (5,), (20,), (60,), (200,)]),
    # generators, comprehensions, closures
    (jl_gen_sq, JL_SMALL), (jl_gen_filter, JL_SMALL), (jl_gen_zip, JL_SMALL),
    (jl_gen_product, JL_SMALL), (jl_gen_flatten, JL_SMALL), (jl_gen_dup, JL_SMALL),
    (jl_gen_protect, JL_SMALL), (jl_typed_comp, JL_SMALL), (jl_comp3, JL_SMALL),
    (jl_typed_comp_filter, JL_SMALL), (jl_comp_throw, JL_SMALL), (jl_gen_in_try, JL_SMALL),
    (jl_closure_assign, JL_INTS), (jl_closure_isdefined, JL_INTS),
    (jl_closure_typed, JL_INTS), (jl_mutual_recursion, JL_SMALL), (jl_do_block, JL_SMALL),
    (jl_anon, JL_INTS), (jl_closure_control, JL_SMALL), (jl_closure_try, JL_INTS),
    # scoping
    (jl_let_shadow, JL_INTS), (jl_let_rebind, JL_INTS), (jl_let_closure_outer, JL_INTS),
    (jl_arg_decl, JL_INTS), (jl_typed_local, JL_INTS), (jl_local_in_loop, JL_SMALL),
    (jl_local_in_loop_catch, JL_SMALL), (jl_global_in_loop, JL_SMALL),
    (jl_global_try, JL_SMALL),
    # ScopedValues
    (jl_with_nested, JL_INTS), (jl_with_two, JL_INTS), (jl_with_in_loop, JL_SMALL),
    (jl_with_in_try, JL_INTS), (jl_try_in_with, JL_INTS), (jl_with_return, JL_INTS),
    (jl_with_break, JL_SMALL), (jl_with_function, JL_INTS), (jl_with_in_catch, JL_INTS),
    # misc
    (jl_preserve, JL_INTS),
]

@testset "JuliaLowering control-flow cases" begin

@testset "$(nameof(f))" for (f, inputs) in JL_CASES
    for exceptions in (:structured, :trap), promote in (true, false)
        @test isempty(jl_roundtrip(f, inputs; exceptions, promote))
    end
end

# The same cases in less tidy IR. With `Any` arguments, dispatch is dynamic and phis
# are typed narrower than the values that feed them. Stopping the optimizer after
# its first or second compaction (before or after inlining) keeps the redundant
# blocks, `isa`/`=== nothing` refinements and unions it would later fold away.
@testset "$(nameof(f)), Any arguments" for (f, inputs) in JL_CASES
    argtypes = Tuple{map(_ -> Any, first(inputs))...}
    for exceptions in (:structured, :trap)
        @test isempty(jl_roundtrip(f, inputs; exceptions, argtypes))
    end
end
@static if VERSION >= v"1.12-"   # the pass numbering of `optimize_until`
    @testset "$(nameof(f)), optimize_until=$stage" for (f, inputs) in JL_CASES, stage in (3, 5)
        @test isempty(jl_roundtrip(f, inputs; optimize_until=stage))
    end
end

end  # JuliaLowering control-flow cases
