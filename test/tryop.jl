#=============================================================================
 TryOp and LeaveOp in the structured IR, on hand-built IR: the accessors,
 printing, use tracking, validation and lowering back to `IRCode`. Lifting
 try/catch from Julia IR is tested in test/exceptions.jl.
=============================================================================#

using IRStructurizer: TryOp, LeaveOp, IRValue, BlockArgument

@noinline tryop_g(x::Int) = x > 10 ? error("big: $x") : x + 1

"""`f(x) = try tryop_g(x) catch; -x end`, by hand: the upsilon %1 feeds the
handler's argument with `x`; the handler pops the exception and yields `-x`."""
function tryop_sci()
    entry = Block()
    push!(entry, 1, Core.UpsilonNode(Core.Argument(2)), Nothing)
    body = Block()
    push!(body, 3, Expr(:call, tryop_g, Core.Argument(2)), Int)
    body.terminator = YieldOp(IRValue[SSAValue(3)])
    handler = Block()
    arg = BlockArgument(1, Int)
    push!(handler.args, arg)
    push!(handler, 4, Expr(:pop_exception, SSAValue(2)), Nothing)
    push!(handler, 6, Expr(:call, Core.Intrinsics.neg_int, arg), Int)
    handler.terminator = YieldOp(IRValue[SSAValue(6)])
    push!(entry, 2, TryOp(body, handler, [[1]]), Tuple{Int})
    push!(entry, 5, Expr(:call, Core.getfield, SSAValue(2), 1), Int)
    entry.terminator = ReturnNode(SSAValue(5))
    return StructuredIRCode(Any[Any, Int], Any[], entry, 10)
end

"""`f(x) = try (x < 0 && exit with -3x); tryop_g(x) catch; 0 end`, by hand: the exit
from the IfOp arm leaves the try body with a `LeaveOp`."""
function tryop_leave_sci()
    entry = Block()
    body = Block()
    push!(body, 3, Expr(:call, Core.Intrinsics.slt_int, Core.Argument(2), 0), Bool)
    then_blk = Block()
    push!(then_blk, 9, Expr(:call, Core.Intrinsics.mul_int, Core.Argument(2), -3), Int)
    then_blk.terminator = LeaveOp(IRValue[SSAValue(9)])
    else_blk = Block()
    else_blk.terminator = YieldOp()
    push!(body, 8, IfOp(SSAValue(3), then_blk, else_blk), Nothing)
    push!(body, 10, Expr(:call, GlobalRef(@__MODULE__, :tryop_g), Core.Argument(2)), Int)
    body.terminator = YieldOp(IRValue[SSAValue(10)])
    handler = Block()
    push!(handler, 4, Expr(:pop_exception, SSAValue(2)), Nothing)
    handler.terminator = YieldOp(IRValue[0])
    push!(entry, 2, TryOp(body, handler), Tuple{Int})
    push!(entry, 5, Expr(:call, Core.getfield, SSAValue(2), 1), Int)
    entry.terminator = ReturnNode(SSAValue(5))
    return StructuredIRCode(Any[Any, Int], Any[], entry, 20)
end

tryop_of(sci) = only(e.stmt for (_, e) in sci.entry.body if e.stmt isa TryOp)

@testset "TryOp and LeaveOp" begin

@testset "accessors" begin
    sci = tryop_sci()
    op = tryop_of(sci)
    @test op isa IRStructurizer.ControlFlowOp
    @test IRStructurizer.blocks(op) == (op.body, op.handler)
    @test isempty(IRStructurizer.operands(op))
    @test op.upsilons == [[1]]
    # A scope-only try has no handler region, and its scope is its operand.
    scoped = TryOp(Block(), nothing, Vector{Int}[], Core.Argument(2))
    @test IRStructurizer.blocks(scoped) == (scoped.body,)
    @test IRStructurizer.operands(scoped) == Any[Core.Argument(2)]
    @test !IRStructurizer.has_scope(TryOp(Block(), Block()))
    # A `nothing` scope operand (an empty scope) is still an operand.
    emptied = TryOp(Block(), nothing, Vector{Int}[], nothing)
    @test IRStructurizer.has_scope(emptied)
    @test IRStructurizer.operands(emptied) == Any[nothing]
    # LeaveOp carries values like a BreakOp.
    leave = LeaveOp(IRValue[SSAValue(3)])
    @test IRStructurizer.operands(leave) == IRValue[SSAValue(3)]
    @test isempty(IRStructurizer.operands(LeaveOp()))
    @test leave isa IRStructurizer.Terminator
end

@testset "printing" begin
    @test @filecheck begin
        @check "%1 = ϒ (_2)"
        @check "%2 = try -> Tuple{Int64}"
        @check "body:"
        @check "tryop_g(_2)"
        @check "yield %3"
        @check "catch(%arg1::Int64 ← ϒ[%1]):"
        @check "pop_exception %2"
        @check "yield %6"
        @check "getfield(%2, 1)"
        tryop_sci()
    end
    @test @filecheck begin
        @check "try -> Tuple{Int64}"
        @check "if %3"
        @check "leave %9"
        @check "catch:"
        tryop_leave_sci()
    end
    @test sprint(show, TryOp(Block(), nothing)) == "TryOp(scope-only)"
    # A scope-only try shows its scope, here an empty one, and no catch region.
    entry = Block()
    body = Block(); body.terminator = YieldOp()
    push!(entry, 1, TryOp(body, nothing, Vector{Int}[], nothing), Nothing)
    entry.terminator = ReturnNode(nothing)
    @test @filecheck begin
        @check "try scope nothing"
        @check_not "catch"
        StructuredIRCode(Any[Any], Any[], entry, 2)
    end
end

@testset "uses" begin
    sci = tryop_sci()
    op = tryop_of(sci)
    # The upsilon reads `x`; the handler's argument is not one of its uses (the
    # `upsilons` lists name statements, not values).
    @test length(uses(sci.entry, Core.Argument(2))) == 2      # the upsilon and the call
    @test length(uses(op.handler, op.handler.args[1])) == 1
    @test isempty(uses(sci.entry, SSAValue(1)))
    replace_uses!(sci.entry, Core.Argument(2), Core.Argument(3))
    @test (sci.entry.body[1].stmt::Core.UpsilonNode).val == Core.Argument(3)
    @test op.body.body[3].stmt.args[2] == Core.Argument(3)
    # A LeaveOp's values are uses, and a scope is a use of the TryOp.
    sci = tryop_leave_sci()
    @test length(uses(sci.entry, SSAValue(9))) == 1
    scoped = TryOp(Block(), nothing, Vector{Int}[], SSAValue(7))
    blk = Block()
    push!(blk, 8, scoped, Nothing)
    @test length(uses(blk, SSAValue(7))) == 1
    replace_uses!(blk, SSAValue(7), SSAValue(70))
    @test scoped.scope == SSAValue(70)
    # Replacing the scope with `nothing` keeps it a scope (an empty one).
    replace_uses!(blk, SSAValue(70), nothing)
    @test IRStructurizer.has_scope(scoped) && scoped.scope === nothing
    # A try without a scope has no scope use.
    blk = Block()
    push!(blk, 8, TryOp(Block(), Block()), Nothing)
    @test isempty(uses(blk, nothing))
end

@testset "validation" begin
    invalid(sci) = try validate_terminators(sci); "" catch e; e.msg end
    @test validate_terminators(tryop_sci()) && validate_ssa_defs(tryop_sci())
    @test validate_terminators(tryop_leave_sci()) && validate_ssa_defs(tryop_leave_sci())
    # The body yield must carry the op's one result, of its type.
    sci = tryop_sci(); op = tryop_of(sci)
    op.body.terminator = YieldOp()
    @test occursin("body yield carries 0 values", invalid(sci))
    op.body.terminator = YieldOp(Any["str"])
    @test occursin("not <: declared Int64", invalid(sci))
    # A value return inside the body would skip leaving the try.
    op.body.terminator = ReturnNode(42)
    @test occursin("value return inside the body", invalid(sci))
    # A LeaveOp out of a nested arm carries the op's results too.
    sci = tryop_leave_sci(); op = tryop_of(sci)
    ifop = only(e.stmt for (_, e) in op.body.body if e.stmt isa IfOp)
    ifop.then_region.terminator = LeaveOp()
    @test occursin("leave carries 0 values", invalid(sci))
    # Upsilons name UpsilonNode statements.
    sci = tryop_sci(); op = tryop_of(sci)
    push!(op.upsilons[1], 3)
    @test occursin("upsilon %3 of handler argument 1 is not an UpsilonNode", invalid(sci))
    # ... that runs before the handler can: in the body or before the op, not after
    # it or in the handler.
    sci = tryop_sci(); op = tryop_of(sci)
    push!(sci.entry, 11, Core.UpsilonNode(Core.Argument(2)), Nothing)
    push!(op.handler, 12, Core.UpsilonNode(Core.Argument(2)), Nothing)
    op.upsilons[1] = [1, 11, 12]
    @test occursin("upsilon %11 of handler argument 1 is neither in the body nor before the op", invalid(sci))
    @test occursin("upsilon %12 of handler argument 1 is neither", invalid(sci))
    @test !occursin("upsilon %1 ", invalid(sci))
    # An upsilon in an op before the try runs before it too.
    sci = tryop_sci(); op = tryop_of(sci)
    arm = Block(); push!(arm, 13, Core.UpsilonNode(Core.Argument(2)), Nothing)
    arm.terminator = YieldOp()
    other = Block(); other.terminator = YieldOp()
    entry = Block()
    push!(entry, 14, IfOp(true, arm, other), Nothing)
    for (i, e) in sci.entry.body
        push!(entry, i, e.stmt, e.type)
    end
    entry.terminator = sci.entry.terminator
    op.upsilons[1] = [1, 13]
    @test invalid(StructuredIRCode(Any[Any, Int], Any[], entry, 20)) == ""
    # An upsilon feeds a single handler argument.
    sci = tryop_sci(); op = tryop_of(sci)
    push!(op.handler.args, BlockArgument(2, Int))
    push!(op.upsilons, [1])
    @test occursin("upsilon %1 of handler argument 2 also feeds another", invalid(sci))
    # One upsilon list per handler argument, none without a handler; no body arguments.
    sci = tryop_sci(); op = tryop_of(sci)
    push!(op.upsilons, Int[])
    @test occursin("2 upsilon lists for 1 handler arguments", invalid(sci))
    @test occursin("1 upsilon lists for 0 handler arguments",
                   invalid(StructuredIRCode(Any[Any], Any[],
                       (b = Block(); push!(b, 1, TryOp(Block(), nothing, [Int[]]), Nothing); b), 1)))
    sci = tryop_sci(); op = tryop_of(sci)
    push!(op.body.args, BlockArgument(2, Int))
    @test occursin("body has 1 arguments", invalid(sci))
    # %t is read with getfield and named by pop_exception, and used no other way.
    sci = tryop_sci()
    sci.entry.terminator = ReturnNode(SSAValue(2))
    @test occursin("%2 used other than by getfield or pop_exception", invalid(sci))
    sci = tryop_sci(); op = tryop_of(sci)
    push!(op.handler, 7, Expr(:call, Core.getfield, SSAValue(2), 1), Int)
    @test occursin("TryOp at %2: its result is read inside it", invalid(sci))
    # A LeaveOp exits a try body only: not the handler of the outermost try, not
    # a loop inside a body, and not code without a TryOp.
    sci = tryop_sci(); op = tryop_of(sci)
    op.handler.terminator = LeaveOp(IRValue[0])
    @test occursin("LeaveOp outside a TryOp body", invalid(sci))
    sci = tryop_leave_sci(); op = tryop_of(sci)
    loop_body = Block(); loop_body.terminator = LeaveOp()
    push!(op.body, 11, LoopOp(loop_body, IRValue[]), Tuple{})
    @test occursin("LeaveOp outside a TryOp body", invalid(sci))
    blk = Block(); blk.terminator = LeaveOp()
    @test occursin("LeaveOp outside a TryOp body",
                   invalid(StructuredIRCode(Any[Any], Any[], blk, 1)))
    # Nor may a BreakOp or ContinueOp exit a body, which would skip the leave, or a
    # handler; one that exits a loop inside the body is fine.
    sci = tryop_leave_sci(); op = tryop_of(sci)
    ifop = only(e.stmt for (_, e) in op.body.body if e.stmt isa IfOp)
    ifop.then_region.terminator = BreakOp()
    @test occursin("BreakOp out of a TryOp", invalid(sci))
    ifop.then_region.terminator = ContinueOp()
    @test occursin("ContinueOp out of a TryOp", invalid(sci))
    sci = tryop_leave_sci(); op = tryop_of(sci)
    then_blk = Block(); then_blk.terminator = BreakOp()
    else_blk = Block(); else_blk.terminator = YieldOp()
    push!(op.handler, 12, IfOp(true, then_blk, else_blk), Nothing)
    @test occursin("BreakOp out of a TryOp", invalid(sci))
    sci = tryop_leave_sci(); op = tryop_of(sci)
    loop_body = Block(); loop_body.terminator = BreakOp()
    push!(op.body, 11, LoopOp(loop_body, IRValue[]), Tuple{})
    @test invalid(sci) == ""
    # `pop_exception %t` is scoped like any use of %t: after the TryOp in its own
    # region it is in scope, after the IfOp whose arm holds the TryOp it is not.
    sci = tryop_sci()
    push!(sci.entry, 7, Expr(:pop_exception, SSAValue(2)), Nothing)
    @test validate_ssa_defs(sci)
    arm = Block()
    push!(arm, 2, tryop_of(tryop_sci()), Tuple{Int})
    arm.terminator = YieldOp()
    entry = Block()
    push!(entry, 1, Core.UpsilonNode(Core.Argument(2)), Nothing)
    push!(entry, 8, IfOp(true, arm, (b = Block(); b.terminator = YieldOp(); b)), Nothing)
    push!(entry, 7, Expr(:pop_exception, SSAValue(2)), Nothing)
    entry.terminator = ReturnNode(nothing)
    @test_throws "SSA values used but not defined: %2" validate_ssa_defs(
        StructuredIRCode(Any[Any, Int], Any[], entry, 10))
end

end  # TryOp and LeaveOp
