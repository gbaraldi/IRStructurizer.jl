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

end  # TryOp and LeaveOp
