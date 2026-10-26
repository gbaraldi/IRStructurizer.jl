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

end  # TryOp and LeaveOp
