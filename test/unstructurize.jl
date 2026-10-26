@testset "unstructurize" begin

@testset "linear code" begin
    @test @roundtrip (x -> x + 1)(5)
    @test @roundtrip ((x, y) -> (x + y) * (x - y))(3, 2)
end

@testset "if-then-else" begin
    @test @roundtrip (x -> x > 0 ? x + 1 : x - 1)(5)
    @test @roundtrip (x -> x > 0 ? x + 1 : x - 1)(-3)
    @test @roundtrip (x -> x > 0 ? x : -x)(5)
    @test @roundtrip (x -> x > 0 ? x : -x)(-5)
    # Result used after if (merge with PhiNodes)
    @test @roundtrip ((x::Int) -> (x > 0 ? x + 1 : x - 1) + 10)(5)
    @test @roundtrip ((x::Int) -> (x > 0 ? x + 1 : x - 1) + 10)(-3)
end

@testset "early return (termination)" begin
    f_early = function (x::Int, y::Int)
        if x > y
            return y * x
        end
        y - x
    end
    @test @roundtrip f_early(5, 3)
    @test @roundtrip f_early(3, 5)
end

@testset "for loop (counter)" begin
    f_count = function (n::Int)
        i = 0
        while i < n
            i += 1
        end
        return i
    end
    @test @roundtrip f_count(5)
    @test @roundtrip f_count(0)
end

@testset "for loop (accumulator)" begin
    f_acc = function (n::Int)
        i = 0
        s = 0
        while i < n
            s += i
            i += 1
        end
        return s
    end
    @test @roundtrip f_acc(5)
    @test @roundtrip f_acc(0)
end

@testset "nested if in loop" begin
    f_nested = function (n::Int)
        i = 0
        s = 0
        while i < n
            if i > 2
                s += i * 2
            else
                s += i
            end
            i += 1
        end
        return s
    end
    @test @roundtrip f_nested(5)
    @test @roundtrip f_nested(0)
end

@testset "round trip preserves valid_worlds and passes verify_ir" begin
    # `gcd` references globals from Base submodules; without the original
    # `valid_worlds` the reconstructed IRCode defaults to the full world range
    # and `verify_ir` rejects those as partitioned GlobalRefs on 1.12+.
    CC = Core.Compiler
    ir, _ = only(code_ircode(gcd, (Int, Int)))
    sci = StructuredIRCode(ir)
    ir2 = CC.IRCode(sci)
    CC.verify_ir(ir2)  # should not throw
    @static if VERSION >= v"1.12-"
        @test ir2.valid_worlds == ir.valid_worlds
    end
end

@testset "undef loop initial values" begin
    # `y` escapes the loop, so it is carried with an `Undef` initial value. That
    # must lower to an unassigned phi slot, not to the `Undef` object itself.
    f_escape = function (n::Int)
        i = 0
        while true
            y = i * 2
            i += 1
            i > n && return y + 1
        end
    end
    @test @roundtrip f_escape(3)
    sci, _ = only(code_structured(f_escape, Tuple{Int}))
    ir = Core.Compiler.IRCode(sci)
    phis = [s for s in ir.stmts.stmt if s isa PhiNode]
    @test any(p -> !all(k -> isassigned(p.values, k), eachindex(p.values)), phis)
    @test !any(p -> any(k -> isassigned(p.values, k) && p.values[k] isa Undef,
                        eachindex(p.values)), phis)
end

end  # unstructurize
