# normalize_cf!'s :fast driver against the original one (the passes one at a time, each computing its
# own analyses; both share the passes' rewrites, so the execution checks are what test those): the
# same normalized CFG, and code that runs the same.
using IRStructurizer, Test
const IRS = IRStructurizer
# The original driver, as the oracle.
function normalize_reference!(m)
    guard=0; limit=8*(length(m.blocks)+16)
    while IRS.normalize_one_try!(m) || IRS.normalize_one_irreducible!(m) ||
          IRS.normalize_one_loop_latch!(m) || IRS.normalize_one_continuation!(m) ||
          IRS.normalize_one_preheader!(m)
        guard+=1; guard>limit && error("reference did not converge")
    end
    IRS.has_enter(m) && IRS.verify_try_structure(m)
    m
end
function reuse_gates(n::Int, k::Int)
    s=0
    for i in 1:n
        if (i % 2 == 0 || i < k) && (i % 3 != 0 || k < 0)
            s += i
        elseif i==k
            s -= 2
        end
    end
    s
end
function reuse_exits(n::Int, k::Int)
    s=0; i=0
    while i<n
        i+=1
        i==k && continue
        i>2k && k>0 && break
        for j in 1:k
            j==i && continue
            j+i>20 && return s
            s+=i-j
        end
    end
    s
end
function reuse_try(n::Int,k::Int)
    s=0
    for i in 1:n
        try
            i==k && error("synthetic")
            s += 10 ÷ (i-k)
        catch
            s -= 1
        end
    end
    s
end
# a loop with several exits and several values defined in it and used after it (escaping values
# that the latch carries out)
function reuse_escapes(n::Int, k::Int)
    a = 0; b = 1; c = 0
    i = 0
    while i < n
        i += 1
        a = i * k
        b = b + a
        i == k && break
        c = a - b
        b > 1000 && break
        a < -50 && return a + c
    end
    a + 2b + 3c
end
function compile_reused(sci)
    ir=IRS.CC.copy(IRS.CC.IRCode(sci))
    ir.argtypes[1]=Tuple{}
    @static if VERSION >= v"1.12-"
        ir.debuginfo.def=:normalization_reuse
    end
    Core.OpaqueClosure(ir)
end
function normalize_in_mode!(m, mode)
    old = IRS.NORMALIZE_MODE[]
    IRS.NORMALIZE_MODE[] = mode
    try
        IRS.normalize_cf!(m)
    finally
        IRS.NORMALIZE_MODE[] = old
    end
    m
end
@testset "normalization modes give the original driver's CFG" begin
    for f in (reuse_gates,reuse_exits,reuse_try,reuse_escapes)
        ir=first(only(Base.code_ircode(f,Tuple{Int,Int})))
        # the original driver, as the oracle
        reference=IRS.ingest(ir)
        old = IRS.NORMALIZE_MODE[]
        IRS.NORMALIZE_MODE[] = :original
        try
            normalize_reference!(reference)
        finally
            IRS.NORMALIZE_MODE[] = old
        end
        for mode in (:original, :fast)
            optimized=normalize_in_mode!(IRS.ingest(ir), mode)
            @test repr(reference)==repr(optimized)
            sci=IRS.lift_mcfg(optimized;promote=false)
            oc=compile_reused(sci)
            for n in (0,1,7,20), k in (-1,0,2,6,10)
                @test oc(n,k)==f(n,k)
            end
        end
    end
end
