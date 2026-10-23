#=============================================================================
 Exception handling on the MCFG: handler stacks, try regions, and the
 normalization that gives every try a single continuation.

 Julia's IR expresses try/catch with an `EnterNode` terminator (push a handler,
 fall through into the protected region, with an implicit edge to the catch
 block), `Expr(:leave, enters...)` (pop handlers), `Expr(:pop_exception, enter)`
 (restore the exception stack), and `UpsilonNode`/`PhiCNode` pairs that carry
 values into the catch block. `ingest` turns the `EnterNode` into an `MEnter`
 terminator and gives every `:leave` a block of its own that pops one handler
 (`split_leaves!`). A try is then three sets of blocks:

     B ── MEnter ──body──▶ R   blocks entered with the handler active
                 └─catch─▶ C   blocks the catch block dominates that run on B's
                               own handler stack, minus the `:leave` blocks that
                               pop one of B's outer handlers

 Every edge out of R starts at a `:leave` block, and every edge out of R ∪ C lands
 on a block with B's handler stack. `normalize_one_try!` routes those edges to a
 single continuation K (an `EdgeMux` when there are several), and threads values
 defined in R ∪ C and read past K through K's block arguments (the reduce form,
 as for loops). The lift (`emit_try!`) then builds a `TryOp` whose body is R,
 whose handler is C, and whose results are K's arguments.

 Because K holds B's handler stack, every mux the other passes build joins edges
 that hold the same handlers. An edge out of a try body from a nested region (an
 `IfOp` arm whose continuation stays in the body) is not a continuation edge of
 that branch: `branch_continuation` skips it by handler depth, and the lift turns
 it into a `LeaveOp`. A loop that contains B also contains R ∪ C
 (`extend_loops_over_trys!`), so a throwing dead end in R stays in the try instead
 of becoming a loop exit that the latch would hoist past the handler.
=============================================================================#

is_leave(@nospecialize(s)) = s isa Expr && s.head === :leave

has_enter(m::MCFG) = any(b -> b.term isa MEnter, m.blocks)

"""
    handler_stacks(m::MCFG) -> Vector{Union{Nothing, Vector{Int}}}

The active handlers at each block's entry, as the ids of their `MEnter`s,
outermost first (`nothing` for an unreachable block). This is the forward dataflow
of Julia's `compute_trycatch`. An `MEnter` pushes its id onto the stack of its
body edge, and its catch edge keeps the stack it was entered with. A `:leave`
pops one handler per argument. Throws `UnstructuredControlFlowError` when two
paths reach a block with different stacks (an unbalanced try/catch).
"""
function handler_stacks(m::MCFG)
    nb = length(m.blocks)
    stacks = Vector{Union{Nothing, Vector{Int}}}(nothing, nb)
    stacks[m.entry] = Int[]
    work = Int[m.entry]
    while !isempty(work)
        b = pop!(work)
        s = copy(stacks[b]::Vector{Int})
        blk = m.blocks[b]
        for st in blk.body
            is_leave(st.stmt) || continue
            n = length(st.stmt.args)
            n <= length(s) || throw(UnstructuredControlFlowError(
                "unbalanced try/catch: the :leave at %$(st.id) pops more handlers than are active"))
            resize!(s, length(s) - n)
        end
        t = blk.term
        if t isa MEnter
            _flow_stack!(stacks, work, t.body.target, push!(copy(s), t.id))
            t.catch_dest != 0 && _flow_stack!(stacks, work, t.catch_dest, s)
        else
            for tgt in _term_targets(t)
                _flow_stack!(stacks, work, tgt, s)
            end
        end
    end
    return stacks
end

function _flow_stack!(stacks, work::Vector{Int}, tgt::Int, s::Vector{Int})
    cur = stacks[tgt]
    if cur === nothing
        stacks[tgt] = copy(s)
        push!(work, tgt)
    elseif cur != s
        throw(UnstructuredControlFlowError(
            "unbalanced try/catch: block $tgt is entered with different active handlers"))
    end
end

"""Handler-stack depth at each block's entry (-1 for an unreachable block). An edge
into a shallower block leaves a try body."""
handler_depths(m::MCFG) = handler_depths(handler_stacks(m))
handler_depths(stacks::Vector) = Int[s === nothing ? -1 : length(s) for s in stacks]

"""
    TryRegion

One try, by its `MEnter`. `enter` is the `MEnter` id (the `TryOp`'s SSA index) and
`block` the block it terminates. `body` (R) holds the blocks entered with this
handler active. `handler` (C) holds the blocks the catch block dominates that run
on `block`'s handler stack, minus the `:leave` blocks that pop one of the outer
handlers. It is empty for a scope-only enter. `depth` is the handler-stack depth
at `block`'s entry.
"""
struct TryRegion
    enter::Int
    block::Int
    body::Set{Int}
    handler::Set{Int}
    depth::Int
end

construct_blocks(r::TryRegion) = union(r.body, r.handler)

# A block holding only a `:leave` that pops one of the handlers in `outer`. It
# belongs to the enclosing try body, not to the handler region that reaches it.
function is_outer_leave(b::MBlock, outer::Vector{Int})
    length(b.body) == 1 || return false
    st = b.body[1].stmt
    is_leave(st) || return false
    return any(a -> a isa SSAValue && a.id in outer, st.args)
end

"""
    try_info(m, stacks=handler_stacks(m), domtree=...) -> Dict{Int, TryRegion}

The `TryRegion` of every reachable `MEnter`, keyed by the `MEnter` id.
"""
function try_info(m::MCFG, stacks=handler_stacks(m),
                  domtree::DomTree=construct_domtree(build_cfg(m)))
    regions = Dict{Int, TryRegion}()
    for (bid, b) in enumerate(m.blocks)
        t = b.term
        t isa MEnter || continue
        σ = stacks[bid]
        σ === nothing && continue
        body = Set{Int}()
        handler = Set{Int}()
        for (x, s) in enumerate(stacks)
            s === nothing && continue
            t.id in s && push!(body, x)
        end
        if t.catch_dest != 0
            c = t.catch_dest
            d = length(σ)
            for (x, s) in enumerate(stacks)
                s === nothing && continue
                (length(s) >= d && view(s, 1:d) == σ) || continue
                dominates(domtree, c, x) || continue
                is_outer_leave(m.blocks[x], σ) && continue
                push!(handler, x)
            end
        end
        regions[t.id] = TryRegion(t.id, bid, body, handler, length(σ))
    end
    return regions
end

"""The edges that leave a try's body and handler (R ∪ C). Each lands on a block that
holds the try block's handler stack; together they are the `TryOp`'s continuation."""
function try_exit_refs(m::MCFG, r::TryRegion)
    construct = construct_blocks(r)
    refs = EdgeRef[]
    for src in sort!(collect(construct))
        for ref in _edge_refs_of(src, m.blocks[src].term)
            edge_of(m, ref).target in construct || push!(refs, ref)
        end
    end
    return refs
end

#=============================================================================
 Ingest-time cleanup
=============================================================================#

"""Clear the blocks that no edge reaches from the entry (they keep their ids). A
dead block's edges would otherwise still feed the block it used to jump to: under
`exceptions=:trap`, a catch block whose join is live."""
function prune_unreachable!(m::MCFG)
    reach = falses(length(m.blocks))
    wl = Int[m.entry]
    while !isempty(wl)
        b = pop!(wl)
        reach[b] && continue
        reach[b] = true
        append!(wl, _term_targets(m.blocks[b].term))
    end
    pruned = false
    for b in eachindex(m.blocks)
        reach[b] && continue
        m.blocks[b] = MBlock()
        pruned = true
    end
    pruned && fold_single_edge_args!(m)
end

"""Replace the block arguments of every block reached by exactly one edge with that
edge's operands. Optimized Julia IR has no such phis, but pruning a dead handler
leaves its join with a single predecessor (`exceptions=:trap`)."""
function fold_single_edge_args!(m::MCFG)
    incoming = [MEdge[] for _ in m.blocks]
    for b in m.blocks
        t = b.term
        edges = t isa MGoto ? (t.edge,) : t isa MCondBr ? (t.t, t.f) :
                t isa MEnter ? (t.body,) : ()
        for e in edges
            push!(incoming[e.target], e)
        end
    end
    subst = Dict{Int, Any}()
    for (bid, b) in enumerate(m.blocks)
        (isempty(b.args) || length(incoming[bid]) != 1) && continue
        e = only(incoming[bid])
        for (k, a) in enumerate(b.args)
            subst[a] = e.args[k]
        end
        empty!(b.args)
        empty!(e.args)
    end
    isempty(subst) && return
    # Chase substitutions whose operand is itself a folded argument.
    resolve(@nospecialize(v)) = (v isa SSAValue && haskey(subst, v.id)) ? resolve(subst[v.id]) : v
    for b in m.blocks
        _rewrite_mblock!(b, resolve)
    end
end

"""Give every `:leave` a block of its own that pops one handler. Julia ends a block
at each `:leave`, which may pop several handlers at once (a `return` out of nested
trys). The leave moves to a chain of new blocks that pop the handlers innermost
first, each naming the `MEnter` it pops. A block ending in a leave then no longer
crosses a try boundary partway: everything before the leave is in the try body,
and the leave block's one outgoing edge is the exit. Runs after edge operands are
filled, since the moved terminator keeps its operands. A catch block that holds
nothing but a leave (a handler that exits an outer try at once) is split too, so
the handler region keeps its entry block."""
function split_leaves!(m::MCFG)
    stacks = handler_stacks(m)
    catch_dests = Set{Int}(b.term.catch_dest for b in m.blocks if b.term isa MEnter)
    for bid in 1:length(m.blocks)              # range fixed: new blocks are already split
        b = m.blocks[bid]
        k = findfirst(s -> is_leave(s.stmt), b.body)
        k === nothing && continue
        k == length(b.body) || throw(UnstructuredControlFlowError(
            "the :leave at %$(b.body[k].id) does not end its block"))
        s = stacks[bid]::Vector{Int}
        leave = b.body[k]
        n = length(leave.stmt.args)
        if length(b.body) == 1 && n == 1 && !(bid in catch_dests)
            # Already a single-pop leave block. Name the handler it actually pops.
            b.body[1] = MStmt(leave.id, Expr(:leave, SSAValue(s[end])), leave.type,
                              leave.codeloc, leave.flag)
            continue
        end
        pop!(b.body)
        term, term_codeloc, term_id = b.term, b.term_codeloc, b.term_id
        prev = bid
        for j in 1:n
            id = j == 1 ? leave.id : alloc_id!(m)
            if j > 1
                m.types[id] = leave.type
                m.codelocs[id] = leave.codeloc
            end
            stmt = MStmt(id, Expr(:leave, SSAValue(s[end - j + 1])), leave.type,
                         leave.codeloc, leave.flag)
            nb = add_block!(m, MBlock(Int[], MStmt[stmt], MReturn(), leave.codeloc))
            m.blocks[prev].term = MGoto(MEdge(nb))
            m.blocks[prev].term_codeloc = leave.codeloc
            m.blocks[prev].term_id = 0
            prev = nb
        end
        m.blocks[prev].term = term
        m.blocks[prev].term_codeloc = term_codeloc
        m.blocks[prev].term_id = term_id
    end
end

is_pop_exception(@nospecialize(s)) =
    s isa Expr && s.head === :pop_exception && length(s.args) == 1 && s.args[1] isa SSAValue

"""
    exception_states(m, enter_block) -> (state at each block's entry, state at each enter)

The exception-stack state, as in the second component of Julia's
`compute_trycatch`: the id of the `MEnter` whose catch region the code is in (0
outside every catch). The catch edge of an `MEnter` sets it to that enter's id, and
`pop_exception %X` restores the state that `X` was entered with.
"""
function exception_states(m::MCFG, enter_block::Dict{Int, Int})
    nb = length(m.blocks)
    state = fill(-1, nb)          # -1: not reached yet
    at_enter = Dict{Int, Int}()
    state[m.entry] = 0
    work = Int[m.entry]
    while !isempty(work)
        b = pop!(work)
        s = state[b]
        blk = m.blocks[b]
        for st in blk.body
            is_pop_exception(st.stmt) || continue
            x = st.stmt.args[1].id
            haskey(enter_block, x) || continue
            s = get(at_enter, x, s)
        end
        t = blk.term
        t isa MEnter && (at_enter[t.id] = s)
        for tgt in _term_targets(t)
            s′ = (t isa MEnter && tgt == t.catch_dest && tgt != t.body.target) ? t.id : s
            if state[tgt] == -1
                state[tgt] = s′
                push!(work, tgt)
            end
        end
    end
    return state, at_enter
end

"""Retarget a `pop_exception %E` that runs after an enclosing try body was left (a
`return` out of a catch nested in another try: Julia leaves the outer try, then pops
the exception) to the outermost try around E whose enter saw the same exception
stack. Both name the same exception-stack state, but after the lift such a pop
sits after that outer `TryOp`, where only the outer enter still dominates it."""
function retarget_pop_exceptions!(m::MCFG)
    stacks = handler_stacks(m)
    regions = try_info(m, stacks)
    enter_block = Dict{Int, Int}(id => r.block for (id, r) in regions)
    _, at_enter = exception_states(m, enter_block)
    for (bid, b) in enumerate(m.blocks)
        for (k, st) in enumerate(b.body)
            is_pop_exception(st.stmt) || continue
            e = st.stmt.args[1].id
            r = get(regions, e, nothing)
            (r === nothing || bid in r.handler) && continue
            target = get(at_enter, e, nothing)
            target === nothing && continue
            σ = stacks[r.block]::Vector{Int}
            outer = e
            for j in length(σ):-1:1
                get(at_enter, σ[j], nothing) == target || break
                outer = σ[j]
            end
            outer == e && continue
            b.body[k] = MStmt(st.id, Expr(:pop_exception, SSAValue(outer)), st.type,
                              st.codeloc, st.flag)
        end
    end
end

"""Turn the `Union{}`-typed phis and `PhiCNode`s that try/catch leaves behind into
explicit `Undef`. A variable that is never assigned on the path into a catch block
gets a `Union{}` PhiC, and phis carry that around merges and loops. A `Union{}`
value never exists at runtime: Julia only reads such a variable behind
`throw_undef_if_not`. Each edge operand that carries one becomes `Undef`, and each
upsilon of one becomes an undefined upsilon. The `Union{}` phis are then left
without any value, so the lift drops them instead of trying to give a loop or
`IfOp` a `Union{}` carry."""
function undef_bottom_values!(m::MCFG)
    bottom = Set{Int}()
    for b in m.blocks
        for a in b.args
            get(m.types, a, Any) === Union{} && push!(bottom, a)
        end
        for s in b.body
            s.stmt isa Core.PhiCNode && s.type === Union{} && push!(bottom, s.id)
        end
    end
    isempty(bottom) && return
    isbottom(@nospecialize(v)) = v isa SSAValue && v.id in bottom
    for b in m.blocks
        t = b.term
        edges = t isa MGoto ? (t.edge,) : t isa MCondBr ? (t.t, t.f) :
                t isa MEnter ? (t.body,) : ()
        for e in edges
            targs = m.blocks[e.target].args
            for k in eachindex(e.args)
                isbottom(e.args[k]) && (e.args[k] = Undef(get(m.types, targs[k], Any)))
            end
        end
        for (i, s) in enumerate(b.body)
            if s.stmt isa Core.UpsilonNode && isdefined(s.stmt, :val) && isbottom(s.stmt.val)
                b.body[i] = MStmt(s.id, Core.UpsilonNode(), s.type, s.codeloc, s.flag)
            end
        end
    end
end

#=============================================================================
 Loops containing a try
=============================================================================#

"""Grow each natural loop that contains a try's `MEnter` block by that try's body
and handler. A dead end in the body (a throw) cannot reach the back edge, so the
plain natural loop leaves it out and the latch would route it out of the loop, past
the handler. A handler that leaves the loop would make the exceptional edge itself
a loop exit. With the whole try inside, the loop's exits leave from the `TryOp`'s
continuation, where no handler of the try is active."""
function extend_loops_over_trys!(loops::Dict{Int, Set{Int}}, m::MCFG)
    isempty(loops) && return loops
    regions = try_info(m)
    for body in values(loops)
        grew = true
        while grew
            grew = false
            for r in values(regions)
                r.block in body || continue
                n = length(body)
                union!(body, r.body, r.handler)
                length(body) > n && (grew = true)
            end
        end
    end
    return loops
end

#=============================================================================
 Try normalization
=============================================================================#

"""
    normalize_one_try!(m::MCFG) -> Bool

Normalize one try, returning `true` if it changed the CFG:

1. If the edges out of its body and handler reach more than one block, or reach a
   loop header (a back edge, or the entry of a later loop), route them all through
   one mux. The `TryOp` then has a single continuation K that is an ordinary block,
   and the dispatch to the original targets runs after it, outside the try.
2. Otherwise, if a value defined in the body or handler is read past K, thread it
   through a new argument of K (`reduce_try!`).
"""
function normalize_one_try!(m::MCFG)
    has_enter(m) || return false
    stacks = handler_stacks(m)
    cfg = build_cfg(m)
    domtree = construct_domtree(cfg)
    regions = try_info(m, stacks, domtree)
    headers = nothing
    for id in sort!(collect(keys(regions)))
        r = regions[id]
        refs = try_exit_refs(m, r)
        isempty(refs) && continue
        targets = unique!(Int[edge_of(m, x).target for x in refs])
        if length(targets) == 1
            headers === nothing && (headers = keys(natural_loops_m(m)))
            if !(only(targets) in headers)
                reduce_try!(m, r, only(targets), cfg, domtree) && return true
                continue
            end
        end
        single_entry_mux!(m, refs)
        return true
    end
    return false
end

"""
    reduce_try!(m, r, K, cfg, domtree) -> Bool

The reduce form for a try: give the continuation K one new block argument per value
defined in the body or handler that a block dominated by K reads. Each incoming
edge feeds the value where it dominates the edge's source, and `Undef` elsewhere,
and the reads past K are rewritten to the new argument. Then nothing defined inside
the `TryOp` is read outside it except through its results. Only applies when every
predecessor of K is inside the try; otherwise no value from inside dominates K.
Returns `true` if it added an argument.
"""
function reduce_try!(m::MCFG, r::TryRegion, K::Int, cfg::CFG, domtree::DomTree)
    construct = construct_blocks(r)
    all(p -> p in construct, cfg.blocks[K].preds) || return false

    def_block = Dict{Int, Int}()
    def_type = Dict{Int, Any}()
    for bid in construct
        b = m.blocks[bid]
        for a in b.args; def_block[a] = bid; def_type[a] = get(m.types, a, Any); end
        for s in b.body; def_block[s.id] = bid; def_type[s.id] = s.type; end
    end

    past = Int[bid for bid in 1:length(m.blocks)
               if !(bid in construct) && dominates(domtree, K, bid)]
    escaping = Int[]
    for bid in past, id in _mblock_use_ids(m.blocks[bid])
        haskey(def_block, id) && !(id in escaping) && push!(escaping, id)
    end
    isempty(escaping) && return false

    pred_refs = EdgeRef[]
    for src in sort!(unique(cfg.blocks[K].preds))
        append!(pred_refs, edge_refs(m, src, K))
    end
    for v in escaping
        T = widenconst(def_type[v])
        karg = alloc_id!(m)
        m.types[karg] = T
        m.codelocs[karg] = get(m.codelocs, v, _NOLOC)
        push!(m.blocks[K].args, karg)
        dv = def_block[v]
        for ref in pred_refs
            push!(edge_of(m, ref).args,
                  dominates(domtree, dv, ref.src) ? SSAValue(v) : Undef(T))
        end
        fv(@nospecialize(w)) = (w isa SSAValue && w.id == v) ? SSAValue(karg) : w
        for bid in past
            _rewrite_mblock!(m.blocks[bid], fv)
        end
    end
    return true
end

"""Check the invariants the lift relies on after `normalize_cf!`: balanced handler
stacks, a catch block reached only by its exceptional edge, and a single
continuation per try that is not a loop header. A shape that breaks one of them is
not supported, so this throws `UnstructuredControlFlowError` rather than lift wrong
code."""
function verify_try_structure(m::MCFG)
    stacks = handler_stacks(m)
    cfg = build_cfg(m)
    domtree = construct_domtree(cfg)
    regions = try_info(m, stacks, domtree)
    headers = keys(natural_loops_m(m))
    for id in sort!(collect(keys(regions)))
        r = regions[id]
        t = m.blocks[r.block].term::MEnter
        if t.catch_dest != 0 && cfg.blocks[t.catch_dest].preds != [r.block]
            throw(UnstructuredControlFlowError(
                "unsupported try/catch: the catch block of %$id is also reached by ordinary control flow"))
        end
        targets = unique!(Int[edge_of(m, x).target for x in try_exit_refs(m, r)])
        if length(targets) > 1 || (length(targets) == 1 && only(targets) in headers)
            throw(UnstructuredControlFlowError(
                "unsupported try/catch: the try at %$id has no single continuation"))
        end
    end
end
