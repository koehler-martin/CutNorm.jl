"""
    AlternatingLinearSearch(model::AbstractBilinearModel)

Local subsolver for the bilinear relaxation, alternating between the two blocks of
variables. This is the **default** subsolver of [`cutnorm`](@ref) and the fastest of
the three.

The objective is linear in `s` for fixed `t` and vice versa, so each block can be
minimized exactly by moving every coordinate to the bound indicated by the sign of
its partial gradient: `s[i] = 1` where `(A*t)[i] < 0` and `s[i] = 0` where it is
positive, then the same for `t` with `A'*s`. Iterating this converges to a vertex
that is optimal for both blocks. The construction allocates `x = [s; t]` and the two
partial gradients once and reuses them.

The solver implements the `SolverCore` interface, i.e. it is used as

```julia
solve!(solver, model, stats; x = x0, max_time = 30.0, verbose = 0)
```

with `stats::GenericExecutionStats`, and reset with `SolverCore.reset!`. Normally the
multistart solvers do this for you — you only pass the type:

```julia
sol = cutnorm(A; method = MultistartSigned{AlternatingLinearSearch}())
```

See also [`GreedySolver`](@ref), `TronSolver`, [`MultistartSignedSolver`](@ref).
"""
mutable struct AlternatingLinearSearch{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    S<:SubArray{T}
} <: AbstractOptimizationSolver
    x::V    # x = [s; t]
    s::S    # s = view of x
    t::S    # t = view of x
    gs::V   # gradient with respect to s
    gt::V   # gradient with respect to t
end

function AlternatingLinearSearch(model::AbstractBilinearModel{T,V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
    m, n = model.data.dims
    x = Vector{T}(undef, m + n)
    s = view(x, 1:m)
    t = view(x, m+1:m+n)
    gs = Vector{T}(undef, m)
    gt = Vector{T}(undef, n)

    return AlternatingLinearSearch{T,V,typeof(s)}(
        x,
        s,
        t,
        gs,
        gt,
    )
end

function SolverCore.reset!(solver::AlternatingLinearSearch)
    fill!(solver.x, 0)
    fill!(solver.gs, 0)
    fill!(solver.gt, 0)
end

"""
    solve!(solver::AlternatingLinearSearch, model, stats; x, max_iter, max_time, verbose)

Run the alternating block minimization on `model` from the starting point `x`
(clamped into the bounds) and write the result into `stats`.

# Keywords

- `x`: initial guess, defaults to `model.meta.x0`,
- `max_iter`: iteration limit, unlimited by default,
- `max_time`: time limit in seconds, `30.0` by default,
- `verbose`: `0` is silent, `k > 0` logs every `k`-th iteration.

Errors if `model` is not a bound-constrained or unconstrained minimization problem.
`stats.status` is `:first_order` when the vertex reached is optimal for both blocks,
otherwise the limit that was hit.
"""
function SolverCore.solve!(solver::AlternatingLinearSearch{T,V},
    model::AbstractBilinearModel{T,V},
    stats::GenericExecutionStats{T,V};
    x::V=model.meta.x0,
    max_iter::Int=typemax(Int),
    max_time::Float64=30.0,
    verbose::Int=0,
) where {T<:AbstractFloat,V<:AbstractVector{T}}
    if !(model.meta.minimize)
        error("Alternating Linear Search only works for minimization problem")
    end
    if !(unconstrained(model) || bound_constrained(model))
        error("Alternating Linear Search should only be called for unconstrained or bound-constrained problems")
    end

    SolverCore.reset!(stats)
    t0 = time_ns()
    set_time!(stats, 0.0)

    ℓ = model.meta.lvar
    u = model.meta.uvar
    m, n = model.data.dims

    if (verbose > 0) && !(u >= x >= ℓ)
        @warn "Warning: Initial guess is not within bounds."
    end

    solver.x .= x
    x = solver.x
    s = solver.s
    t = solver.t
    gs = solver.gs
    gt = solver.gt

    #x .= max.(ℓ, min.(x, u))
    @. x = clamp(x, ℓ, u)

    set_iter!(stats, 0)
    set_objective!(stats, obj(model, x))
    set_solution!(stats, x)

    verbose > 0 && @info log_header(
        [:iter, :f],
        [Int, T],
        hdr_override=Dict(:f => "obj"),
    )
    verbose > 0 && @info log_row(Any[stats.iter, obj(model, x)])

    set_status!(
        stats,
        get_status(
            model,
            elapsed_time=stats.elapsed_time,
            optimal=false,
            iter=stats.iter,
            max_iter=max_iter,
            max_time=max_time,
        ),
    )

    done = stats.status != :unknown

    while !done
        # s, t updates affect x, due to s, t being view of x
        grad_s!(model, t, gs)
        for i in eachindex(s)
            (gs[i] < 0) && (s[i] = u[i])
            (gs[i] > 0) && (s[i] = ℓ[i])
        end

        grad_t!(model, s, gt)
        for j in eachindex(t)
            (gt[j] < 0) && (t[j] = u[m+j])
            (gt[j] > 0) && (t[j] = ℓ[m+j])
        end

        # gs is stale after t changed above; gt is still valid
        grad_s!(model, t, gs)

        optimal = is_optimal(gs, s, gt, t, ℓ, u)

        set_iter!(stats, stats.iter + 1)

        elapsed = (time_ns() - t0) / 1e9
        set_time!(stats, elapsed)


        set_status!(
            stats,
            get_status(
                model,
                elapsed_time=stats.elapsed_time,
                optimal=optimal,
                iter=stats.iter,
                max_iter=max_iter,
                max_time=max_time,
            ),
        )

        (verbose > 0) &&
            (mod(stats.iter, verbose) == 0) &&
            @info log_row(Any[stats.iter, obj(model, x)])


        (stats.status != :unknown) && (done = true)
    end
    set_solution!(stats, x)
    set_objective!(stats, obj(model, x))
    return stats
end

"""
    is_optimal(gs, s, gt, t, ℓ, u) -> Bool

Check the first-order condition of the bound-constrained bilinear problem at the
point `[s; t]` with partial gradients `gs` and `gt` and bounds `ℓ`, `u`: every
coordinate with a negative partial derivative must sit at its upper bound, every
coordinate with a positive one at its lower bound. The bounds are indexed as in the
stacked variable, i.e. `t[j]` is compared against `ℓ[m+j]` and `u[m+j]`.
"""
function is_optimal(gs, s, gt, t, ℓ, u)
    m = length(s)
    for i in eachindex(s)
        at_upper = s[i] == u[i]
        at_lower = s[i] == ℓ[i]
        (gs[i] < 0 && !at_upper) && return false
        (gs[i] > 0 && !at_lower) && return false
    end
    for j in eachindex(t)
        at_upper = t[j] == u[m+j]
        at_lower = t[j] == ℓ[m+j]
        (gt[j] < 0 && !at_upper) && return false
        (gt[j] > 0 && !at_lower) && return false
    end
    return true
end