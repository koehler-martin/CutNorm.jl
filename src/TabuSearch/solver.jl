"""
    TabuSearchWorkspace(T::Type, m::Int, n::Int)

Work storage of one search of [`TabuSearchSolver`](@ref), for an `m x n` matrix with
element type `T`. The vectors are overwritten at the start of every search, so their
initial contents do not matter.

The indicator vectors have entries in `{0, 1}`, stored with element type `T` like
`S` and `T` of the solution.

| Field        | Length   | Content                                                  |
|:-------------|:---------|:---------------------------------------------------------|
| `s`, `t`     | `m`, `n` | Current row and column indicators                        |
| `u`, `v`     | `m`, `n` | The products `u = A*t` and `v = A'*s`                    |
| `tabu_until` | `m + n`  | Last iteration in which flipping the bit is tabu         |
| `es`, `et`   | `m`, `n` | Elite, the best indicators of the current episode        |
| `bs`, `bt`   | `m`, `n` | Best indicators of this search                           |
| `perm`       | `m + n`  | Scratch permutation for choosing the bits to perturb     |
| `rng`        |          | Random number generator of this search                   |
"""
struct TabuSearchWorkspace{T<:AbstractFloat}
    s::Vector{T}
    t::Vector{T}
    u::Vector{T}
    v::Vector{T}
    tabu_until::Vector{Int}
    es::Vector{T}
    et::Vector{T}
    bs::Vector{T}
    bt::Vector{T}
    perm::Vector{Int}
    rng::Xoshiro
end

function TabuSearchWorkspace(T::Type, m::Int, n::Int)
    return TabuSearchWorkspace{T}(
        Vector{T}(undef, m),
        Vector{T}(undef, n),
        Vector{T}(undef, m),
        Vector{T}(undef, n),
        Vector{Int}(undef, m + n),
        Vector{T}(undef, m),
        Vector{T}(undef, n),
        Vector{T}(undef, m),
        Vector{T}(undef, n),
        Vector{Int}(undef, m + n),
        Xoshiro(0)
    )
end

"""
    TabuSearchIncumbent(T::Type)

State shared by the parallel searches of one [`solve!`](@ref) of a
[`TabuSearchSolver`](@ref). `value` is the best value over all searches, updated by each
search at the end of every phase, and `improvements` counts how often it improved; both
are guarded by `lock`, which also keeps the log rows in order. `stop` tells all searches
to stop once one of them has reached the target.
"""
mutable struct TabuSearchIncumbent{T<:AbstractFloat}
    value::T
    improvements::Int
    lock::ReentrantLock
    stop::Threads.Atomic{Bool}
end

TabuSearchIncumbent(T::Type) = TabuSearchIncumbent{T}(typemin(T), 0, ReentrantLock(), Threads.Atomic{Bool}(false))

"""
    TabuSearchSolver(A::AbstractMatrix; kwargs...)

Build the tabu search solver for the cut norm of `A`, i.e. the solver behind
`cutnorm(A; method = TabuSearch())`. `kwargs` are forwarded to
[`TabuSearchSettings`](@ref).

The search maximizes `σ s'A t` over `s ∈ {0,1}^m`, `t ∈ {0,1}^n` and `σ = ±1` by
flipping one bit of `s` or `t` per iteration. It keeps `u = A*t` and `v = A'*s` up to
date, so the value after each of the `m + n` possible flips can be read off `u` and
`v`, and applying a flip of `s[i]` or `t[j]` adds row `i` of `A` to `v`, or column `j`
to `u`, or subtracts it. One iteration therefore costs `O(m + n)`.

The solver stores a copy of `A` together with its transpose `At`, so that the rows of
`A` are available as the columns of `At`. Both are built with `copy` and `permutedims`,
which keep the storage type of `A` where it has its own methods: a `SparseMatrixCSC`
stays sparse, while wrappers such as `Adjoint`, `Transpose` or views become a dense
`Matrix`. It also keeps one [`CutNorm.TabuSearchWorkspace`](@ref) per search, holding
all vectors that search needs. Everything is reused across calls to [`solve!`](@ref);
only the list of workspaces grows if `ntasks` is increased.

# Examples

```julia
solver = TabuSearchSolver(A; max_time = 5.0)
sol = solve!(solver)
sol = solve!(solver; S0 = sol.S, T0 = sol.T, max_time = 30.0)  # continue from sol
```

See also [`cutnorm`](@ref), [`TabuSearch`](@ref), [`TabuSearchSolution`](@ref),
[`TabuSearchSettings`](@ref).
"""
mutable struct TabuSearchSolver{T<:AbstractFloat,M<:AbstractMatrix{T},Mt<:AbstractMatrix{T}} <: AbstractSolver{T}
    A::M
    At::Mt
    workspaces::Vector{TabuSearchWorkspace{T}}
    settings::TabuSearchSettings
end

function TabuSearchSolver(A::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat}
    settings = TabuSearchSettings()
    populate!(settings; kwargs...)

    M = copy(A)
    m, n = size(M)
    workspaces = [TabuSearchWorkspace(T, m, n) for _ in 1:max(1, settings.ntasks)]

    return TabuSearchSolver(M, permutedims(M), workspaces, settings)
end

"""
    solve!(solver::TabuSearchSolver; kwargs...) -> TabuSearchSolution
    solve!(solver::TabuSearchSolver, sol::TabuSearchSolution; kwargs...)

Run `ntasks` independent tabu searches in parallel and return the best solution found.
`kwargs` update the solver's [`TabuSearchSettings`](@ref) before the run.

Each search starts from `S0` and `T0`, or from a random point, and repeats the following
iteration until a limit is hit:

1. Evaluate all `m + n` single-bit flips and apply the best admissible one, even if it
   makes the value worse. A flip is admissible if its bit is not tabu, or if it beats the
   elite, the best value of the current episode (aspiration). If every flip is tabu, the
   best one is applied anyway.
2. Make the flipped bit tabu for `tenure + rand(0:tenure_rand)` iterations.
3. After `stall` iterations without improving the elite, perturb it: continue from the
   elite with a random fraction `perturb_frac` of all bits flipped. After
   `max_perturbations` perturbations in a row that did not improve the elite, restart
   from a random point with the opposite sign instead, which starts a new episode.

`A*t` and `A'*s` are recomputed from scratch after every perturbation and restart, and
every `refresh` iterations, to remove the rounding drift of the incremental updates.

The indicators of the best search are returned in `sol.S` and `sol.T`, and `sol.value`
is recomputed from them. A `DimensionMismatch` is thrown if `S0` or `T0` does not match
the size of the matrix.

The second form writes into the solution object you pass in (it is reset first), which
lets you reuse the same storage across solves.
"""
function solve!(solver::TabuSearchSolver{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(solver.A)
    sol = TabuSearchSolution(T, m, n)
    solve!(solver, sol; kwargs...)
end

function solve!(solver::TabuSearchSolver{T}, sol::TabuSearchSolution{T}; kwargs...) where {T}
    t0 = time_ns()
    reset!(sol)
    settings = solver.settings
    populate!(settings; kwargs...)

    A = solver.A
    At = solver.At
    m, n = sol.dims
    N = m + n

    S0 = settings.S0
    T0 = settings.T0
    if S0 !== nothing && length(S0) != m
        throw(DimensionMismatch("S0 has length $(length(S0)), but the matrix has $m rows"))
    end
    if T0 !== nothing && length(T0) != n
        throw(DimensionMismatch("T0 has length $(length(T0)), but the matrix has $n columns"))
    end

    # With a complete initial guess, the odd searches start with its sign.
    σ_start = (S0 !== nothing && T0 !== nothing && sum(view(A, S0, T0)) < 0) ? -one(T) : one(T)

    # The searches work with unscaled values; `scale` only converts them for printing.
    scale = settings.scaled ? one(T) / (m * n) : one(T)
    params = (
        tenure=settings.tenure > 0 ? settings.tenure : max(1, round(Int, 0.005 * N)),
        tenure_rand=clamp(settings.tenure_rand, 0, N ÷ 4),
        stall=settings.stall > 0 ? settings.stall : max(100, 5 * N),
        nflips=clamp(round(Int, settings.perturb_frac * N), 1, N),
        max_perturbations=settings.max_perturbations,
        refresh=settings.refresh > 0 ? settings.refresh : max(1000, 10 * N),
        max_iter=settings.max_iter,
        max_time=settings.max_time,
        target=settings.scaled ? T(settings.target) * (m * n) : T(settings.target),
        tol=eps(T)^T(3 / 4) * max(one(T), sum(abs, A)),
        scale=scale,
        S0=S0,
        T0=T0,
        print_level=settings.print_level,
    )

    ntasks = max(1, settings.ntasks)
    while length(solver.workspaces) < ntasks
        push!(solver.workspaces, TabuSearchWorkspace(T, m, n))
    end

    io = stdout
    pl = settings.print_level
    (pl >= 1) && (print_header(io); print_tabu_header_info(io, m, n, ntasks, params, settings); print_tabu_iter_header(io))

    incumbent = TabuSearchIncumbent(T)
    tasks = map(1:ntasks) do k
        ws = solver.workspaces[k]
        seed!(ws.rng, settings.seed + k)
        σ0 = isodd(k) ? σ_start : -σ_start
        Threads.@spawn tabu_search!(ws, A, At, σ0, k, params, incumbent, t0, io)
    end
    results = map(fetch, tasks)

    # Among the searches that reached the best value up to `tol`, take the one that got
    # there in the fewest iterations. This ignores rounding noise between equal values and,
    # unlike the order of the log rows, does not depend on the timing of the threads.
    vmax = maximum(r -> r.value, results)
    candidates = filter(k -> results[k].value + params.tol >= vmax, 1:ntasks)
    best = argmin(k -> results[k].best_iteration, candidates)
    res = results[best]
    best_ws = solver.workspaces[best]

    # Recompute the value from the indicators, free of the drift of the incremental updates.
    mul!(best_ws.u, A, best_ws.bt)
    total = dot(best_ws.bs, best_ws.u)
    copyto!(sol.S, best_ws.bs)
    copyto!(sol.T, best_ws.bt)
    sol.value = settings.scaled ? abs(total) / (m * n) : abs(total)
    sol.sign = total < 0 ? Int8(-1) : Int8(1)

    sol.iterations = sum(r -> r.iterations, results)
    sol.perturbations = sum(r -> r.perturbations, results)
    sol.restarts = sum(r -> r.restarts, results)
    sol.best_task = best
    sol.best_iteration = res.best_iteration
    sol.improvements = incumbent.improvements
    sol.time_to_best = res.time_to_best
    if any(r -> r.status === :target, results)
        sol.termination_status = :target
    elseif any(r -> r.status === :max_time, results)
        sol.termination_status = :max_time
    else
        sol.termination_status = :max_iter
    end
    sol.runtime = (time_ns() - t0) / 1e9

    if pl >= 1
        print_tabu_summary_header(io)
        for (k, r) in enumerate(results)
            print_tabu_summary_row(io, k, r.iterations, r.perturbations, r.restarts, r.sign,
                r.value * scale, r.time_to_best)
        end
        print_tabu_footer(io, sol, ntasks)
    end

    return sol
end

"""
    tabu_search!(ws, A, At, σ0, id, params, incumbent, t0, io) -> NamedTuple

Run search number `id` of [`TabuSearchSolver`](@ref) with initial sign `σ0`, in the work
storage `ws` and with the parameters `params` derived by [`solve!`](@ref).

The search is divided into phases, each running from the start, a perturbation or a
restart up to the next one, or until the search stops. At the end of every phase the
search reports to the [`CutNorm.TabuSearchIncumbent`](@ref) shared by all searches, see
[`CutNorm.tabu_end_phase!`](@ref).

The best indicators of the search are left in `ws.bs` and `ws.bt`. The returned named
tuple holds their (unscaled, incrementally updated) `value` and `sign`, the iteration
and time at which they were found, the counters, and the `status` the search stopped
with: `:target`, `:max_iter`, `:max_time`, or `:stopped` if another search reached the
target first. `t0` is the `time_ns()` at the start of the solve.
"""
function tabu_search!(ws::TabuSearchWorkspace{T}, A::AbstractMatrix{T}, At::AbstractMatrix{T}, σ0::T, id::Int,
    params, incumbent::TabuSearchIncumbent{T}, t0::UInt64, io::IO) where {T<:AbstractFloat}
    (; s, t, u, v, tabu_until, es, et, bs, bt, perm, rng) = ws
    (; tenure, tenure_rand, stall, nflips, max_perturbations, refresh,
        max_iter, max_time, target, tol, scale, S0, T0, print_level) = params
    m, n = size(A)
    pl = print_level

    S0 === nothing ? rand!(rng, s, (zero(T), one(T))) : copyto!(s, S0)
    T0 === nothing ? rand!(rng, t, (zero(T), one(T))) : copyto!(t, T0)
    σ = σ0
    G = σ * tabu_recompute!(u, v, A, At, s, t)
    fill!(tabu_until, 0)

    # Elite: best value of the current episode, attained at (es, et).
    elite = G
    copyto!(es, s)
    copyto!(et, t)

    # Best value of this search, attained at (bs, bt) with sign best_sign.
    best = G
    copyto!(bs, s)
    copyto!(bt, t)
    best_sign = σ
    best_iteration = 0
    time_to_best = (time_ns() - t0) / 1e9

    perturbations = 0
    restarts = 0
    it = 0
    last = 0    # last iteration that improved the elite or diversified
    fails = 0   # perturbations in a row that did not improve the elite

    phase = 1
    phase_start = 0   # iteration after which the phase began
    phase_best = G    # best value of the phase

    status = best + tol >= target ? :target : :max_iter
    while status === :max_iter && it < max_iter
        it += 1

        # Scan all m + n flips: k is the best admissible flip, f the best flip overall.
        k = 0
        kval = typemin(T)
        f = 0
        fval = typemin(T)
        @inbounds for i in 1:m
            val = G + σ * (one(T) - 2 * s[i]) * u[i]
            if val > fval
                fval = val
                f = i
            end
            if val > kval && (tabu_until[i] < it || val > elite + tol)
                kval = val
                k = i
            end
        end
        @inbounds for j in 1:n
            val = G + σ * (one(T) - 2 * t[j]) * v[j]
            if val > fval
                fval = val
                f = m + j
            end
            if val > kval && (tabu_until[m+j] < it || val > elite + tol)
                kval = val
                k = m + j
            end
        end
        if k == 0
            k = f
            kval = fval
        end

        # Apply the flip; d = +1 adds the row or column to the set, d = -1 removes it.
        if k <= m
            d = one(T) - 2 * s[k]
            s[k] += d
            v .+= d .* view(At, :, k)
        else
            j = k - m
            d = one(T) - 2 * t[j]
            t[j] += d
            u .+= d .* view(A, :, j)
        end
        G = kval
        tabu_until[k] = it + tenure + rand(rng, 0:tenure_rand)
        (G > phase_best) && (phase_best = G)

        if G > elite + tol
            elite = G
            copyto!(es, s)
            copyto!(et, t)
            last = it
            fails = 0
            if G > best + tol
                best = G
                copyto!(bs, s)
                copyto!(bt, t)
                best_sign = σ
                best_iteration = it
                time_to_best = (time_ns() - t0) / 1e9
                if best + tol >= target
                    status = :target
                    break
                end
            end
        end

        if it - last > stall
            tabu_end_phase!(incumbent, io, pl, id, phase, it - phase_start, σ * phase_best, best, tol, scale, t0)
            fails += 1
            if fails > max_perturbations
                # Restart at a random point with the opposite sign.
                restarts += 1
                rand!(rng, s, (zero(T), one(T)))
                rand!(rng, t, (zero(T), one(T)))
                σ = -σ
                elite = typemin(T)
                fails = 0
            else
                # Continue from the elite with nflips random bits flipped.
                perturbations += 1
                copyto!(s, es)
                copyto!(t, et)
                randperm!(rng, perm)
                @inbounds for b in 1:nflips
                    i = perm[b]
                    if i <= m
                        s[i] = one(T) - s[i]
                    else
                        t[i-m] = one(T) - t[i-m]
                    end
                end
            end
            G = σ * tabu_recompute!(u, v, A, At, s, t)
            fill!(tabu_until, 0)
            last = it
            phase += 1
            phase_start = it
            phase_best = G
        elseif it % refresh == 0
            G = σ * tabu_recompute!(u, v, A, At, s, t)
        end

        if (it & 1023) == 0
            if incumbent.stop[]
                status = :stopped
            elseif (time_ns() - t0) / 1e9 >= max_time
                status = :max_time
            end
        end
    end
    (status === :target) && (incumbent.stop[] = true)
    tabu_end_phase!(incumbent, io, pl, id, phase, it - phase_start, σ * phase_best, best, tol, scale, t0)

    return (value=best, sign=best_sign > 0 ? Int8(1) : Int8(-1), iterations=it,
        perturbations=perturbations, restarts=restarts, best_iteration=best_iteration,
        time_to_best=time_to_best, status=status)
end

"""
    tabu_end_phase!(incumbent, io, print_level, id, phase, iterations, obj, best, tol, scale, t0)

Called by search `id` of [`CutNorm.tabu_search!`](@ref) at the end of each of its
phases. If the search's best value `best` beats the shared `incumbent`, the incumbent
is updated and its improvement counter increased. Depending on `print_level`, a row of
the log is printed: the phase number, the incumbent after the update, the number of
`iterations` of the phase, and `obj = s'A t` at the best point of the phase, whose sign
is the sign the phase searched with. The incumbent's lock is held while printing, so
that the `Best Value` and `Improv` columns increase down the log.
"""
function tabu_end_phase!(incumbent::TabuSearchIncumbent{T}, io::IO, print_level::Int, id::Int, phase::Int,
    iterations::Int, obj::T, best::T, tol::T, scale::T, t0::UInt64) where {T<:AbstractFloat}
    pl = print_level
    lock(incumbent.lock)
    try
        improved = best > incumbent.value + tol
        if improved
            incumbent.value = best
            incumbent.improvements += 1
        end
        if (pl >= 3) || (pl == 2 && (improved || should_print(phase))) || (pl == 1 && improved)
            print_tabu_row(io, id, phase, incumbent.value * scale, incumbent.improvements, iterations,
                obj * scale, (time_ns() - t0) / 1e9, improved)
        end
    finally
        unlock(incumbent.lock)
    end
    return nothing
end

"""
    tabu_recompute!(u, v, A, At, s, t) -> s'A*t

Set `u = A*t` and `v = A'*s` from scratch, using the stored transpose `At`, and return
`s'A*t`. Used by [`CutNorm.tabu_search!`](@ref) at the start, after every perturbation
and restart, and every `refresh` iterations.
"""
function tabu_recompute!(u, v, A, At, s, t)
    mul!(u, A, t)
    mul!(v, At, s)
    return dot(s, u)
end
