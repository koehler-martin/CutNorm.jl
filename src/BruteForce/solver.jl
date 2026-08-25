"""
    BruteForceSolver(A::AbstractMatrix; kwargs...)

Build the exact enumeration solver for the cut norm of `A`, i.e. the solver behind
`cutnorm(A; method = BruteForce())`. `kwargs` are forwarded to
[`BruteForceSettings`](@ref).

The solver keeps the matrix and three work vectors (`s`, `t` and `A*t`), so the
enumeration itself does not allocate.

The cost is `2^m * 2^n` objective evaluations. As a rule of thumb, keep `m + n` below
roughly 30 unless you also set `max_time` — on a time-out the best value found so far
is returned.

# Examples

```jldoctest
julia> solver = BruteForceSolver(Float64[1 -1; -1 1]);

julia> sol = solve!(solver);

julia> sol.value, sol.iterations, sol.termination_status
(1.0, 16, :optimal)
```

See also [`cutnorm`](@ref), [`BruteForce`](@ref), [`BruteForceSolution`](@ref).
"""
mutable struct BruteForceSolver{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractSolver{T}
    A::M
    s::Vector{T}
    t::Vector{T}
    At::Vector{T}
    settings::BruteForceSettings
end

function BruteForceSolver(A::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(A)
    settings = BruteForceSettings()
    populate!(settings; kwargs...)
    return BruteForceSolver(A, Vector{T}(undef, m), Vector{T}(undef, n), Vector{T}(undef, m), settings)
end

"""
    solve!(solver::BruteForceSolver; kwargs...) -> BruteForceSolution
    solve!(solver::BruteForceSolver, sol::BruteForceSolution; kwargs...)

Enumerate all `2^m * 2^n` pairs of binary indicator vectors and return the solution.
`kwargs` update the solver's [`BruteForceSettings`](@ref) before the run.

The outer loop runs over the column indicators `t`, so the product `A*t` is formed
once per `t` and reused for all `2^m` row indicators, leaving a dot product per pair.
The run stops early if `max_time` is exceeded, in which case `termination_status`
becomes `:max_time` instead of `:optimal`.

The second form writes into the solution object you pass in (it is reset first),
which lets you reuse the same storage across solves.
"""
function solve!(solver::BruteForceSolver{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(solver.A)
    sol = BruteForceSolution(T, m, n)
    solve!(solver, sol; kwargs...)
end

function solve!(solver::BruteForceSolver{T}, sol::BruteForceSolution{T}; kwargs...) where {T}
    t0 = time_ns()
    reset!(sol)
    settings = solver.settings
    populate!(settings; kwargs...)

    A = solver.A
    m, n = sol.dims

    io = stdout
    pl = settings.print_level
    (pl >= 1) && (print_header(io); print_bruteforce_header_info(io, m, n, settings); print_bruteforce_iter_header(io))

    s = solver.s
    t = solver.t
    At = solver.At

    prev_value = sol.value
    last_obj = zero(T)

    for kt in 0:(2^n-1)
        fill_binary_vector!(t, kt)
        mul!(At, A, t)

        for ks in 0:(2^m-1)
            fill_binary_vector!(s, ks)

            value = dot(s, At)
            abs_value = abs(value)

            settings.scaled && (abs_value /= (m * n))

            if abs_value > sol.value
                sol.value = abs_value
                sol.improvements += 1
                sol.best_iteration = sol.iterations + 1
                copyto!(sol.S, s)
                copyto!(sol.T, t)
            end

            sol.iterations += 1

            sol.runtime = (time_ns() - t0) / 1e9

            improved = sol.value > prev_value
            if (pl >= 3) || (pl >= 2 && (improved || should_print(sol.iterations))) || (pl == 1 && improved)
                print_bruteforce_row(io, sol.iterations, sol.improvements, abs_value, sol.value, sol.runtime, improved)
            end
            prev_value = sol.value

            if sol.runtime > settings.max_time
                sol.termination_status = :max_time
                @goto done
            end
        end
    end

    sol.termination_status = :optimal

    @label done
    sol.runtime = (time_ns() - t0) / 1e9

    (pl >= 1) && print_bruteforce_footer(io, sol)

    return sol
end

"""
    fill_binary_vector!(v, k)

Write the binary expansion of the integer `k` into `v`, least significant bit first,
so that `v[i]` is bit `i-1` of `k`. Used to enumerate the `2^length(v)` indicator
vectors.
"""
function fill_binary_vector!(v::AbstractVector, k::Integer)
    @inbounds for i in eachindex(v)
        v[i] = (k >> (i - 1)) & 1
    end
end
