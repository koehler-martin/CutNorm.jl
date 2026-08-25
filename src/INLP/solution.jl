"""
    INLPSolution(A::AbstractMatrix)
    INLPSolution(m::Int, n::Int)
    INLPSolution(F::Type, m::Int, n::Int)

Result of [`INLPSolver`](@ref), i.e. of `cutnorm(A; method = INLP(optimizer))`.

The constructors allocate an empty solution for a problem of size `m x n` with
element type `F` (`Float64` unless given, or taken from `A`). Normally you do not
call them yourself — [`cutnorm`](@ref) and [`solve!`](@ref) return a solution
already. Passing one to [`solve!`](@ref) explicitly lets you reuse the storage
across solves; it is reset first.

# Fields

| Field                | Description                                                        |
|:---------------------|:-------------------------------------------------------------------|
| `value`              | Objective value of the solver, scaled by `1/(m*n)` if `scaled`     |
| `S`, `T`             | Row and column indicator vectors, entries in `{0, 1}`              |
| `dims`               | Problem size `(m, n)`                                              |
| `runtime`            | Total elapsed wall-clock time in seconds, including model setup    |
| `solve_time`         | Time reported by the solver itself (`JuMP.solve_time`)             |
| `termination_status` | `:optimal`, `:max_time`, or the JuMP status name as a `Symbol`     |

`value` is the exact cut norm only if `termination_status == :optimal`. If the solver
returned no solution at all, `value`, `S` and `T` stay at zero — check
`termination_status` before using them.

See also [`INLPSettings`](@ref), [`ILPSolution`](@ref), [`QUBOSolution`](@ref).
"""
mutable struct INLPSolution{F<:AbstractFloat}
    dims::Tuple{Int,Int}
    S::Vector{F}
    T::Vector{F}
    value::F
    runtime::F
    solve_time::F
    termination_status::Symbol
end

INLPSolution(A::M) where {F<:AbstractFloat,M<:AbstractMatrix{F}} = INLPSolution(F, size(A)...)

INLPSolution(m::Int, n::Int) = INLPSolution(Float64, m, n)

function INLPSolution(F::Type, m::Int, n::Int)
    return INLPSolution(
        (m, n),
        Vector{F}(undef, m),
        Vector{F}(undef, n),
        zero(F),
        zero(F),
        zero(F),
        :unknown
    )
end

function reset!(solution::INLPSolution{F}) where {F<:AbstractFloat}
    fill!(solution.S, 0)
    fill!(solution.T, 0)
    solution.value = zero(F)
    solution.runtime = zero(F)
    solution.solve_time = zero(F)
    solution.termination_status = :unknown
end
