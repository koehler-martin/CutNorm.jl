"""
    ILPSolution(A::AbstractMatrix)
    ILPSolution(m::Int, n::Int)
    ILPSolution(F::Type, m::Int, n::Int)

Result of [`ILPSolver`](@ref), i.e. of `cutnorm(A; method = ILP(optimizer))`.

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

`value` is the exact cut norm only if `termination_status == :optimal`; on
`:max_time` it is the best incumbent the solver found, and hence a lower bound. If
the solver returned no solution at all, `value`, `S` and `T` stay at zero — check
`termination_status` before using them.

See also [`ILPSettings`](@ref), [`INLPSolution`](@ref), [`QUBOSolution`](@ref).
"""
mutable struct ILPSolution{F<:AbstractFloat}
    dims::Tuple{Int,Int}
    S::Vector{F}
    T::Vector{F}
    value::F
    runtime::F
    solve_time::F
    termination_status::Symbol
end

ILPSolution(A::M) where {F<:AbstractFloat,M<:AbstractMatrix{F}} = ILPSolution(F, size(A)...)

ILPSolution(m::Int, n::Int) = ILPSolution(Float64, m, n)

function ILPSolution(F::Type, m::Int, n::Int)
    return ILPSolution(
        (m, n),
        Vector{F}(undef, m),
        Vector{F}(undef, n),
        zero(F),
        zero(F),
        zero(F),
        :unknown
    )
end

function reset!(solution::ILPSolution{F}) where {F<:AbstractFloat}
    fill!(solution.S, 0)
    fill!(solution.T, 0)
    solution.value = zero(F)
    solution.runtime = zero(F)
    solution.solve_time = zero(F)
    solution.termination_status = :unknown
end
