"""
    QUBOSolution(A::AbstractMatrix)
    QUBOSolution(m::Int, n::Int)
    QUBOSolution(F::Type, m::Int, n::Int)

Result of [`QUBOSolver`](@ref), i.e. of `cutnorm(A; method = QUBO(optimizer))`.

The constructors allocate an empty solution for a problem of size `m x n` with
element type `F` (`Float64` unless given, or taken from `A`). Here `m` and `n` are
the dimensions of the *original* matrix, not of the augmented one. Normally you do
not call them yourself — [`cutnorm`](@ref) and [`solve!`](@ref) return a solution
already. Passing one to [`solve!`](@ref) explicitly lets you reuse the storage
across solves; it is reset first.

# Fields

| Field                | Description                                                        |
|:---------------------|:-------------------------------------------------------------------|
| `value`              | Objective value of the solver, scaled by `1/(m*n)` if `scaled`     |
| `S`, `T`             | Row and column indicator vectors of length `m` and `n`             |
| `dims`               | Problem size `(m, n)` of the original matrix                       |
| `runtime`            | Total elapsed wall-clock time in seconds, including model setup    |
| `solve_time`         | Time reported by the solver itself (`JuMP.solve_time`)             |
| `termination_status` | `:optimal`, `:max_time`, or the JuMP status name as a `Symbol`     |

The model is solved on the augmented `(m+1) x (n+1)` matrix; the extra pivot bits are
used to undo the bit-flip symmetry and are then dropped, so `S` and `T` refer to the
original matrix. `value` is the exact cut norm only if
`termination_status == :optimal`. If the solver returned no solution at all, `value`,
`S` and `T` stay at zero — check `termination_status` before using them.

See also [`QUBOSettings`](@ref), [`INLPSolution`](@ref), [`ILPSolution`](@ref).
"""
mutable struct QUBOSolution{F<:AbstractFloat}
    dims::Tuple{Int,Int}
    S::Vector{F}
    T::Vector{F}
    value::F
    runtime::F
    solve_time::F
    termination_status::Symbol
end

QUBOSolution(A::M) where {F<:AbstractFloat,M<:AbstractMatrix{F}} = QUBOSolution(F, size(A)...)

QUBOSolution(m::Int, n::Int) = QUBOSolution(Float64, m, n)

function QUBOSolution(F::Type, m::Int, n::Int)
    return QUBOSolution(
        (m, n),
        Vector{F}(undef, m),
        Vector{F}(undef, n),
        zero(F),
        zero(F),
        zero(F),
        :unknown
    )
end

function reset!(solution::QUBOSolution{F}) where {F<:AbstractFloat}
    fill!(solution.S, 0)
    fill!(solution.T, 0)
    solution.value = zero(F)
    solution.runtime = zero(F)
    solution.solve_time = zero(F)
    solution.termination_status = :unknown
end
