"""
    BruteForceSolution(A::AbstractMatrix)
    BruteForceSolution(m::Int, n::Int)
    BruteForceSolution(F::Type, m::Int, n::Int)

Result of [`BruteForceSolver`](@ref), i.e. of `cutnorm(A; method = BruteForce())`.

The constructors allocate an empty solution for a problem of size `m x n` with
element type `F` (`Float64` unless given, or taken from `A`). Normally you do not
call them yourself — [`cutnorm`](@ref) and [`solve!`](@ref) return a solution
already. Passing one to [`solve!`](@ref) explicitly lets you reuse the storage
across solves; it is reset first.

# Fields

| Field                | Description                                                       |
|:---------------------|:------------------------------------------------------------------|
| `value`              | Cut norm value, scaled by `1/(m*n)` if `scaled = true`            |
| `S`, `T`             | Row and column indicator vectors, entries in `{0, 1}`             |
| `dims`               | Problem size `(m, n)`                                             |
| `iterations`          | Number of `(s, t)` pairs evaluated                               |
| `improvements`       | How often the best value improved                                 |
| `best_iteration`     | Iteration that produced `value`                                   |
| `runtime`            | Total elapsed time in seconds                                     |
| `termination_status` | `:optimal` if the enumeration finished, `:max_time` otherwise     |

`value` is the exact cut norm only if `termination_status == :optimal`; on
`:max_time` it is the best value found so far, and hence a lower bound.

See also [`BruteForceSettings`](@ref).
"""
mutable struct BruteForceSolution{F<:AbstractFloat}
    dims::Tuple{Int,Int}
    S::Vector{F}
    T::Vector{F}
    value::F
    iterations::Int
    improvements::Int
    best_iteration::Int
    runtime::F
    termination_status::Symbol
end

BruteForceSolution(A::M) where {F<:AbstractFloat,M<:AbstractMatrix{F}} = BruteForceSolution(F, size(A)...)

BruteForceSolution(m::Int, n::Int) = BruteForceSolution(Float64, m, n)

function BruteForceSolution(F::Type, m::Int, n::Int)
    return BruteForceSolution(
        (m, n),
        Vector{F}(undef, m),
        Vector{F}(undef, n),
        zero(F),
        0,
        0,
        0,
        zero(F),
        :unknown
    )
end

function reset!(solution::BruteForceSolution{F}) where {F<:AbstractFloat}
    fill!(solution.S, 0)
    fill!(solution.T, 0)
    solution.value = zero(F)
    solution.iterations = 0
    solution.improvements = 0
    solution.best_iteration = 0
    solution.runtime = zero(F)
    solution.termination_status = :unknown
end
