"""
    MultistartAugmentedSolution(A::AbstractMatrix)
    MultistartAugmentedSolution(m::Int, n::Int)
    MultistartAugmentedSolution(F::Type, m::Int, n::Int)

Result of [`MultistartAugmentedSolver`](@ref), i.e. of `cutnorm(A; method =
MultistartAugmented{S}())`.

The constructors allocate an empty solution for a problem of size `m x n` with
element type `F` (`Float64` unless given, or taken from `A`). Here `m` and `n` are
the dimensions of the *original* matrix, not of the augmented one. Normally you do
not call them yourself — [`cutnorm`](@ref) and [`solve!`](@ref) return a solution
already. Passing one to [`solve!`](@ref) explicitly lets you reuse the storage
across solves; it is reset first.

# Fields

| Field                | Description                                                      |
|:---------------------|:-----------------------------------------------------------------|
| `value`              | Best cut norm value found, scaled by `1/(m*n)` if `scaled = true` |
| `S`, `T`             | Row and column indicator vectors, entries in `{0, 1}`            |
| `dims`               | Problem size `(m, n)` of the original matrix                     |
| `restarts`           | Number of restarts performed                                     |
| `improvements`       | How often the best value improved                                |
| `best_restart`       | Index of the restart that produced `value`                       |
| `best_initial_guess` | Sobol point `x = [s; t]` (length `m+n+2`) behind the best value   |
| `all_solutions`      | Per-restart `(objective, S, T)` tuples, if `save_all_solutions`  |
| `runtime`            | Total elapsed time in seconds                                    |
| `termination_status` | `:max_restarts` or `:max_time`                                   |

`S` and `T` have length `m` and `n`: the augmented row and column are used to undo
the bit-flip symmetry and are then dropped.

See also [`MultistartSignedSolution`](@ref), [`MultistartSettings`](@ref).
"""
mutable struct MultistartAugmentedSolution{F<:AbstractFloat}
    dims::Tuple{Int,Int}
    S::Vector{F}
    T::Vector{F}
    restarts::Int
    improvements::Int
    value::F
    best_restart::Int
    best_initial_guess::Vector{F}
    all_solutions::Vector{@NamedTuple{objective::F, S::Vector{F}, T::Vector{F}}}
    runtime::F
    termination_status::Symbol
end

MultistartAugmentedSolution(A::M) where {F<:AbstractFloat,M<:AbstractMatrix{F}} = MultistartAugmentedSolution(F, size(A)...)

MultistartAugmentedSolution(m::Int, n::Int) = MultistartAugmentedSolution(Float64, m, n)

function MultistartAugmentedSolution(F::Type, m::Int, n::Int)
    return MultistartAugmentedSolution(
        (m, n),
        Vector{F}(undef, m),
        Vector{F}(undef, n),
        0,
        0,
        zero(F),
        0,
        Vector{F}(undef, m + n + 2),
        Vector{@NamedTuple{objective::F, S::Vector{F}, T::Vector{F}}}(),
        zero(F),
        :unknown
    )
end

function reset!(solution::MultistartAugmentedSolution{F}) where {F<:AbstractFloat}
    fill!(solution.S, 0)
    fill!(solution.T, 0)
    solution.restarts = 0
    solution.improvements = 0
    solution.value = zero(F)
    solution.best_restart = 0
    fill!(solution.best_initial_guess, 0)
    empty!(solution.all_solutions)
    solution.runtime = zero(F)
    solution.termination_status = :unknown
end