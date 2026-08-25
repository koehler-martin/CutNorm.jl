"""
    MultistartSignedSolution(A::AbstractMatrix)
    MultistartSignedSolution(m::Int, n::Int)
    MultistartSignedSolution(F::Type, m::Int, n::Int)

Result of [`MultistartSignedSolver`](@ref), i.e. of `cutnorm(A; method =
MultistartSigned{S}())`.

The constructors allocate an empty solution for a problem of size `m x n` with
element type `F` (`Float64` unless given, or taken from `A`). Normally you do not
call them yourself — [`cutnorm`](@ref) and [`solve!`](@ref) return a solution
already. Passing one to [`solve!`](@ref) explicitly lets you reuse the storage
across solves; it is reset first.

# Fields

| Field                | Description                                                        |
|:---------------------|:-------------------------------------------------------------------|
| `value`              | Best cut norm value found, scaled by `1/(m*n)` if `scaled = true`   |
| `S`, `T`             | Row and column indicator vectors, entries in `{0, 1}`              |
| `sign`               | Sign `±1` of the objective that produced `value`                   |
| `dims`               | Problem size `(m, n)`                                              |
| `restarts`           | Number of restarts performed                                       |
| `improvements`       | How often the best value improved                                  |
| `best_restart`       | Index of the restart that produced `value`                         |
| `best_initial_guess` | Sobol point `x = [s; t]` that led to the best value                |
| `all_solutions`      | Per-restart `(objective, S, T)` tuples, if `save_all_solutions`    |
| `runtime`            | Total elapsed time in seconds                                      |
| `termination_status` | `:max_restarts` or `:max_time`                                     |

Note that both sign variants are solved per restart, so `all_solutions` holds two
entries per restart.

See also [`MultistartAugmentedSolution`](@ref), [`MultistartSettings`](@ref).
"""
mutable struct MultistartSignedSolution{F<:AbstractFloat}
    dims::Tuple{Int,Int}
    S::Vector{F}
    T::Vector{F}
    restarts::Int
    improvements::Int
    value::F
    sign::Int8
    best_restart::Int
    best_initial_guess::Vector{F}
    all_solutions::Vector{@NamedTuple{objective::F, S::Vector{F}, T::Vector{F}}}
    runtime::F
    termination_status::Symbol
end

MultistartSignedSolution(A::M) where {F<:AbstractFloat,M<:AbstractMatrix{F}} = MultistartSignedSolution(F, size(A)...)

MultistartSignedSolution(m::Int, n::Int) = MultistartSignedSolution(Float64, m, n)

function MultistartSignedSolution(F::Type, m::Int, n::Int)
    return MultistartSignedSolution(
        (m, n),
        Vector{F}(undef, m),
        Vector{F}(undef, n),
        0,
        0,
        zero(F),
        Int8(0),
        0,
        Vector{F}(undef, m + n),
        Vector{@NamedTuple{objective::F, S::Vector{F}, T::Vector{F}}}(),
        zero(F),
        :unknown
    )
end

function reset!(solution::MultistartSignedSolution{F}) where {F<:AbstractFloat}
    fill!(solution.S, 0)
    fill!(solution.T, 0)
    solution.restarts = 0
    solution.improvements = 0
    solution.value = zero(F)
    solution.sign = Int8(0)
    solution.best_restart = 0
    fill!(solution.best_initial_guess, 0)
    empty!(solution.all_solutions)
    solution.runtime = zero(F)
    solution.termination_status = :unknown
end