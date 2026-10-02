"""
    TabuSearchSolution(A::AbstractMatrix)
    TabuSearchSolution(m::Int, n::Int)
    TabuSearchSolution(F::Type, m::Int, n::Int)

Result of [`TabuSearchSolver`](@ref), i.e. of `cutnorm(A; method = TabuSearch())`.

The constructors allocate an empty solution for a problem of size `m x n` with
element type `F` (`Float64` unless given, or taken from `A`). Normally you do not
call them yourself — [`cutnorm`](@ref) and [`solve!`](@ref) return a solution
already. Passing one to [`solve!`](@ref) explicitly lets you reuse the storage
across solves; it is reset first.

# Fields

| Field                | Description                                                         |
|:---------------------|:--------------------------------------------------------------------|
| `value`              | Best cut norm value found, scaled by `1/(m*n)` if `scaled = true`    |
| `S`, `T`             | Row and column indicator vectors, entries in `{0, 1}`               |
| `sign`               | Sign `±1` of the sum of `A` over `S × T`                            |
| `dims`               | Problem size `(m, n)`                                               |
| `iterations`         | Number of flips, summed over all searches                           |
| `perturbations`      | Number of perturbations of the elite, summed over all searches      |
| `restarts`           | Number of random restarts, summed over all searches                 |
| `best_task`          | Index of the search that produced `value`                           |
| `best_iteration`     | Iteration of that search that produced `value`                      |
| `improvements`       | How often the best value over all searches improved, checked at the end of every phase; depends on the order in which the searches end their phases |
| `time_to_best`       | Seconds from the start of [`solve!`](@ref) until `value` was found  |
| `runtime`            | Total elapsed time in seconds                                       |
| `termination_status` | `:target`, `:max_iter` or `:max_time`                               |

`value` is recomputed from `S` and `T` once the searches have finished, so it carries
none of the rounding drift of the incremental updates. Tabu search is a heuristic:
`value` is a lower bound on the cut norm.

See also [`TabuSearchSettings`](@ref), [`MultistartSignedSolution`](@ref).
"""
mutable struct TabuSearchSolution{F<:AbstractFloat}
    dims::Tuple{Int,Int}
    S::Vector{F}
    T::Vector{F}
    value::F
    sign::Int8
    iterations::Int
    perturbations::Int
    restarts::Int
    best_task::Int
    best_iteration::Int
    improvements::Int
    time_to_best::F
    runtime::F
    termination_status::Symbol
end

TabuSearchSolution(A::M) where {F<:AbstractFloat,M<:AbstractMatrix{F}} = TabuSearchSolution(F, size(A)...)

TabuSearchSolution(m::Int, n::Int) = TabuSearchSolution(Float64, m, n)

function TabuSearchSolution(F::Type, m::Int, n::Int)
    return TabuSearchSolution(
        (m, n),
        Vector{F}(undef, m),
        Vector{F}(undef, n),
        zero(F),
        Int8(0),
        0,
        0,
        0,
        0,
        0,
        0,
        zero(F),
        zero(F),
        :unknown
    )
end

function reset!(solution::TabuSearchSolution{F}) where {F<:AbstractFloat}
    fill!(solution.S, 0)
    fill!(solution.T, 0)
    solution.value = zero(F)
    solution.sign = Int8(0)
    solution.iterations = 0
    solution.perturbations = 0
    solution.restarts = 0
    solution.best_task = 0
    solution.best_iteration = 0
    solution.improvements = 0
    solution.time_to_best = zero(F)
    solution.runtime = zero(F)
    solution.termination_status = :unknown
end
