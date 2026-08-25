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