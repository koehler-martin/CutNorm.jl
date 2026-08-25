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