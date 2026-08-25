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
