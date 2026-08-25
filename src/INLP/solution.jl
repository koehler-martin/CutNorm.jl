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
