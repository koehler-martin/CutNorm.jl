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
