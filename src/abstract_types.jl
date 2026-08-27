"""
    AbstractSolver{T<:AbstractFloat}

Supertype of all cut norm solvers in CutNorm.jl, parameterized by the element type
`T` of the input matrix.

A solver stores everything that can be reused across calls — the matrix or its
JuMP/NLP model, preallocated work vectors, and a settings object — so that
[`solve!`](@ref) can be called repeatedly without rebuilding it.

Concrete subtypes: [`MultistartAugmentedSolver`](@ref), [`MultistartSignedSolver`](@ref),
[`BruteForceSolver`](@ref), [`INLPSolver`](@ref), [`ILPSolver`](@ref), [`QUBOSolver`](@ref).
"""
abstract type AbstractSolver{T<:AbstractFloat} end

"""
    AbstractSolution{T<:AbstractFloat}

Supertype for the result objects returned by [`solve!`](@ref) and [`cutnorm`](@ref).

All solution types carry at least the fields `value` (the cut norm estimate), `S` and
`T` (the row and column indicator vectors), `dims`, `runtime`, and
`termination_status`.

Concrete subtypes: [`MultistartAugmentedSolution`](@ref), [`MultistartSignedSolution`](@ref),
[`BruteForceSolution`](@ref), [`INLPSolution`](@ref), [`ILPSolution`](@ref), [`QUBOSolution`](@ref).
"""
abstract type AbstractSolution{T<:AbstractFloat} end

"""
    AbstractSettings

Supertype of the mutable settings objects of the solvers. Each solver owns one
settings instance, which can be modified in place with [`CutNorm.populate!`](@ref) —
this is how keyword arguments passed to [`cutnorm`](@ref) or [`solve!`](@ref) reach
the solver.

Concrete subtypes: [`MultistartSettings`](@ref), [`BruteForceSettings`](@ref),
[`INLPSettings`](@ref), [`ILPSettings`](@ref), [`QUBOSettings`](@ref).
"""
abstract type AbstractSettings end

"""
    populate!(s::AbstractSettings; kwargs...) -> AbstractSettings

Change the fields of the `AbstractSettings` object `s` in-place with the values
given through keyword arguments and return `s`.

Only existing fields are accepted; an `ArgumentError` is thrown if an
unknown keyword is supplied. Each value is automatically converted to
the field's declared type.
"""
function populate!(s::S; kwargs...) where {S<:AbstractSettings}
    for (name, val) in pairs(kwargs)
        if !hasfield(S, name)
            throw(ArgumentError("unknown setting: $name. Allowed keywords: $(fieldnames(S))"))
        end
        setproperty!(s, name, val)
    end
    return s
end
