abstract type AbstractCutNormMethod end
struct MultistartAugmented{S} <: AbstractCutNormMethod end
struct MultistartSigned{S} <: AbstractCutNormMethod end
struct SDPRelaxation <: AbstractCutNormMethod end
struct BruteForce <: AbstractCutNormMethod end
struct INLP <: AbstractCutNormMethod
    optimizer
end
struct ILP <: AbstractCutNormMethod
    optimizer
end
struct QUBO <: AbstractCutNormMethod
    optimizer
end

function cutnorm(A::AbstractMatrix;
    method::AbstractCutNormMethod=MultistartSigned{AlternatingLinearSearch}(),
    kwargs...
)
    return _cutnorm(A, method; kwargs...)
end

function _cutnorm(A, ::MultistartAugmented{S}; kwargs...) where S
    solver = MultistartAugmentedSolver(A, S; kwargs...)
    #solution = MultistartSignedSolution(A)
    return solve!(solver)
end

function _cutnorm(A, ::MultistartSigned{S}; kwargs...) where S
    solver = MultistartSignedSolver(A, S; kwargs...)
    return solve!(solver)
end

function _cutnorm(A, ::SDPRelaxation; kwargs...)
    # future SDP implementation
end

function _cutnorm(A, ::BruteForce; kwargs...)
    solver = BruteForceSolver(A; kwargs...)
    return solve!(solver)
end

function _cutnorm(A, method::INLP; kwargs...)
    solver = INLPSolver(A, method.optimizer; kwargs...)
    return solve!(solver)
end

function _cutnorm(A, method::ILP; kwargs...)
    solver = ILPSolver(A, method.optimizer; kwargs...)
    return solve!(solver)
end

function _cutnorm(A, method::QUBO; kwargs...)
    solver = QUBOSolver(A, method.optimizer; kwargs...)
    return solve!(solver)
end
