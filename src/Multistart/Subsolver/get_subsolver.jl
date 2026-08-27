"""
    get_subsolver(S::Type, model) -> (subsolver, stats)

Instantiate the subsolver type `S` for `model` together with a matching
`GenericExecutionStats` object, as needed by the multistart solvers.

Supported types are `TronSolver`, [`AlternatingLinearSearch`](@ref) and
[`GreedySolver`](@ref); any other `AbstractOptimizationSolver` raises an error naming
the three.
"""
function get_subsolver(::Type{TronSolver}, model)
    return TronSolver(model), GenericExecutionStats(model)
end

function get_subsolver(::Type{AlternatingLinearSearch}, model)
    return AlternatingLinearSearch(model), GenericExecutionStats(model)
end

function get_subsolver(::Type{GreedySolver}, model)
    return GreedySolver(model), GenericExecutionStats(model)
end

function get_subsolver(::Type{<:AbstractOptimizationSolver}, model)
    return error("This solver is not compatible. Try: `TronSolver`, `AlternatingLinearSearch`, or `GreedySolver`.")
end