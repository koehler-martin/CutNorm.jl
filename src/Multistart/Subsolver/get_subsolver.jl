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