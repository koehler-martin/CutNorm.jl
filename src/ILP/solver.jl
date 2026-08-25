mutable struct ILPSolver{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractSolver{T}
    A::M
    optimizer
    model::JuMP.Model
    settings::ILPSettings
end

"""
    compute_bigM(A) -> eltype(A)

Return a valid big-M bound for the absolute-value reformulation, equal to
`2 * max(Σ positive entries, -Σ negative entries)`. This is an upper bound on
twice the attainable `|s'At|`, which is large enough to keep the inactive
absolute-value constraint slack.
"""
function compute_bigM(A::AbstractMatrix{T}) where {T}
    sum_pos = sum(a for a in A if a > zero(T); init=zero(T))
    sum_neg = sum(a for a in A if a < zero(T); init=zero(T))
    return 2 * max(sum_pos, -sum_neg)
end

"""
    build_ilp_model(A, optimizer) -> JuMP.Model

Construct the integer **linear** program for the cut norm of `A` using the given
`optimizer`. The product `x[i] * y[j]` of the binary row/column indicators is
linearized with McCormick constraints on `w[i,j]` (exact for binary `x`, `y`),
and the absolute value of the bilinear term is modeled with a binary sign
variable `sgn` and a big-M reformulation (no indicator constraints, so
MILP-only solvers such as HiGHS are supported).

The indicators are registered and can be retrieved as `model[:x]` and
`model[:y]`; the objective value is `model[:z]`.
"""
function build_ilp_model(A::AbstractMatrix, optimizer)
    m, n = size(A)
    M = compute_bigM(A)

    model = JuMP.Model(optimizer)

    JuMP.@variable(model, x[1:m], Bin)
    JuMP.@variable(model, y[1:n], Bin)
    JuMP.@variable(model, 0 <= w[1:m, 1:n] <= 1)
    JuMP.@variable(model, sgn, Bin)
    JuMP.@variable(model, z >= 0)

    JuMP.@expression(model, bilinearterm, sum(A[i, j] * w[i, j] for i in 1:m, j in 1:n))

    # McCormick linearization of w[i,j] = x[i] * y[j] (exact for binary x, y)
    JuMP.@constraint(model, [i in 1:m, j in 1:n], w[i, j] <= x[i])
    JuMP.@constraint(model, [i in 1:m, j in 1:n], w[i, j] <= y[j])
    JuMP.@constraint(model, [i in 1:m, j in 1:n], w[i, j] >= x[i] + y[j] - 1)

    # z = |bilinearterm| via big-M (sgn selects the active branch)
    JuMP.@constraint(model, z <= bilinearterm + M * sgn)
    JuMP.@constraint(model, z <= -bilinearterm + M * (1 - sgn))

    JuMP.@objective(model, Max, z)

    return model
end

function ILPSolver(A::AbstractMatrix{T}, optimizer; kwargs...) where {T<:AbstractFloat}
    settings = ILPSettings()
    populate!(settings; kwargs...)
    model = build_ilp_model(A, optimizer)
    return ILPSolver(A, optimizer, model, settings)
end

function solve!(solver::ILPSolver{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(solver.A)
    sol = ILPSolution(T, m, n)
    solve!(solver, sol; kwargs...)
end

function solve!(solver::ILPSolver{T}, sol::ILPSolution{T}; kwargs...) where {T}
    t0 = time_ns()
    reset!(sol)
    settings = solver.settings
    populate!(settings; kwargs...)

    model = solver.model
    m, n = sol.dims

    io = stdout
    pl = settings.print_level
    (pl >= 1) && (print_header(io); print_ilp_header_info(io, m, n, solver.optimizer, settings))

    pl < 2 ? JuMP.set_silent(model) : JuMP.unset_silent(model)
    JuMP.set_time_limit_sec(model, settings.max_time)

    JuMP.optimize!(model)

    if JuMP.has_values(model)
        x = model[:x]
        y = model[:y]
        for i in 1:m
            sol.S[i] = T(round(JuMP.value(x[i])))
        end
        for j in 1:n
            sol.T[j] = T(round(JuMP.value(y[j])))
        end
        obj = T(JuMP.objective_value(model))
        sol.value = settings.scaled ? obj / (m * n) : obj
    end

    sol.solve_time = T(JuMP.solve_time(model))
    sol.termination_status = _jump_status(JuMP.termination_status(model))
    sol.runtime = (time_ns() - t0) / 1e9
    
    (pl >= 1) && print_ilp_footer(io, sol)

    return sol
end
