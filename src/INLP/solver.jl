mutable struct INLPSolver{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractSolver{T}
    A::M
    optimizer
    model::JuMP.Model
    settings::INLPSettings
end

"""
    build_model(A, optimizer) -> JuMP.Model

Construct the integer nonlinear program for the cut norm of `A` using the given
`optimizer`. Introduces binary vectors `s`, `t` and a binary sign variable `σ`,
with objective `max (2σ - 1) * s'At`. The variables are registered in the model
and can be retrieved as `model[:s]`, `model[:t]` and `model[:σ]`.
"""
function build_model(A::AbstractMatrix, optimizer)
    m, n = size(A)
    model = JuMP.Model(optimizer)
    JuMP.@variable(model, s[1:m], Bin)
    JuMP.@variable(model, t[1:n], Bin)
    JuMP.@variable(model, σ, Bin)
    JuMP.@objective(model, Max, (2 * σ - 1) * sum(A[i, j] * s[i] * t[j] for i in 1:m, j in 1:n))
    return model
end

function INLPSolver(A::AbstractMatrix{T}, optimizer; kwargs...) where {T<:AbstractFloat}
    settings = INLPSettings()
    populate!(settings; kwargs...)
    model = build_model(A, optimizer)
    return INLPSolver(A, optimizer, model, settings)
end

function solve!(solver::INLPSolver{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(solver.A)
    sol = INLPSolution(T, m, n)
    solve!(solver, sol; kwargs...)
end

function solve!(solver::INLPSolver{T}, sol::INLPSolution{T}; kwargs...) where {T}
    t0 = time_ns()
    reset!(sol)
    settings = solver.settings
    populate!(settings; kwargs...)

    model = solver.model
    m, n = sol.dims

    io = stdout
    pl = settings.print_level
    (pl >= 1) && (print_header(io); print_inlp_header_info(io, m, n, solver.optimizer, settings))

    pl < 2 ? JuMP.set_silent(model) : JuMP.unset_silent(model)
    JuMP.set_time_limit_sec(model, settings.max_time)

    JuMP.optimize!(model)

    if JuMP.has_values(model)
        s = model[:s]
        t = model[:t]
        for i in 1:m
            sol.S[i] = T(round(JuMP.value(s[i])))
        end
        for j in 1:n
            sol.T[j] = T(round(JuMP.value(t[j])))
        end
        obj = T(JuMP.objective_value(model))
        sol.value = settings.scaled ? obj / (m * n) : obj
    end

    sol.solve_time = T(JuMP.solve_time(model))
    sol.termination_status = _jump_status(JuMP.termination_status(model))
    sol.runtime = (time_ns() - t0) / 1e9

    (pl >= 1) && print_inlp_footer(io, sol)

    return sol
end
