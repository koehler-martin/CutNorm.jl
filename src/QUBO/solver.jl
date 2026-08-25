mutable struct QUBOSolver{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractSolver{T}
    A::M
    optimizer
    model::JuMP.Model
    settings::QUBOSettings
end

"""
    build_qubo_model(A, optimizer) -> JuMP.Model

Construct the quadratic unconstrained binary optimization (QUBO) problem for the
cut norm of `A` using the given `optimizer`. The matrix is first augmented to the
`(m+1) × (n+1)` matrix with zero row/column sums (see [`augment_matrix`](@ref)),
which removes the sign ambiguity, so the cut norm is simply

    max  Σ A_aug[i,j] * x[i] * y[j]

over binary `x`, `y` (no sign variable needed). The indicators are registered and
can be retrieved as `model[:x]` and `model[:y]`. The objective is quadratic in the
binaries, so a QUBO-capable solver (e.g. Gurobi, SCIP) is required.
"""
function build_qubo_model(A::AbstractMatrix, optimizer)
    A_aug = augment_matrix(A)
    p, q = size(A_aug)

    model = JuMP.Model(optimizer)
    JuMP.@variable(model, x[1:p], Bin)
    JuMP.@variable(model, y[1:q], Bin)
    JuMP.@objective(model, Max, sum(A_aug[i, j] * x[i] * y[j] for i in 1:p, j in 1:q))

    return model
end

function QUBOSolver(A::AbstractMatrix{T}, optimizer; kwargs...) where {T<:AbstractFloat}
    settings = QUBOSettings()
    populate!(settings; kwargs...)
    model = build_qubo_model(A, optimizer)
    return QUBOSolver(A, optimizer, model, settings)
end

function solve!(solver::QUBOSolver{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(solver.A)
    sol = QUBOSolution(T, m, n)
    solve!(solver, sol; kwargs...)
end

function solve!(solver::QUBOSolver{T}, sol::QUBOSolution{T}; kwargs...) where {T}
    t0 = time_ns()
    reset!(sol)
    settings = solver.settings
    populate!(settings; kwargs...)

    model = solver.model
    m, n = sol.dims

    io = stdout
    pl = settings.print_level
    (pl >= 1) && (print_header(io); print_qubo_header_info(io, m, n, solver.optimizer, settings))

    pl < 2 ? JuMP.set_silent(model) : JuMP.unset_silent(model)
    JuMP.set_time_limit_sec(model, settings.max_time)

    JuMP.optimize!(model)

    if JuMP.has_values(model)
        # The model is solved on the augmented (m+1) x (n+1) matrix, so x has
        # m+1 entries and y has n+1. Reduce back to the original size-m, size-n
        # indicators: x[m+1] and y[n+1] are pivot bits. If a pivot is 1 the
        # corresponding block is flipped, otherwise it is taken as-is. This
        # undoes the global bit-flip symmetry introduced by the augmentation.
        x = round.(JuMP.value.(model[:x]))
        y = round.(JuMP.value.(model[:y]))

        flip_rows = x[m+1] == 1
        flip_cols = y[n+1] == 1
        for i in 1:m
            sol.S[i] = flip_rows ? T(1 - x[i]) : T(x[i])
        end
        for j in 1:n
            sol.T[j] = flip_cols ? T(1 - y[j]) : T(y[j])
        end

        obj = T(JuMP.objective_value(model))
        sol.value = settings.scaled ? obj / (m * n) : obj
    end

    sol.solve_time = T(JuMP.solve_time(model))
    sol.termination_status = _jump_status(JuMP.termination_status(model))
    sol.runtime = (time_ns() - t0) / 1e9
    
    (pl >= 1) && print_qubo_footer(io, sol)

    return sol
end
