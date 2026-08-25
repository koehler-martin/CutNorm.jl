mutable struct MultistartSignedSolver{
    T<:AbstractFloat,
    S1<:AbstractOptimizationSolver,
    S2<:AbstractExecutionStats,
    M<:AbstractBilinearModel{T,Vector{T}},
    Seq<:AbstractSobolSeq
} <: AbstractSolver{T}
    subsolver::S1
    subsolver_stats::S2
    model::M
    seq::Seq
    settings::MultistartSettings
end

function MultistartSignedSolver(
    A::M,
    subsolver::Type{<:AbstractOptimizationSolver}=TronSolver;
    kwargs...
) where {T<:AbstractFloat,M<:AbstractMatrix{T}}
    settings = MultistartSettings()
    populate!(settings; kwargs...)

    model = SignedBilinearModel(A)

    subsolver, subsolver_stats = get_subsolver(subsolver, model)

    m, n = size(A)
    d = m + n

    seq = skip(SobolSeq(d), d)

    return MultistartSignedSolver(
        subsolver,
        subsolver_stats,
        model,
        seq,
        settings
    )
end

function solve!(solver::MultistartSignedSolver{T}; kwargs...) where {T<:AbstractFloat}
    m, n = solver.model.data.dims
    sol = MultistartSignedSolution(T, m, n)
    solve!(solver, sol; kwargs...)
end

function solve!(solver::MultistartSignedSolver{T}, sol::MultistartSignedSolution{T}; kwargs...) where {T}
    t0 = time_ns()
    reset!(sol)
    settings = solver.settings
    populate!(settings; kwargs...)

    subsolver = solver.subsolver
    subsolver_stats = solver.subsolver_stats

    model = solver.model
    done = false

    io = stdout
    subsolver_name = string(nameof(typeof(subsolver)))
    method_name = "Signed Multistart"
    method_description = "Solves both +/- objectives per restart"
    m, n = sol.dims
    pl = settings.print_level
    (pl >= 1) && (print_header(io); print_header_info(io, m, n, method_name, method_description, subsolver_name, settings); print_signed_iter_header(io))

    subsolver_verbose = pl >= 4 ? 1 : 0
    prev_value = sol.value

    while !done
        sol.restarts += 1
        next!(solver.seq, solver.model.meta.x0)

        sub_obj_pos = NaN
        sub_obj_neg = NaN
        sub_iter_pos = 0
        sub_iter_neg = 0

        for σ in (Int8(1), Int8(-1))
            model.sign = σ

            subsolver_max_time = settings.max_time - (time_ns() - t0) / 1e9
            (subsolver_max_time <= 0) && break
            solve!(subsolver, model, subsolver_stats; max_time=subsolver_max_time, verbose=subsolver_verbose)

            if σ == Int8(1)
                sub_obj_pos = subsolver_stats.objective
                sub_iter_pos = subsolver_stats.iter
            else
                sub_obj_neg = subsolver_stats.objective
                sub_iter_neg = subsolver_stats.iter
            end

            update_solution!(sol, solver)

            SolverCore.reset!(subsolver)
            SolverCore.reset!(subsolver_stats)
        end

        current_runtime = (time_ns() - t0) / 1e9
        sol.runtime = current_runtime

        improved = sol.value > prev_value
        if (pl >= 3) || (pl == 2 && (improved || should_print(sol.restarts))) || (pl == 1 && improved)
            print_signed_row(io, sol.restarts, sol.value, sol.improvements,
                sub_iter_pos, sub_obj_pos, sub_iter_neg, sub_obj_neg, sol.runtime, improved)
        end
        prev_value = sol.value

        (sol.restarts >= settings.max_restarts) && (done = true; sol.termination_status = :max_restarts)
        (sol.runtime >= settings.max_time) && (done = true; sol.termination_status = :max_time)
    end

    (pl >= 1) && print_signed_footer(io, sol)

    return sol
end

function update_solution!(sol::MultistartSignedSolution, solver::MultistartSignedSolver)
    model = solver.model
    subsolver_stats = solver.subsolver_stats
    settings = solver.settings

    m, n = sol.dims
    x = subsolver_stats.solution
    @. x = round(x)
    objective = -obj(model, x)

    settings.scaled && (objective /= (m * n))

    if objective > sol.value
        sol.improvements += 1
        sol.best_restart = sol.restarts
        sol.value = objective
        sol.sign = sign(model)
        copyto!(sol.best_initial_guess, solver.model.meta.x0)

        copyto!(sol.S, 1, x, 1, m)
        copyto!(sol.T, 1, x, m + 1, n)
    end

    if settings.save_all_solutions
        current_S = x[1:m]
        current_T = x[m+1:m+n]

        push!(sol.all_solutions, (objective=objective, S=current_S, T=current_T))
    end
end