mutable struct MultistartAugmentedSolver{
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

function MultistartAugmentedSolver(
    A::M,
    subsolver::Type{<:AbstractOptimizationSolver}=TronSolver;
    kwargs...
) where {T<:AbstractFloat,M<:AbstractMatrix{T}}
    A_aug = augment_matrix(A)

    settings = MultistartSettings()
    populate!(settings; kwargs...)

    model = BilinearModel(A_aug)

    subsolver, subsolver_stats = get_subsolver(subsolver, model)

    m, n = size(A_aug)
    d = m + n

    seq = skip(SobolSeq(d), d)

    return MultistartAugmentedSolver(
        subsolver,
        subsolver_stats,
        model,
        seq,
        settings
    )
end

function solve!(solver::MultistartAugmentedSolver{T}; kwargs...) where {T<:AbstractFloat}
    m, n = solver.model.data.dims
    sol = MultistartAugmentedSolution(T, m - 1, n - 1)
    solve!(solver, sol; kwargs...)
end

function solve!(solver::MultistartAugmentedSolver{T}, sol::MultistartAugmentedSolution{T}; kwargs...) where {T}
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
    method_name = "Augmented Multistart"
    method_description = "Removes sign ambiguity via matrix augmentation"
    m, n = sol.dims
    pl = settings.print_level
    (pl >= 1) && (print_header(io); print_header_info(io, m, n, method_name, method_description, subsolver_name, settings); print_augmented_iter_header(io))

    subsolver_verbose = pl >= 4 ? 1 : 0
    prev_value = sol.value

    while !done
        sol.restarts += 1
        next!(solver.seq, solver.model.meta.x0)

        subsolver_max_time = settings.max_time - (time_ns() - t0) / 1e9
        solve!(subsolver, model, subsolver_stats; max_time=subsolver_max_time, verbose=subsolver_verbose)

        sub_iter = subsolver_stats.iter
        sub_obj = subsolver_stats.objective

        update_solution!(sol, solver)

        current_runtime = (time_ns() - t0) / 1e9
        sol.runtime = current_runtime

        improved = sol.value > prev_value
        if (pl >= 3) || (pl == 2 && (improved || should_print(sol.restarts))) || (pl == 1 && improved)
            print_augmented_iter(io, sol.restarts, sol.value, sol.improvements,
                sub_iter, sub_obj, sol.runtime, improved)
        end
        prev_value = sol.value

        SolverCore.reset!(subsolver)
        SolverCore.reset!(subsolver_stats)

        (sol.restarts >= settings.max_restarts) && (done = true; sol.termination_status = :max_restarts)
        (sol.runtime >= settings.max_time) && (done = true; sol.termination_status = :max_time)
    end

    (pl >= 1) && print_augmented_footer(io, sol)

    return sol
end

function update_solution!(sol::MultistartAugmentedSolution, solver::MultistartAugmentedSolver)
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
        copyto!(sol.best_initial_guess, solver.model.meta.x0)

        fill_ST!(sol.S, sol.T, x)
    end

    if settings.save_all_solutions
        current_S = similar(sol.S)
        current_T = similar(sol.T)
        fill_ST!(current_S, current_T, x)

        push!(sol.all_solutions, (objective=objective, S=current_S, T=current_T))
    end
end

function fill_ST!(S::AbstractVector, T::AbstractVector, x::AbstractVector)
    m = length(S)
    n = length(T)
    !(length(x) == (m + n + 2)) && throw(DimensionMismatch("Length of x is not length of S and T + 2. Received length(x)=$(length(x)), length(S)=$(length(S)), length(T)=$(length(T))"))

    # f and g are the pivot bits (augmented row/col entries); XOR-ing with them
    # normalizes the {0,1} solution against the global bit-flip symmetry.

    f = x[m+1]
    g = x[m+n+2]
    for i in 1:m
        S[i] = abs(x[i] - f)
    end
    for j in 1:n
        T[j] = abs(x[m+1+j] - g)
    end
end