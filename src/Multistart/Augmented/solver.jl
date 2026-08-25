"""
    MultistartAugmentedSolver(A::AbstractMatrix, subsolver = TronSolver; kwargs...)

Build the multistart solver for the augmented formulation of the cut norm of `A`,
i.e. the solver behind `cutnorm(A; method = MultistartAugmented{subsolver}())`.

The matrix is augmented once, at construction time, with
[`CutNorm.augment_matrix`](@ref); the internal [`BilinearModel`](@ref) therefore has
dimension `(m+1) + (n+1)`. `subsolver` is the *type* of the local solver used in each
restart: [`AlternatingLinearSearch`](@ref), `TronSolver`, or [`GreedySolver`](@ref).
Any other `AbstractOptimizationSolver` raises an error. `kwargs` are forwarded to
[`MultistartSettings`](@ref).

The solver owns the model, the subsolver and its statistics object, a Sobol sequence
for the initial points, and the settings. All of it is reused, so calling
[`solve!`](@ref) repeatedly allocates nothing new — but note that the Sobol sequence
keeps advancing, so a second `solve!` explores different starting points than the
first.

# Examples

```julia
solver = MultistartAugmentedSolver(A, AlternatingLinearSearch; max_restarts = 200)
sol = solve!(solver)
```

See also [`cutnorm`](@ref), [`MultistartSignedSolver`](@ref),
[`MultistartAugmentedSolution`](@ref), [`MultistartSettings`](@ref).
"""
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

"""
    solve!(solver::MultistartAugmentedSolver; kwargs...) -> MultistartAugmentedSolution
    solve!(solver::MultistartAugmentedSolver, sol::MultistartAugmentedSolution; kwargs...)

Run the multistart loop on the augmented model and return the solution. `kwargs`
update the solver's [`MultistartSettings`](@ref) before the run, so options can be
changed between solves.

In each restart the next Sobol point is used as the initial guess, the local
subsolver is called once, and the rounded result updates the incumbent. The loop
stops once `max_restarts` restarts have been done or `max_time` seconds have elapsed,
and sets `termination_status` accordingly.

The second form writes into the solution object you pass in (it is reset first),
which lets you reuse the same storage across solves.
"""
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

"""
    update_solution!(sol::MultistartAugmentedSolution, solver::MultistartAugmentedSolver)

Round the subsolver's solution to the nearest vertex, evaluate the cut norm value it
attains, and update `sol` if it beats the incumbent. The augmented indicators are
reduced to the original dimensions with [`CutNorm.fill_ST!`](@ref). Also appends to
`sol.all_solutions` when `save_all_solutions` is set. Called once per restart by
[`solve!`](@ref).
"""
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

"""
    fill_ST!(S, T, x)

Extract the indicator vectors of the original matrix from a solution `x` of the
augmented model, where `length(x) == length(S) + length(T) + 2`.

The entries `x[m+1]` and `x[m+n+2]` are the pivot bits belonging to the augmented row
and column. XOR-ing the remaining entries with them (here `abs(x[i] - f)`, since the
entries are `0` or `1`) normalizes the solution against the global bit-flip symmetry
of the augmented problem, so that `S` and `T` describe the same cut in the original
matrix. Throws a `DimensionMismatch` if the lengths do not fit.
"""
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