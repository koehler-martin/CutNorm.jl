mutable struct GreedySolver{
    T<:AbstractFloat,
    V<:AbstractVector{T},
} <: AbstractOptimizationSolver
    x::V    # x = [s; t]
    g::V    # full gradient g = [A*t; A'*s]
end

function GreedySolver(model::AbstractBilinearModel{T,V}) where {T<:AbstractFloat,V<:AbstractVector{T}}
    m, n = model.data.dims
    x = Vector{T}(undef, m + n)
    g = Vector{T}(undef, m + n)

    return GreedySolver{T,V}(x, g)
end

function SolverCore.reset!(solver::GreedySolver)
    fill!(solver.x, 0)
    fill!(solver.g, 0)
end

function SolverCore.solve!(solver::GreedySolver{T,V},
    model::AbstractBilinearModel{T,V},
    stats::GenericExecutionStats{T,V};
    x::V=model.meta.x0,
    max_iter::Int=typemax(Int),
    max_time::Float64=30.0,
    verbose::Int=0,
) where {T<:AbstractFloat,V<:AbstractVector{T}}
    if !(model.meta.minimize)
        error("GreedySolver only works for minimization problems")
    end
    if !(unconstrained(model) || bound_constrained(model))
        error("GreedySolver should only be called for unconstrained or bound-constrained problems")
    end

    SolverCore.reset!(stats)
    t0 = time_ns()
    set_time!(stats, 0.0)

    ℓ = model.meta.lvar
    u = model.meta.uvar
    m, n = model.data.dims
    A = model.data.A

    solver.x .= x
    x = solver.x
    g = solver.g

    # Round initial point to nearest vertex
    @. x = clamp(x, ℓ, u)
    for i in eachindex(x)
        x[i] = x[i] < (ℓ[i] + u[i]) / 2 ? ℓ[i] : u[i]
    end

    # Initial gradient computation
    grad!(model, x, g)

    set_iter!(stats, 0)
    set_objective!(stats, obj(model, x))
    set_solution!(stats, x)

    verbose > 0 && @info log_header(
        [:iter, :f],
        [Int, T],
        hdr_override=Dict(:f => "obj"),
    )
    verbose > 0 && @info log_row(Any[stats.iter, stats.objective])

    set_status!(
        stats,
        get_status(
            model,
            elapsed_time=stats.elapsed_time,
            optimal=false,
            iter=stats.iter,
            max_iter=max_iter,
            max_time=max_time,
        ),
    )

    done = stats.status != :unknown

    while !done
        # Find the best single flip (allocation-free scan)
        best_delta = zero(T)
        best_idx = 0

        for i in eachindex(x)
            Δ = x[i] == ℓ[i] ? u[i] - ℓ[i] : ℓ[i] - u[i]
            delta = Δ * g[i]
            if delta < best_delta
                best_delta = delta
                best_idx = i
            end
        end

        optimal = best_delta >= zero(T)

        if !optimal
            # Apply the best flip and update gradient incrementally
            k = best_idx
            Δ = x[k] == ℓ[k] ? u[k] - ℓ[k] : ℓ[k] - u[k]
            x[k] = x[k] == ℓ[k] ? u[k] : ℓ[k]
            σ = sign(model)
            if k <= m
                # Flipped s_k: update the t-part of the gradient
                @views g[m+1:m+n] .+= (σ * Δ) .* A[k, :]
            else
                # Flipped t_j: update the s-part of the gradient
                j = k - m
                @views g[1:m] .+= (σ * Δ) .* A[:, j]
            end
        end

        set_iter!(stats, stats.iter + 1)

        elapsed = (time_ns() - t0) / 1e9
        set_time!(stats, elapsed)

        set_status!(
            stats,
            get_status(
                model,
                elapsed_time=stats.elapsed_time,
                optimal=optimal,
                iter=stats.iter,
                max_iter=max_iter,
                max_time=max_time,
            ),
        )

        (verbose > 0) &&
            (mod(stats.iter, verbose) == 0) &&
            @info log_row(Any[stats.iter, obj(model, x)])

        (stats.status != :unknown) && (done = true)
    end
    set_solution!(stats, x)
    set_objective!(stats, obj(model, x))
    return stats
end
