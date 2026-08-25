mutable struct BruteForceSolver{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractSolver{T}
    A::M
    s::Vector{T}
    t::Vector{T}
    At::Vector{T}
    settings::BruteForceSettings
end

function BruteForceSolver(A::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(A)
    settings = BruteForceSettings()
    populate!(settings; kwargs...)
    return BruteForceSolver(A, Vector{T}(undef, m), Vector{T}(undef, n), Vector{T}(undef, m), settings)
end

function solve!(solver::BruteForceSolver{T}; kwargs...) where {T<:AbstractFloat}
    m, n = size(solver.A)
    sol = BruteForceSolution(T, m, n)
    solve!(solver, sol; kwargs...)
end

function solve!(solver::BruteForceSolver{T}, sol::BruteForceSolution{T}; kwargs...) where {T}
    t0 = time_ns()
    reset!(sol)
    settings = solver.settings
    populate!(settings; kwargs...)

    A = solver.A
    m, n = sol.dims

    io = stdout
    pl = settings.print_level
    (pl >= 1) && (print_header(io); print_bruteforce_header_info(io, m, n, settings); print_bruteforce_iter_header(io))

    s = solver.s
    t = solver.t
    At = solver.At

    prev_value = sol.value
    last_obj = zero(T)

    for kt in 0:(2^n-1)
        fill_binary_vector!(t, kt)
        mul!(At, A, t)

        for ks in 0:(2^m-1)
            fill_binary_vector!(s, ks)

            value = dot(s, At)
            abs_value = abs(value)

            settings.scaled && (abs_value /= (m * n))

            if abs_value > sol.value
                sol.value = abs_value
                sol.improvements += 1
                sol.best_iteration = sol.iterations + 1
                copyto!(sol.S, s)
                copyto!(sol.T, t)
            end

            sol.iterations += 1

            sol.runtime = (time_ns() - t0) / 1e9

            improved = sol.value > prev_value
            if (pl >= 3) || (pl >= 2 && (improved || should_print(sol.iterations))) || (pl == 1 && improved)
                print_bruteforce_row(io, sol.iterations, sol.improvements, abs_value, sol.value, sol.runtime, improved)
            end
            prev_value = sol.value

            if sol.runtime > settings.max_time
                sol.termination_status = :max_time
                @goto done
            end
        end
    end

    sol.termination_status = :optimal

    @label done
    sol.runtime = (time_ns() - t0) / 1e9

    (pl >= 1) && print_bruteforce_footer(io, sol)

    return sol
end

function fill_binary_vector!(v::AbstractVector, k::Integer)
    @inbounds for i in eachindex(v)
        v[i] = (k >> (i - 1)) & 1
    end
end
