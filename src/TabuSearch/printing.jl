function print_tabu_header_info(io::IO, m, n, ntasks, params, settings)
    max_iter = settings.max_iter == typemax(Int) ? "unlimited" : string(settings.max_iter)
    @printf(io, "  Problem size:      %d x %d\n", m, n)
    @printf(io, "  Method:            Tabu Search\n")
    @printf(io, "  Description:       Adds or removes one row or column per iteration\n")
    @printf(io, "  Searches:          %d on %d threads\n", ntasks, Threads.nthreads())
    @printf(io, "  Tabu tenure:       %d + rand(0:%d)\n", params.tenure, params.tenure_rand)
    @printf(io, "  Stall:             %d\n", params.stall)
    @printf(io, "  Perturbed bits:    %d\n", params.nflips)
    @printf(io, "  Max iterations:    %s per search\n", max_iter)
    @printf(io, "  Max time:          %.1fs\n", settings.max_time)
    isfinite(settings.target) && @printf(io, "  Target:            %.5f\n", settings.target)
end

function print_tabu_iter_header(io::IO)
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  Thread    Phase    Best Value   Improv       Iter           Obj   Time (s)")
    println(io, "--------------------------------------------------------------------------------")
end

function print_tabu_row(io::IO, thread, phase, value, improvements, iterations, obj, elapsed, improved)
    @printf(io, "  %6d  %7d   %11.4e   %6d   %8d   %11.4e   %8.2f  %s\n",
        thread, phase, value, improvements, iterations, obj, elapsed, improved ? "*" : "")
end

function print_tabu_summary_header(io::IO)
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  Thread    Iterations   Perturb  Restarts  Sign      Best Value   Found (s)")
    println(io, "--------------------------------------------------------------------------------")
end

function print_tabu_summary_row(io::IO, task, iterations, perturbations, restarts, sign, best, time_to_best)
    @printf(io, "  %6d  %12d  %8d  %8d  %4s  %14.6e  %10.2f\n",
        task, iterations, perturbations, restarts, sign > 0 ? "+" : "-", best, time_to_best)
end

function print_tabu_footer(io::IO, sol, ntasks)
    println(io, "--------------------------------------------------------------------------------")
    @printf(io, "  Terminated:        %s\n", sol.termination_status)
    @printf(io, "  Cut norm:          %.5f\n", sol.value)
    @printf(io, "  Best sign:         (%s)\n", sol.sign > 0 ? "+" : "-")
    @printf(io, "  Best thread:       %d / %d\n", sol.best_task, ntasks)
    @printf(io, "  Best iteration:    %d\n", sol.best_iteration)
    @printf(io, "  Improvements:      %d\n", sol.improvements)
    @printf(io, "  Time to best:      %.2fs\n", sol.time_to_best)
    @printf(io, "  Iterations:        %d\n", sol.iterations)
    @printf(io, "  Perturbations:     %d\n", sol.perturbations)
    @printf(io, "  Restarts:          %d\n", sol.restarts)
    @printf(io, "  Runtime:           %.2fs\n", sol.runtime)
    println(io, "================================================================================")
end
