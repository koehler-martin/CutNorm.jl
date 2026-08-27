function print_bruteforce_header_info(io::IO, m, n, settings)
    total = BigInt(2)^m * BigInt(2)^n
    @printf(io, "  Problem size:      %d x %d\n", m, n)
    @printf(io, "  Method:            Brute Force\n")
    @printf(io, "  Description:       Enumerates all 2^m * 2^n binary vector pairs\n")
    @printf(io, "  Total pairs:       %s\n", string(total))
    @printf(io, "  Max time:          %.1fs\n", settings.max_time)
end

function print_bruteforce_iter_header(io::IO)
    println(io, "--------------------------------------------------------------------------------")
    println(io, "   Iteration        Best Value      Improv         Objective          Time (s)")
    println(io, "--------------------------------------------------------------------------------")
end

function print_bruteforce_row(io::IO, iterations, improvements, obj, value, elapsed, improved)
    @printf(io, "  %10d        %10.4e      %6d        %10.4e       %10.2f  %s\n",
        iterations, value, improvements, obj, elapsed, improved ? "*" : "")
end

function print_bruteforce_footer(io::IO, sol)
    println(io, "--------------------------------------------------------------------------------")
    @printf(io, "  Terminated:        %s\n", sol.termination_status)
    @printf(io, "  Cut norm:          %.5f\n", sol.value)
    @printf(io, "  Best iteration:    %d / %d\n", sol.best_iteration, sol.iterations)
    @printf(io, "  Improvements:      %d\n", sol.improvements)
    @printf(io, "  Runtime:           %.2fs\n", sol.runtime)
    println(io, "================================================================================")
end
