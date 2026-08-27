function print_augmented_iter_header(io)
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  Restart      Best Value      Improv      Sub Iter      Sub Obj      Time (s)")
    println(io, "--------------------------------------------------------------------------------")
end

function print_augmented_iter(io::IO, restart, value, improvements, sub_iter, sub_obj, elapsed, improved)
    @printf(io, "  %7d     %11.4e      %5d     %8d      %10.4e   %8.2f  %s\n",
        restart, value, improvements, sub_iter, sub_obj, elapsed, improved ? "*" : "")
end

function print_augmented_footer(io::IO, sol)
    println(io, "--------------------------------------------------------------------------------")
    @printf(io, "  Terminated:        %s\n", sol.termination_status)
    @printf(io, "  Cut norm:          %.5f\n", sol.value)
    @printf(io, "  Best restart:      %d / %d\n", sol.best_restart, sol.restarts)
    @printf(io, "  Improvements:      %d\n", sol.improvements)
    @printf(io, "  Runtime:           %.2fs\n", sol.runtime)
    println(io, "================================================================================")
end