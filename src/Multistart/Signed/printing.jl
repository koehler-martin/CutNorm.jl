function print_signed_iter_header(io::IO)
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  Restart  Best Value  Improv  Iter(+)   Obj(+)    Iter(-)   Obj(-)   Time (s)")
    println(io, "--------------------------------------------------------------------------------")
end

function print_signed_row(io::IO, restart, value, improvements, sub_iter_pos, sub_obj_pos, sub_iter_neg, sub_obj_neg, elapsed, improved)
    @printf(io, "  %7d  %8.4e  %5d   %6d  %4.2e   %6d  %4.2e %8.2f  %s\n",
        restart, value, improvements, sub_iter_pos, sub_obj_pos, sub_iter_neg, sub_obj_neg, elapsed, improved ? "*" : "")
end

function print_signed_footer(io::IO, sol)
    println(io, "--------------------------------------------------------------------------------")
    @printf(io, "  Terminated:        %s\n", sol.termination_status)
    @printf(io, "  Cut norm:          %.5f\n", sol.value)
    @printf(io, "  Best sign:         (%s)\n", sol.sign > 0 ? "+" : "-")
    @printf(io, "  Best restart:      %d / %d\n", sol.best_restart, sol.restarts)
    @printf(io, "  Improvements:      %d\n", sol.improvements)
    @printf(io, "  Runtime:           %.2fs\n", sol.runtime)
    println(io, "================================================================================")
end