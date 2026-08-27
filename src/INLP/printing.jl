function print_inlp_header_info(io::IO, m, n, optimizer, settings)
    @printf(io, "  Problem size:      %d x %d\n", m, n)
    @printf(io, "  Method:            Integer Nonlinear Program (INLP)\n")
    @printf(io, "  Description:       Exact solution via JuMP\n")
    @printf(io, "  Optimizer:         %s\n", optimizer)
    @printf(io, "  Max time:          %.1fs\n", settings.max_time)
end

function print_inlp_footer(io::IO, sol)
    println(io, "--------------------------------------------------------------------------------")
    @printf(io, "  Terminated:        %s\n", sol.termination_status)
    @printf(io, "  Cut norm:          %.5f\n", sol.value)
    @printf(io, "  Solver time:       %.2fs\n", sol.solve_time)
    @printf(io, "  Runtime:           %.2fs\n", sol.runtime)
    println(io, "================================================================================")
end
