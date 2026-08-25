"""
    should_print(restart::Int) -> Bool

Decide whether iteration `restart` is one of the logarithmically spaced iterations
printed at `print_level = 2`: the first five, then every 10th up to 100, every 100th
up to 1000, and so on.
"""
function should_print(restart::Int)
    restart <= 5 && return true
    magnitude = 10^floor(Int, log10(restart))
    return restart % magnitude == 0
end

function print_header(io::IO)
    println(io, "================================================================================")
    println(io, "                               CutNorm.jl - v", pkgversion(CutNorm))
    println(io, "                                (c) Martin Köhler")
    println(io, "                               TU Braunschweig 2026")
    println(io, "================================================================================")
end

function print_header_info(io::IO, m, n, method_name, description, subsolver_name, settings)
    @printf(io, "  Problem size:      %d x %d\n", m, n)
    @printf(io, "  Method:            %s\n", method_name)
    @printf(io, "  Description:       %s\n", description)
    @printf(io, "  Subsolver:         %s\n", subsolver_name)
    @printf(io, "  Max restarts:      %d\n", settings.max_restarts)
    @printf(io, "  Max time:          %.1fs\n", settings.max_time)
end