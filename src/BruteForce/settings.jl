"""
    BruteForceSettings(; kwargs...)

Settings for [`BruteForceSolver`](@ref), the exact enumeration method.

Every field can be set as a keyword argument, either when constructing the solver,
when calling [`cutnorm`](@ref), or on an existing settings object with
[`CutNorm.populate!`](@ref).

# Fields

| Field         | Type      | Default  | Description                          |
|:--------------|:----------|:---------|:-------------------------------------|
| `max_time`    | `Float64` | `3600.0` | Wall-clock time limit in seconds     |
| `scaled`      | `Bool`    | `false`  | Divide the returned value by `m * n` |
| `print_level` | `Int`     | `0`      | Verbosity, see below                 |

Enumeration is exhaustive, so the result is the exact cut norm whenever the solver
runs to completion (`termination_status == :optimal`). If `max_time` is hit first,
the status is `:max_time` and the value is the best one found so far, i.e. a lower
bound.

# Print levels

| Level | Output                                                                    |
|:------|:--------------------------------------------------------------------------|
| `0`   | Silent (default)                                                          |
| `1`   | Header, footer, and a row whenever the best value improves                 |
| `2`   | As `1`, plus logarithmically spaced iterations (1–5, then 10, 100, 1000, …) |
| `3`   | A row for every enumerated pair                                            |

See also [`cutnorm`](@ref), [`MultistartSettings`](@ref).
"""
Base.@kwdef mutable struct BruteForceSettings <: AbstractSettings
    max_time::Float64 = 3600.0
    scaled::Bool = false
    print_level::Int = 0
end
