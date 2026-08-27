"""
    MultistartSettings(; kwargs...)

Settings for the multistart solvers [`MultistartAugmentedSolver`](@ref) and
[`MultistartSignedSolver`](@ref).

Every field can be set as a keyword argument, either when constructing the solver,
when calling [`cutnorm`](@ref), or on an existing settings object with
[`CutNorm.populate!`](@ref).

# Fields

| Field                | Type      | Default  | Description                                       |
|:---------------------|:----------|:---------|:--------------------------------------------------|
| `max_restarts`       | `Int`     | `1000`   | Maximum number of restarts                        |
| `max_time`           | `Float64` | `3600.0` | Wall-clock time limit in seconds                  |
| `scaled`             | `Bool`    | `false`  | Divide the returned value by `m * n`              |
| `save_all_solutions` | `Bool`    | `false`  | Keep the rounded solution of every restart        |
| `print_level`        | `Int`     | `0`      | Verbosity, see below                              |

Both stopping criteria are checked after every restart, so the solver stops at the
first of `max_restarts` and `max_time` that is reached; the corresponding
`termination_status` is `:max_restarts` or `:max_time`.

# Print levels

| Level | Output                                                                  |
|:------|:------------------------------------------------------------------------|
| `0`   | Silent (default)                                                        |
| `1`   | Header, footer, and a row whenever the best value improves               |
| `2`   | As `1`, plus logarithmically spaced restarts (1–5, then 10, 100, 1000, …) |
| `3`   | A row for every restart                                                 |
| `4`   | As `3`, plus the log of the subsolver                                    |

# Examples

```jldoctest
julia> settings = MultistartSettings(; max_restarts = 50, scaled = true)
MultistartSettings(50, 3600.0, true, false, 0)

julia> settings.max_restarts
50
```

See also [`cutnorm`](@ref), [`BruteForceSettings`](@ref).
"""
Base.@kwdef mutable struct MultistartSettings <: AbstractSettings
    max_restarts::Int = 1000
    max_time::Float64 = 3600.0
    scaled::Bool = false
    save_all_solutions::Bool = false
    print_level::Int = 0
end
