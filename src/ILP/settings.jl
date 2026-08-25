"""
    ILPSettings(; kwargs...)

Settings for [`ILPSolver`](@ref), the exact integer linear program formulation.

Every field can be set as a keyword argument, either when constructing the solver,
when calling [`cutnorm`](@ref), or on an existing settings object with
[`CutNorm.populate!`](@ref).

# Fields

| Field         | Type      | Default  | Description                          |
|:--------------|:----------|:---------|:-------------------------------------|
| `max_time`    | `Float64` | `3600.0` | Time limit in seconds, handed to the solver via `JuMP.set_time_limit_sec` |
| `scaled`      | `Bool`    | `false`  | Divide the returned value by `m * n` |
| `print_level` | `Int`     | `0`      | Verbosity, see below                 |

# Print levels

| Level | Output                                                              |
|:------|:--------------------------------------------------------------------|
| `0`   | Silent (default)                                                    |
| `1`   | CutNorm.jl header and summary; the underlying solver stays silent    |
| `≥ 2` | Header and summary **plus** the solver's own log                    |

Print level `0` and `1` call `JuMP.set_silent` on the model, level `≥ 2` calls
`JuMP.unset_silent`.

See also [`cutnorm`](@ref), [`ILPSolver`](@ref).
"""
Base.@kwdef mutable struct ILPSettings <: AbstractSettings
    max_time::Float64 = 3600.0
    scaled::Bool = false
    print_level::Int = 0
end
