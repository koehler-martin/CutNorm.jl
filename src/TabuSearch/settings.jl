"""
    TabuSearchSettings(; kwargs...)

Settings for [`TabuSearchSolver`](@ref), the two-sided flip tabu search.

Every field can be set as a keyword argument, either when constructing the solver,
when calling [`cutnorm`](@ref), or on an existing settings object with
[`CutNorm.populate!`](@ref).

# Fields

With `N = m + n` the number of bits of the search:

| Field               | Type                          | Default              | Description                                                        |
|:--------------------|:------------------------------|:---------------------|:-------------------------------------------------------------------|
| `max_iter`          | `Int`                         | `typemax(Int)`       | Iteration limit of each search                                     |
| `max_time`          | `Float64`                     | `10.0`               | Wall-clock time limit in seconds                                   |
| `target`            | `Float64`                     | `Inf`                | Stop all searches once one of them reaches this value              |
| `tenure`            | `Int`                         | `0`                  | Base tabu tenure `θ`; `0` picks `max(1, round(0.005N))`             |
| `tenure_rand`       | `Int`                         | `10`                 | A flipped bit stays tabu for `θ + rand(0:tenure_rand)` iterations   |
| `stall`             | `Int`                         | `0`                  | Iterations without improvement before diversifying; `0` picks `max(100, 5N)` |
| `perturb_frac`      | `Float64`                     | `0.1`                | Fraction of the `N` bits flipped when the elite is perturbed        |
| `max_perturbations` | `Int`                         | `3`                  | Failed perturbations in a row before a random restart with the opposite sign |
| `refresh`           | `Int`                         | `0`                  | Recompute `A*y` and `A'*x` every `refresh` iterations; `0` picks `max(1000, 10N)` |
| `ntasks`            | `Int`                         | `Threads.nthreads()` | Number of independent searches, run in parallel                    |
| `seed`              | `UInt64`                      | random               | Search `k` is seeded with `seed + k`                               |
| `S0`, `T0`          | `Union{Nothing,Vector{Bool}}` | `nothing`            | Initial row and column indicators; `nothing` starts at random       |
| `scaled`            | `Bool`                        | `false`              | Divide the returned value by `m * n`                               |
| `print_level`       | `Int`                         | `0`                  | Verbosity, see below                                               |

The tabu search has no natural end, so it always runs until a limit is hit: each
search stops after `max_iter` iterations or `max_time` seconds, and all of them stop
as soon as one reaches `target`. The corresponding `termination_status` is `:target`,
`:max_iter` or `:max_time`. `target` is compared with the value in the units it is
returned in, i.e. divided by `m * n` if `scaled = true`. Because there is no
iteration limit by default, `max_time` defaults to `10.0` rather than the `3600.0` of
the other methods.

`tenure_rand` is clamped to `0:N÷4` and the number of perturbed bits to `1:N`.

# Parallel searches

The `ntasks` searches run as tasks on Julia's threads, so start Julia with several
threads (e.g. `julia -t auto`) to profit from them. They start with alternating
signs, `+1` for odd and `-1` for even searches, so that `A` and `-A` are both searched
from the beginning. Keep `ntasks ≤ Threads.nthreads()`: a search never yields, so
surplus tasks only start once others have finished, which is after the time limit.

Every [`solve!`](@ref) reseeds the searches from `seed`, so repeated solves with the
same `seed` and `max_iter` as the only binding limit return the same result. Only the
order of the log rows and the `improvements` counter depend on the timing of the
threads. The default `seed` is drawn once, when the settings are created.

# Initial guess

`S0` and `T0` accept any vector with entries in `{0, 1}`, for example `sol.S` and
`sol.T` of another solver, and are converted to `Vector{Bool}`. Every search starts
from them. If both are given, the odd searches start with the sign of the sum over
`S0 × T0` and the even ones with the opposite sign. Set them back to `nothing` to
start at random again.

# Print levels

The log is printed while the searches run, one row per *phase*: the iterations of a
search from its start, a perturbation or a restart up to the next one, or until it
stops. A phase is the counterpart of one restart of the multistart methods. The row is
printed by the search at the end of the phase, so rows of one search are at least
`stall` iterations apart.

| Level | Output                                                                     |
|:------|:---------------------------------------------------------------------------|
| `0`   | Silent (default)                                                           |
| `1`   | Header, footer, and a row whenever a phase improves the best value          |
| `2`   | As `1`, plus logarithmically spaced phases of every search (1–5, then 10, 100, 1000, …) |
| `3`   | A row for every phase                                                      |

The columns are the search (`Thread`, from `1` to `ntasks`), its phase number, the best
value over all searches and how often it improved, the iterations of the phase, and
`Obj`, the sum `s'A t` at the best point of the phase, whose sign is the sign the phase
searched with. A `*` marks the phases that improved the best value. After all searches
have stopped, a summary row per search and the footer follow.

See also [`cutnorm`](@ref), [`MultistartSettings`](@ref).
"""
Base.@kwdef mutable struct TabuSearchSettings <: AbstractSettings
    max_iter::Int = typemax(Int)
    max_time::Float64 = 10.0
    target::Float64 = Inf
    tenure::Int = 0
    tenure_rand::Int = 10
    stall::Int = 0
    perturb_frac::Float64 = 0.1
    max_perturbations::Int = 3
    refresh::Int = 0
    ntasks::Int = Threads.nthreads()
    seed::UInt64 = rand(UInt64)
    S0::Union{Nothing,Vector{Bool}} = nothing
    T0::Union{Nothing,Vector{Bool}} = nothing
    scaled::Bool = false
    print_level::Int = 0
end
