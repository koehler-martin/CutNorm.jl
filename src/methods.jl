"""
    AbstractCutNormMethod

Supertype of the method selectors accepted by the `method` keyword of
[`cutnorm`](@ref). A method value carries no problem data; it only picks the
algorithm and, for the JuMP-based methods, the optimizer to use.

| Method                        | Kind      | Notes                                     |
|:------------------------------|:----------|:------------------------------------------|
| [`MultistartSigned`](@ref)    | heuristic | default                                   |
| [`MultistartAugmented`](@ref) | heuristic |                                           |
| [`BruteForce`](@ref)          | exact     | exponential, no external solver needed    |
| [`INLP`](@ref)                | exact     | needs a global MINLP solver               |
| [`ILP`](@ref)                 | exact     | works with any MILP solver                |
| [`QUBO`](@ref)                | exact     | needs a nonconvex binary-quadratic solver |
"""
abstract type AbstractCutNormMethod end

"""
    MultistartAugmented{S}()

Multistart method on the augmented matrix, to be passed as `method` to
[`cutnorm`](@ref). `S` is the subsolver type: [`AlternatingLinearSearch`](@ref),
`TronSolver`, or [`GreedySolver`](@ref). The type parameter is mandatory.

The matrix is first replaced by the `(m+1) x (n+1)` matrix with vanishing row and
column sums (see [`CutNorm.augment_matrix`](@ref)). This removes the sign ambiguity
of the relaxation, so only **one** subproblem has to be solved per restart, at the
price of one extra row and column. The indicators returned in `S` and `T` are mapped
back to the original size.

# Examples

```julia
sol = cutnorm(A; method = MultistartAugmented{TronSolver}(), max_restarts = 500)
```

See also [`MultistartSigned`](@ref), [`MultistartAugmentedSolver`](@ref),
[`MultistartSettings`](@ref).
"""
struct MultistartAugmented{S} <: AbstractCutNormMethod end

"""
    MultistartSigned{S}()

Multistart method on the original matrix, solving both sign variants per restart.
This is the **default** method of [`cutnorm`](@ref). `S` is the subsolver type:
[`AlternatingLinearSearch`](@ref), `TronSolver`, or [`GreedySolver`](@ref). The type
parameter is mandatory.

Because the relaxation maximizes the absolute value of `s'At`, both the `+` and the
`-` objective are minimized in every restart -- two subproblems per restart -- and
the better of the two is kept. No augmentation is needed, and the sign that produced
the best value is reported in the `sign` field of the solution.

# Examples

```julia
sol = cutnorm(A)                                          # the default
sol = cutnorm(A; method = MultistartSigned{TronSolver}()) # trust-region Newton
```

See also [`MultistartAugmented`](@ref), [`MultistartSignedSolver`](@ref),
[`MultistartSettings`](@ref).
"""
struct MultistartSigned{S} <: AbstractCutNormMethod end

"""
    SDPRelaxation()

Placeholder for the semidefinite relaxation of Alon and Naor. **Not implemented
yet** -- `cutnorm(A; method = SDPRelaxation())` currently returns `nothing`.
"""
struct SDPRelaxation <: AbstractCutNormMethod end

"""
    BruteForce()

Exact method by exhaustive enumeration, to be passed as `method` to
[`cutnorm`](@ref).

All `2^m * 2^n` pairs of binary indicator vectors are evaluated, so the result is
the exact cut norm -- but the cost is exponential, which makes this practical for
small matrices only. No external solver is required. If the `max_time` limit is
reached first, the best value found so far is returned and `termination_status`
becomes `:max_time`.

# Examples

```julia
sol = cutnorm(A; method = BruteForce())
```

See also [`BruteForceSolver`](@ref), [`BruteForceSettings`](@ref).
"""
struct BruteForce <: AbstractCutNormMethod end

"""
    INLP(optimizer)

Exact method via an integer **nonlinear** program, to be passed as `method` to
[`cutnorm`](@ref). `optimizer` is any JuMP optimizer constructor, for example
`Gurobi.Optimizer`.

The model maximizes `(2σ - 1) * s'At` over binary `s`, `t` and a binary sign
variable `σ`, so the objective is nonconvex and nonlinear in the binaries: a global
MINLP solver is required -- HiGHS is *not* sufficient. The solver package is not a
dependency of CutNorm.jl; install and load it yourself.

# Examples

```julia
using Gurobi
sol = cutnorm(A; method = INLP(Gurobi.Optimizer), max_time = 60.0)
```

See also [`INLPSolver`](@ref), [`INLPSettings`](@ref), [`ILP`](@ref), [`QUBO`](@ref).
"""
struct INLP <: AbstractCutNormMethod
    optimizer
end

"""
    ILP(optimizer)

Exact method via an integer **linear** program, to be passed as `method` to
[`cutnorm`](@ref). `optimizer` is any JuMP optimizer constructor, for example
`HiGHS.Optimizer`.

The bilinear term is linearized with McCormick constraints -- exact for binary
variables -- and the absolute value is modeled with a binary sign variable and a
big-M reformulation, so the model is a plain MILP and any MILP solver can handle it.
The solver package is not a dependency of CutNorm.jl; install and load it yourself.

# Examples

```julia
using HiGHS
sol = cutnorm(A; method = ILP(HiGHS.Optimizer), max_time = 60.0)
```

See also [`ILPSolver`](@ref), [`ILPSettings`](@ref), [`CutNorm.build_ilp_model`](@ref).
"""
struct ILP <: AbstractCutNormMethod
    optimizer
end

"""
    QUBO(optimizer)

Exact method via a quadratic binary program on the augmented matrix, to be passed as
`method` to [`cutnorm`](@ref). `optimizer` is any JuMP optimizer constructor, for
example `Gurobi.Optimizer`.

The matrix is augmented to have vanishing row and column sums (see
[`CutNorm.augment_matrix`](@ref)), which removes the need for a sign variable, and
the resulting problem maximizes a quadratic form over binaries. The objective is a
nonconvex binary quadratic, so a solver supporting that problem class is required --
HiGHS is *not* sufficient. The solver package is not a dependency of CutNorm.jl;
install and load it yourself.

# Examples

```julia
using Gurobi
sol = cutnorm(A; method = QUBO(Gurobi.Optimizer))
```

See also [`QUBOSolver`](@ref), [`QUBOSettings`](@ref), [`CutNorm.build_qubo_model`](@ref).
"""
struct QUBO <: AbstractCutNormMethod
    optimizer
end

"""
    cutnorm(A::AbstractMatrix; method = MultistartSigned{AlternatingLinearSearch}(), kwargs...)

Compute the cut norm of the matrix `A`,

```math
\\|A\\|_{\\square} = \\max_{S \\subseteq [m],\\, T \\subseteq [n]}
                  \\left| \\sum_{i \\in S, j \\in T} A_{ij} \\right|,
```

and return a solution object holding the value and the maximizing row and column
sets.

This is the main entry point of the package. It builds the solver matching `method`,
runs it once, and returns its solution; to solve the same matrix repeatedly, build
the solver yourself and call [`solve!`](@ref) on it.

# Arguments

- `A`: the `m x n` input matrix. Its entries must be floating point — convert integer
  input with `float(A)` first.

# Keywords

- `method`: which algorithm to use, see [`AbstractCutNormMethod`](@ref). The default,
  [`MultistartSigned`](@ref) with [`AlternatingLinearSearch`](@ref), is a heuristic:
  it returns a lower bound on the cut norm together with the sets attaining it.
  [`BruteForce`](@ref), [`INLP`](@ref), [`ILP`](@ref) and [`QUBO`](@ref) are exact.
- all remaining keywords are forwarded to the settings object of the chosen method:
  [`MultistartSettings`](@ref), [`BruteForceSettings`](@ref), [`INLPSettings`](@ref),
  [`ILPSettings`](@ref) or [`QUBOSettings`](@ref). An unknown keyword raises an
  `ArgumentError` listing the accepted ones. The keywords common to all methods are
  `max_time`, `scaled` and `print_level`.

# Returns

A solution object whose type depends on `method` (see [`AbstractSolution`](@ref)),
with at least the fields

- `value`: the best cut norm value found, divided by `m * n` if `scaled = true`,
- `S`, `T`: the row and column indicator vectors, of length `m` and `n`, with entries
  in `{0, 1}` -- `S[i] == 1` means row `i` belongs to the optimal set,
- `dims`: the problem size `(m, n)`,
- `runtime`: elapsed wall-clock time in seconds,
- `termination_status`: why the solver stopped, e.g. `:optimal`, `:max_restarts` or
  `:max_time`.

# Examples

```jldoctest
julia> A = [ 1.0  1  1  1  1;
             1    1  1  1  1;
             1    1  1  1  1;
            -1   -1 -1 -1 -1;
            -1   -1 -1 -1 -1];

julia> sol = cutnorm(A; max_restarts = 20);

julia> sol.value
15.0

julia> sol.S
5-element Vector{Float64}:
 1.0
 1.0
 1.0
 0.0
 0.0

julia> sol.termination_status
:max_restarts
```

The exact and the heuristic methods agree on this small example:

```jldoctest
julia> A = Float64[1 -1; -1 1];

julia> cutnorm(A; method = BruteForce()).value
1.0

julia> cutnorm(A; method = MultistartAugmented{GreedySolver}(), max_restarts = 20).value
1.0
```

Scaling, time limits and verbosity are set through the same keyword mechanism:

```julia
sol = cutnorm(A; scaled = true, max_time = 10.0, print_level = 1)
```

# References

Frieze, A., Kannan, R. *Quick Approximation to Matrices and Applications*.
Combinatorica 19, 175-220 (1999). <https://doi.org/10.1007/s004930050052>

Alon, N., Naor, A. *Approximating the Cut-Norm via Grothendieck's Inequality*.
SIAM Journal on Computing 35(4), 787-803 (2006).
<https://doi.org/10.1137/S0097539704441629>
"""
function cutnorm(A::AbstractMatrix;
    method::AbstractCutNormMethod=MultistartSigned{AlternatingLinearSearch}(),
    kwargs...
)
    return _cutnorm(A, method; kwargs...)
end

"""
    _cutnorm(A, method; kwargs...)

Internal dispatch target of [`cutnorm`](@ref): builds the solver belonging to
`method`, forwarding `kwargs` to its settings, and calls [`solve!`](@ref) on it. One
method exists per subtype of [`AbstractCutNormMethod`](@ref).
"""
function _cutnorm(A, ::MultistartAugmented{S}; kwargs...) where S
    solver = MultistartAugmentedSolver(A, S; kwargs...)
    #solution = MultistartSignedSolution(A)
    return solve!(solver)
end

function _cutnorm(A, ::MultistartSigned{S}; kwargs...) where S
    solver = MultistartSignedSolver(A, S; kwargs...)
    return solve!(solver)
end

function _cutnorm(A, ::SDPRelaxation; kwargs...)
    # future SDP implementation
end

function _cutnorm(A, ::BruteForce; kwargs...)
    solver = BruteForceSolver(A; kwargs...)
    return solve!(solver)
end

function _cutnorm(A, method::INLP; kwargs...)
    solver = INLPSolver(A, method.optimizer; kwargs...)
    return solve!(solver)
end

function _cutnorm(A, method::ILP; kwargs...)
    solver = ILPSolver(A, method.optimizer; kwargs...)
    return solve!(solver)
end

function _cutnorm(A, method::QUBO; kwargs...)
    solver = QUBOSolver(A, method.optimizer; kwargs...)
    return solve!(solver)
end
