"""
    CutNorm

Compute the cut norm of a matrix,

```math
\\|A\\|_{\\square} = \\max_{S \\subseteq [m],\\, T \\subseteq [n]}
                  \\left| \\sum_{i \\in S, j \\in T} A_{ij} \\right|,
```

either heuristically, by multistart nonlinear optimization of the bilinear relaxation
over `[0,1]^m × [0,1]^n`, or exactly, by enumeration or by an integer/quadratic
program solved through [JuMP](https://jump.dev/).

The entry point is [`cutnorm`](@ref); the method is chosen with its `method` keyword,
and all further keywords configure the settings object of that method:

```julia
using CutNorm

sol = cutnorm(A)                                    # default heuristic
sol = cutnorm(A; method = BruteForce())             # exact, small matrices
sol = cutnorm(A; method = ILP(HiGHS.Optimizer))     # exact, via a MILP solver
```

For repeated solves of the same matrix, build a solver — [`MultistartSignedSolver`](@ref),
[`MultistartAugmentedSolver`](@ref), [`BruteForceSolver`](@ref), [`INLPSolver`](@ref),
[`ILPSolver`](@ref), [`QUBOSolver`](@ref) — and call [`solve!`](@ref) on it.

See the documentation at <https://koehler-martin.github.io/CutNorm.jl> for details.
"""
module CutNorm

# Standard library
using LinearAlgebra: mul!, dot, rmul!
using Printf: @printf

# External packages
import JuMP
import NLPModels
using NLPModels: AbstractNLPModel, NLPModelMeta, Counters, @lencheck, obj, grad!,
    unconstrained, bound_constrained
using JSOSolvers: TronSolver
using Sobol: AbstractSobolSeq, SobolSeq, next!, skip

import SolverCore
import SolverCore: reset!, solve!
using SolverCore: GenericExecutionStats, AbstractOptimizationSolver, AbstractExecutionStats,
    set_time!, set_iter!, set_objective!, set_solution!, set_status!, get_status,
    log_header, log_row

# Exports
# Model types and helpers
export BilinearModel, SignedBilinearModel
export grad_s!, grad_t!

# Subsolvers
export TronSolver
export AlternatingLinearSearch
export GreedySolver

# Multistart
export MultistartSettings
export MultistartAugmentedSolver, MultistartAugmentedSolution
export MultistartSignedSolver, MultistartSignedSolution

# BruteForce
export BruteForceSettings
export BruteForceSolver, BruteForceSolution

# INLP
export INLPSettings
export INLPSolver, INLPSolution

# ILP
export ILPSettings
export ILPSolver, ILPSolution

# QUBO
export QUBOSettings
export QUBOSolver, QUBOSolution

# Top-level API
export solve!
export cutnorm, MultistartAugmented, MultistartSigned, BruteForce, INLP, ILP, QUBO

# Source files
include("abstract_types.jl")
include("augment_matrix.jl")

include("Model/types.jl")
include("Model/nlp_interface.jl")

include("Multistart/Subsolver/alternating_linear_search.jl")
include("Multistart/Subsolver/greedy_solver.jl")
include("Multistart/Subsolver/get_subsolver.jl")

include("Multistart/printing.jl")
include("Multistart/settings.jl")

include("Multistart/Augmented/solution.jl")
include("Multistart/Augmented/printing.jl")
include("Multistart/Augmented/solver.jl")

include("Multistart/Signed/solution.jl")
include("Multistart/Signed/printing.jl")
include("Multistart/Signed/solver.jl")

include("BruteForce/settings.jl")
include("BruteForce/solution.jl")
include("BruteForce/printing.jl")
include("BruteForce/solver.jl")

include("jump_status.jl")

include("INLP/settings.jl")
include("INLP/solution.jl")
include("INLP/printing.jl")
include("INLP/solver.jl")

include("ILP/settings.jl")
include("ILP/solution.jl")
include("ILP/printing.jl")
include("ILP/solver.jl")

include("QUBO/settings.jl")
include("QUBO/solution.jl")
include("QUBO/printing.jl")
include("QUBO/solver.jl")

include("methods.jl")

end
