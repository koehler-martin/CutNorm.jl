```@meta
CurrentModule = CutNorm
```

# [Reference](@id reference)

The public API of CutNorm.jl, followed by the internals. Everything here is generated
from the docstrings in the source.

## Contents

```@contents
Pages = ["95-reference.md"]
Depth = 3
```

## Index

```@index
Pages = ["95-reference.md"]
```

## Public API

### Module

```@docs
CutNorm
```

### Entry point

```@docs
cutnorm
```

### Methods

```@docs
AbstractCutNormMethod
MultistartSigned
MultistartAugmented
BruteForce
INLP
ILP
QUBO
```

### Subsolvers

```@docs
AlternatingLinearSearch
GreedySolver
```

`TronSolver` from
[JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) is re-exported
as a third subsolver; see its own documentation.

### Solvers

```@docs
MultistartSignedSolver
MultistartAugmentedSolver
BruteForceSolver
INLPSolver
ILPSolver
QUBOSolver
```

### Settings

```@docs
MultistartSettings
BruteForceSettings
INLPSettings
ILPSettings
QUBOSettings
```

### Solutions

```@docs
MultistartSignedSolution
MultistartAugmentedSolution
BruteForceSolution
INLPSolution
ILPSolution
QUBOSolution
```

### Models

```@docs
BilinearModel
SignedBilinearModel
grad_s!
grad_t!
```

### Solving

```@docs
solve!
```

The internals are documented separately, in [Internals](96-internals.md).
