```@meta
CurrentModule = CutNorm
```

# Home

The package [CutNorm.jl](https://github.com/koehler-martin/CutNorm.jl) implements several approaches to compute the **cut norm** of a matrix in the Julia language.

For a matrix ``A \in \mathbb{R}^{m \times n}`` the cut norm is

```math
\|A\|_{\square} = \max_{S \subseteq [m],\; T \subseteq [n]}
                  \left| \sum_{i \in S,\, j \in T} A_{ij} \right| .
```

Computing it is NP-hard, so CutNorm.jl offers two families of methods:

- **heuristic** — the combinatorial problem is relaxed to a bilinear program over ``[0,1]^m \times [0,1]^n`` and solved from many quasi-random (Sobol) starting points. Every local solution is a feasible assignment of rows and columns, so the result is always a lower bound attained by concrete sets ``S`` and ``T``.
- **exact** — by exhaustive enumeration, or by handing an integer or quadratic formulation to a solver of your choice through [JuMP](https://jump.dev/).

## Installation

CutNorm.jl can be installed via the following:
```julia
using Pkg
Pkg.add("CutNorm")
```

## Quick start

```jldoctest
julia> using CutNorm

julia> A = [ 1.0  1  1  1  1;
             1    1  1  1  1;
             1    1  1  1  1;
            -1   -1 -1 -1 -1;
            -1   -1 -1 -1 -1];

julia> sol = cutnorm(A);

julia> sol.value
15.0

julia> findall(isone, sol.S), findall(isone, sol.T)
([1, 2, 3], [1, 2, 3, 4, 5])
```

The value `15.0` is attained by the first three rows against all five columns, which
is what `sol.S` and `sol.T` report.

## Where to go next

| Page                                        | Contents                                                        |
|:--------------------------------------------|:----------------------------------------------------------------|
| [Getting started](10-getting-started.md)    | A guided tour: solving, reading the result, watching progress    |
| [Methods](20-methods.md)                    | The heuristic and exact methods, and the subsolvers              |
| [Options](30-options.md)                    | Every keyword argument, per method                               |
| [Solution objects](40-solutions.md)         | What comes back from a solve                                     |
| [Advanced usage](50-advanced.md)            | Reusing solvers, the JuMP models, the NLPModel interface         |
| [Reference](95-reference.md)                | The public API                                                   |
| [Internals](96-internals.md)                | Everything not exported                                          |

## Author and Acknowledgements

This package was written by
[Martin Köhler](https://www.tu-braunschweig.de/en/mo/team/koehler) at TU Braunschweig.

Funded by the European Union through ERC Consolidator Grant SCARCE
([101087662](https://cordis.europa.eu/project/id/101087662)). Views and opinions
expressed are however those of the author(s) only and do not necessarily reflect those
of the European Union or the European Research Council Executive Agency. Neither the
European Union nor the granting authority can be held responsible for them.

## References

Frieze, A., Kannan, R. *Quick Approximation to Matrices and Applications*.
Combinatorica 19, 175–220 (1999).
[doi:10.1007/s004930050052](https://doi.org/10.1007/s004930050052)

Alon, N., Naor, A. *Approximating the Cut-Norm via Grothendieck's Inequality*.
SIAM Journal on Computing 35(4), 787–803 (2006).
[doi:10.1137/S0097539704441629](https://doi.org/10.1137/S0097539704441629)
