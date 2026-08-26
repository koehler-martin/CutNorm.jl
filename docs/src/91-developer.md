# Developer docs

Notes for working on CutNorm.jl itself. The package was generated from
[BestieTemplate.jl](https://github.com/JuliaBesties/BestieTemplate.jl), so most of the
tooling follows that template's conventions.

## Layout

```text
src/
  CutNorm.jl                  # module: dependencies, exports, includes
  abstract_types.jl           # AbstractSolver / AbstractSolution / AbstractSettings, populate!
  augment_matrix.jl           # the zero-row/column-sum augmentation
  jump_status.jl              # MOI status -> Symbol
  methods.jl                  # cutnorm and the method selectors
  Model/                      # the bilinear NLPModels
  Multistart/                 # multistart loops, subsolvers, printing
    Subsolver/
    Augmented/
    Signed/
  BruteForce/                 # exact enumeration
  INLP/, ILP/, QUBO/          # the JuMP formulations
```

Each method directory follows the same four-file pattern: `settings.jl`, `solution.jl`,
`printing.jl`, `solver.jl`. Adding a method means adding such a directory, a selector
type in `methods.jl` with its `_cutnorm` method, the `include`s and exports in
`CutNorm.jl`, and a test file.

## Running the tests

```julia
using Pkg
Pkg.test("CutNorm")
```

or, from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test environment is a workspace project in `test/`, declared in the top-level
`Project.toml`. One test file per feature, all included from `test/runtests.jl`.

CI runs the tests on every push and pull request via `.github/workflows/Test.yml`, with
coverage uploaded to Codecov.

## Building the documentation

The documentation is built with [Documenter.jl](https://documenter.juliadocs.org/) from
the `docs/` environment, which uses the checked-out package through a `[sources]` entry —
no `Pkg.develop` needed.

Instantiate once:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

Then either build once into `docs/build/`:

```bash
julia --project=docs docs/make.jl
```

or serve the documentation with live reload while writing, which rebuilds on every save:

```bash
julia --project=docs -e 'using LiveServer; servedocs()'
```

### Adding a page

`docs/make.jl` walks `docs/src/` recursively and builds the page list itself, so a new
page only has to be a `.md` file in that directory. Pages are sorted by filename, hence
the numeric prefixes, and `index.md` is always placed first:

| Prefix           | Purpose                       |
|:-----------------|:------------------------------|
| `index.md`       | Landing page                  |
| `10-`            | Tutorial                      |
| `20-` … `50-`    | Guides                        |
| `91-developer.md`| This page                     |
| `95-reference.md`| Public API reference          |
| `96-internals.md`| Internal API reference        |

The navigation title of a page is its first heading. Two things need an entry in the
`titles` dictionary in `docs/make.jl`: a page whose nav entry should differ from its
heading, and every **subdirectory** — a folder without a title logs an error during the
build.

A new **internal** docstring needs nothing extra: `96-internals.md` collects everything
non-exported with `@autodocs`. A new **public** one has to be added to the matching
`@docs` block in `95-reference.md`, otherwise the build fails with "docstring not
included in the manual". That is deliberate: it keeps the reference grouped by topic
instead of alphabetical, at the cost of one line per addition.

Two things to watch out for on the reference pages:

- a docstring anchor and a heading slug must not collide, so avoid headings that are
  exactly a type or function name — write "### The QUBO formulation", not "### QUBO";
- `Filter` expressions in an `@autodocs` block are evaluated in `Main`, so names have to
  be qualified: `CutNorm.AbstractCutNormMethod`, not `AbstractCutNormMethod`.

### Doctests

Examples in docstrings and in the manual are checked. Use `jldoctest` blocks for
anything whose output is stable, and a plain `julia` block for output that is not —
timings, restart counts on unfixed budgets, and anything that needs an external
optimizer, which is not installed in the docs environment.

Run the doctests alone, the way CI does:

```bash
julia --project=docs -e 'using Documenter, CutNorm; DocMeta.setdocmeta!(CutNorm, :DocTestSetup, :(using CutNorm); recursive=true); doctest(CutNorm)'
```

`DocMeta.setdocmeta!` in `docs/make.jl` puts `using CutNorm` in front of every doctest,
so blocks do not need it; anything else — `using NLPModels`, for instance — has to be in
the block itself. Named blocks (`jldoctest name`) share state across the page, which the
manual uses to set up a matrix once and reuse it.

Because the starting points come from a Sobol sequence rather than an RNG, multistart
results are reproducible and can safely be doctested, as long as `max_restarts` is
fixed.

CI runs the doctests before deploying, in `.github/workflows/Docs.yml`.

### Deployment

`docs/make.jl` ends in `deploydocs`, which pushes the built site to the `gh-pages`
branch. Documenter authenticates that push with a `DOCUMENTER_KEY` secret if one is
set, and otherwise with the workflow's `GITHUB_TOKEN`. The token route is what this
repository uses, which is why the docs job in `.github/workflows/Docs.yml` declares

```yaml
permissions:
  contents: write
```

Without that grant the token is read-only and the push fails with
`Write access to repository not granted` and HTTP 403, *after* an otherwise successful
build — so a green build log up to `Deploying: ✔` followed by a 403 is a permissions
problem, not a docs problem.

Two things are configured outside the repository and cannot be fixed in a commit:

- **GitHub Pages** must be set to deploy from a branch, `gh-pages` at `/ (root)`, under
  Settings → Pages. `deploydocs` creates the branch on its first successful run.
- the repository's **default workflow permissions** (Settings → Actions → General) must
  not be restricted below what the job asks for.

The alternative to the token is a deploy key, which is what a fork without write access
to `gh-pages` needs. Generate the pair with

```julia
using DocumenterTools
DocumenterTools.genkeys(user = "koehler-martin", repo = "CutNorm.jl")
```

and follow its instructions: the private half goes into the repository's Actions secrets
as `DOCUMENTER_KEY`, the public half becomes a deploy key with write access. The
workflow already passes `DOCUMENTER_KEY` through, so nothing else has to change.

## Formatting and linting

The repository ships configuration for several tools. Note that **none of them is
currently run automatically**: there is no lint workflow in `.github/workflows/` and no
`.pre-commit-config.yaml`, so these files only take effect when your editor picks them
up or when you invoke the tool yourself.

| Tool                                                        | Config                | Scope             |
|:------------------------------------------------------------|:----------------------|:------------------|
| [JuliaFormatter](https://domluna.github.io/JuliaFormatter.jl/) | `.JuliaFormatter.toml` | Julia sources    |
| [markdownlint](https://github.com/DavidAnson/markdownlint)   | `.markdownlint.json`  | Markdown          |
| [yamllint](https://yamllint.readthedocs.io/) / yamlfmt       | `.yamllint.yml`, `.yamlfmt.yml` | Workflows |
| [lychee](https://lychee.cli.rs/)                             | `.lychee.toml`        | Links             |
| EditorConfig                                                 | `.editorconfig`       | Whitespace, LF endings |

Format the Julia sources with

```julia
using JuliaFormatter
format(".")
```

## Releasing

Version numbers live in `Project.toml`. TagBot (`.github/workflows/TagBot.yml`) creates
the GitHub release once a version is registered, and `deploydocs` picks the tag up to
publish a versioned copy of the documentation, which is what the `stable` documentation
badge points at.
