```@meta
CurrentModule = CutNorm
```

# Internals

Everything below is internal: not exported, and free to change without a breaking
release. It is documented because it is useful when working on the package itself, or
when reusing a piece of it — see [Developer docs](91-developer.md) and
[Advanced usage](50-advanced.md).

The public API lives in [Reference](95-reference.md).

## Index

```@index
Pages = ["96-internals.md"]
```

## Docstrings

```@autodocs
Modules = [CutNorm]
Public = false
Filter = t -> t !== CutNorm.AbstractCutNormMethod
```
