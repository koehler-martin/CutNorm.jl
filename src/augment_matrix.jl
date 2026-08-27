"""
    augment_matrix(A::AbstractMatrix) -> Matrix

Construct the `(m+1) × (n+1)` augmented matrix whose rows and columns each sum to zero.
Given an `m × n` matrix `A`, the result is:

    [ A        -A*1 ]
    [ -1'A    1'A*1 ]

This canonicalizes the {0,1} bit-flip symmetry in the cut norm relaxation.
"""
function augment_matrix(A::AbstractMatrix{T}) where {T<:Real}
    m, n = size(A)
    A_aug = zeros(T, m + 1, n + 1)
    @inbounds begin
        total = zero(T)
        for j in 1:n
            col_sum = zero(T)
            for i in 1:m
                val = A[i, j]
                A_aug[i, j] = val
                col_sum += val
            end
            A_aug[m+1, j] = -col_sum
            total += col_sum
        end
        for i in 1:m
            A_aug[i, n+1] = -sum(A_aug[i, j] for j in 1:n)
        end
        A_aug[m+1, n+1] = total
    end
    return A_aug
end

#=
function augment_matrix(A::AbstractMatrix)
    last_col = -sum(A, dims=2)
    A_aug = hcat(A, last_col)
    last_row = -sum(A_aug, dims=1)
    return vcat(A_aug, last_row)
end
=#