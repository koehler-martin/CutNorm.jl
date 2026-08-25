@testset "augment_matrix" begin
     # Basic 2x2 matrix
     A = [1.0 2.0;
          3.0 4.0]
     A_aug = CutNorm.augment_matrix(A)
     @test size(A_aug) == (3, 3)
     @test A_aug[1:2, 1:2] == A
     @test A_aug[1, 3] == -(1.0 + 2.0)
     @test A_aug[2, 3] == -(3.0 + 4.0)
     @test A_aug[3, 1] == -(1.0 + 3.0)   # -sum of column 1 of A_aug
     @test A_aug[3, 2] == -(2.0 + 4.0)    # -sum of column 2 of A_aug
     @test A_aug[3, 3] == 1.0 + 2.0 + 3.0 + 4.0
     # Every row and column of A_aug sums to zero
     @test all(sum(A_aug, dims=1) .≈ 0)
     @test all(sum(A_aug, dims=2) .≈ 0)

     # Rectangular matrix
     A = [1.0 2.0 3.0;
          4.0 5.0 6.0]
     A_aug = CutNorm.augment_matrix(A)
     @test size(A_aug) == (3, 4)
     @test A_aug[1:2, 1:3] == A
     @test all(sum(A_aug, dims=1) .≈ 0)
     @test all(sum(A_aug, dims=2) .≈ 0)

     # 1x1 matrix
     A = fill(5.0, 1, 1)
     A_aug = CutNorm.augment_matrix(A)
     @test size(A_aug) == (2, 2)
     @test A_aug[1, 1] == 5.0
     @test all(sum(A_aug, dims=1) .≈ 0)
     @test all(sum(A_aug, dims=2) .≈ 0)

     # Zero matrix
     A = zeros(3, 2)
     A_aug = CutNorm.augment_matrix(A)
     @test size(A_aug) == (4, 3)
     @test all(A_aug .== 0)

     # Negative entries
     A = [-1.0 -2.0;
          -3.0 -4.0]
     A_aug = CutNorm.augment_matrix(A)
     @test size(A_aug) == (3, 3)
     @test A_aug[1:2, 1:2] == A
     @test all(sum(A_aug, dims=1) .≈ 0)
     @test all(sum(A_aug, dims=2) .≈ 0)

     # Integer input
     A = [1 2; 3 4]
     A_aug = CutNorm.augment_matrix(A)
     @test size(A_aug) == (3, 3)
     @test all(sum(A_aug, dims=1) .== 0)
     @test all(sum(A_aug, dims=2) .== 0)
end
