module LinearAlgebraUtils

using LinearAlgebra
using SparseArrays
using Statistics

export dspl, descrip, getratio, getratios, writeratio
export subindices, submat, subvec, subcolmat, subrowmat, submat
export lap_pinv

dspl(x) = Base.Multimedia.display(x), println()

descrip(v::VecOrMat) = (min=reduce(min, v; init=Inf), max=reduce(max, v; init=0), mean=mean(v), median=median(v))

getratio(std::T, test::T) where {T<:Real} = abs(std - test) / std

getratios(std::VecOrMat{T}, test::VecOrMat{T}) where {T<:Real} = getratio.(std, test)

function writeratio(path::AbstractString, std::VecOrMat{T}, test::VecOrMat{T}) where {T<:Real}
    ratios = getratios(std, test)
    N = length(ratios)
    open(path, "w") do io
        write(io, "i, std, test, ratio\n")
        for i in 1:N
            write(io, "$i, $(std[i]), $(test[i]), $(ratios[i])\n")
        end
    end
    return descrip(ratios)
end

function subindices(N::Int, S::Vector{Int})
    mask = trues(N)
    for x in S
        mask[x] = false
    end
    return mask
end

function subindices(N::Int, x::Int)
    mask = trues(N)
    mask[x] = false
    return mask
end

function subvec(v::AbstractVector, S::Union{Vector{Int},Int})
    N = length(v)
    mask = subindices(N, S)
    return v[mask]
end

function submat(L::AbstractMatrix, S::Union{Vector{Int},Int})
    N = size(L, 1)
    mask = subindices(N, S)
    return L[mask, mask]
end

function subrowmat(L::AbstractMatrix, S::Union{Vector{Int},Int})
    N = size(L, 1)
    mask = subindices(N, S)
    return L[mask, :]
end

function subcolmat(L::AbstractMatrix, S::Union{Vector{Int},Int})
    N = size(L, 2)
    mask = subindices(N, S)
    return L[:, mask]
end

function domindiag(L::AbstractMatrix)
    N = size(L, 1)
    return L * ones(eltype(L), N)
end

function lap_pinv(L::AbstractMatrix)
    N = size(L, 1)
    J = ones(eltype(L), N, N) / N
    return inv(Matrix(L) + J) - J
end

end # module LinearAlgebraUtils
