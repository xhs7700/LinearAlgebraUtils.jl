module LinearAlgebraUtils

using LinearAlgebra
using SparseArrays
using Statistics

export dspl, descrip
export getratio, getratios, writeratio
export getdiff, getdiffs, writediff
export subindices, submat, subvec, subcolmat, subrowmat, submat
export lap_pinv
export unitvec

dspl(x) = Base.Multimedia.display(x), println()

descrip(v::VecOrMat) = (min=reduce(min, v; init=Inf), max=reduce(max, v; init=0), mean=mean(v), median=median(v))

getratio(std::T1, test::T2) where {T1<:Real,T2<:Real} = abs(std - test) / std
getratios(std::VecOrMat{T1}, test::VecOrMat{T2}) where {T1<:Real,T2<:Real} = getratio.(std, test)
writeratio(path::AbstractString, std::VecOrMat{T1}, test::VecOrMat{T2}) where {T1<:Real,T2<:Real} = writeinfo(path, std, test, getratios(std, test), "ratio")

getdiff(std::T1, test::T2) where {T1<:Real,T2<:Real} = abs(std - test)
getdiffs(std::VecOrMat{T1}, test::VecOrMat{T2}) where {T1<:Real,T2<:Real} = getdiff.(std, test)
writediff(path::AbstractString, std::VecOrMat{T1}, test::VecOrMat{T2}) where {T1<:Real,T2<:Real} = writeinfo(path, std, test, getdiffs(std, test), "diff")

function writeinfo(path::AbstractString, std::VecOrMat{T1}, test::VecOrMat{T2}, info::VecOrMat, info_name::AbstractString) where {T1<:Real,T2<:Real}
    N = length(info)
    open(path, "w") do io
        write(io, "i, std, test, $info_name\n")
        for i in 1:N
            write(io, "$i, $(std[i]), $(test[i]), $(info[i])\n")
        end
    end
    return descrip(info)
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

unitvec(N::Int, x::Int) = unitvec(Float64, N, x)

function unitvec(::Type{T}, N::Int, x::Int) where {T}
    a = zeros(T, N)
    a[x] = one(T)
    return a
end

end # module LinearAlgebraUtils
