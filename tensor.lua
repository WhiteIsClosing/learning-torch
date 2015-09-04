----------------------------------------------------
-- 0. LongStorage
----------------------------------------------------
-- A Tensor is a potentially multi-dimensional matrix.
-- The number of dimensions is unlimited that can be created using LongStorage with more dimensions.

--- creation of a 4D-tensor 4x5x6x2
z = torch.Tensor(4,5,6,2)
--- for more dimensions, (here a 6D tensor) one can do:
s = torch.LongStorage(6)
s[1] = 4; s[2] = 5; s[3] = 6; s[4] = 2; s[5] = 7; s[6] = 3;
x = torch.Tensor(s)

----------------------------------------------------
-- 1. get nDim, size of a tensor
----------------------------------------------------
-- The number of dimensions of a Tensor can be queried by nDimension() or dim().
-- Size of the i-th dimension is returned by size(i).
-- A LongStorage containing all the dimensions can be returned by size().
print (x:nDimension())
print (x:size())

----------------------------------------------------
-- 2. Internal data representation
----------------------------------------------------
-- The actual data of a Tensor is contained into a Storage.
-- It can be accessed using storage().
-- While the memory of a Tensor has to be contained in this unique Storage,
-- it might not be contiguous: the first position used in the Storage is given by storageOffset() (starting at 1).
-- And the jump needed to go from one element to another element in the i-th dimension is given by stride(i).
-- In other words, given a 3D tensor

x = torch.Tensor(7,7,7)

-- accessing the element (3,4,5) can be done by
print(x[3][4][5])
-- or equivalently (but slowly!)
-- print(x:storage()[x:storageOffset()+(3-1)*x:stride(1)+(4-1)*x:stride(2)+(5-1)*x:stride(3)])
-- print(x:storageOffset())
-- print(x:stride(1))
-- print(x:stride(2))
-- print(x:stride(3))



-- One could say that a Tensor is a particular way of viewing a Storage:
-- a Storage only represents a chunk of memory,
-- while the Tensor interprets this chunk of memory as having dimensions:
x = torch.Tensor(4,5)
s = x:storage()
for i=1,s:size() do -- fill up the Storage
  s[i] = i
end
print(x)
-- print(x:storage())


-- Note also that in Torch7 elements in the same row [elements along the last dimension]
-- are contiguous in memory for a matrix [tensor]:

x = torch.Tensor(4,5)
i = 0

x:apply(function()
  i = i + 1
  return i
end)
print(x)
print(x:stride())
