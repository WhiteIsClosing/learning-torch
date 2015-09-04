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
