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
-- print(x)
-- print(x:storage())


-- Note also that in Torch7 elements in the same row [elements along the last dimension]
-- are contiguous in memory for a matrix [tensor]:

x = torch.Tensor(4,5)
i = 0

x:apply(function()
  i = i + 1
  return i
end)
-- print(x)
-- print(x:stride())

----------------------------------------------------
-- 2. Tensor Type
----------------------------------------------------
--[[
  ByteTensor -- contains unsigned chars
  CharTensor -- contains signed chars
  ShortTensor -- contains shorts
  IntTensor -- contains ints
  FloatTensor -- contains floats
  DoubleTensor -- contains doubles
]]--

-- Most numeric operations are implemented only for FloatTensor and DoubleTensor.
-- Other Tensor types are useful if you want to save memory space.

----------------------------------------------------
-- 2.1 Default Tensor type
----------------------------------------------------
-- For convenience, an alias torch.Tensor is provided,
-- which allows the user to write type-independent scripts,
-- which can then ran after choosing the desired Tensor type with a call like
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------
-- 2.2 Efficient memory management
----------------------------------------------------
-- All tensor operations in this class do not make any memory copy.
-- All these methods transform the existing tensor,
-- or return a new tensor referencing the same storage.
-- This magical behavior is internally obtained by good usage of the stride() and storageOffset(). Example:
x = torch.Tensor(5):zero()
-- print(x)

-- narrow() returns a Tensor
-- referencing the same Storage as x
x:narrow(1, 2, 3):fill(1)
-- print(x)

-- If you really need to copy a Tensor, you can use the copy() method:
y = torch.Tensor(x:size()):copy(x)
-- Or the convenience method
y = x:clone()

-- We now describe all the methods for Tensor.
-- If you want to specify the Tensor type,
-- just replace Tensor by the name of the Tensor variant (like CharTensor).
x = torch.CharTensor(5)

----------------------------------------------------
-- 3. Tensor constructors
----------------------------------------------------
-- Tensor constructors, create new Tensor object, optionally, allocating new memory.
-- By default the elements of a newly allocated memory are not initialized, therefore, might contain arbitrary numbers.
-- Here are several ways to construct a new Tensor.

-- torch.Tensor()
-- Returns an empty tensor.

-- torch.Tensor(tensor)
-- Returns a new tensor which reference the same Storage than the given tensor.
-- The size, stride, and storage offset are the same than the given tensor.

-- The new Tensor is now going to "view" the same storage as the given tensor.
-- As a result, any modification in the elements of the Tensor will have a impact on the elements of the given tensor,
-- and vice-versa. No memory copy!

x = torch.Tensor(2,5):fill(3.14)
print(x)
y = torch.Tensor(x)
y:zero()
print(x)

----------------------------------------------------
-- 3.1 torch.Tensor(sz1 [,sz2 [,sz3 [,sz4]]]])
----------------------------------------------------
-- Create a tensor up to 4 dimensions.
-- The tensor size will be sz1 x sz2 x sx3 x sz4.

----------------------------------------------------
-- 3.2 torch.Tensor(sizes, [strides])
----------------------------------------------------
-- Create a tensor of any number of dimensions.
-- The LongStorage sizes gives the size in each dimension of the tensor.
-- The optional LongStorage strides gives the jump necessary to go from one element to the next one in the each dimension.
-- Of course, sizes and strides must have the same number of elements.
-- If not given, or if some elements of strides are negative,
-- the stride() will be computed such that the tensor is as contiguous as possible in memory.

-- Example, create a 4D 4x4x3x2 tensor:
x = torch.Tensor(torch.LongStorage({4,4,3,2}))
-- Playing with the strides can give some interesting things:
x = torch.Tensor(torch.LongStorage({4}), torch.LongStorage({0})):zero() -- zeroes the tensor
print(x)
x[1] = 1 -- all elements point to the same address!
print(x)

----------------------------------------------------
-- 3.3 torch.Tensor(storage, [storageOffset, sizes, [strides]])
----------------------------------------------------
-- Returns a tensor which uses the existing Storage storage,
-- starting at position storageOffset (>=1).
-- The size of each dimension of the tensor is given by the LongStorage sizes.

-- If only storage is provided, it will create a 1D Tensor viewing the all Storage.

-- The jump necessary to go from one element to the next one in each dimension is given by the optional argument LongStorage strides.
-- If not given, or if some elements of strides are negative,
-- the stride() will be computed such that the tensor is as contiguous as possible in memory.

-- Any modification in the elements of the Storage will have an impact on the elements of the new Tensor,
-- and vice-versa. There is no memory copy!


-- creates a storage with 10 elements
s = torch.Storage(10):fill(1)

-- we want to see it as a 2x5 tensor
x = torch.Tensor(s, 1, torch.LongStorage{2,5})

x:zero()
print(s)
print(x)

----------------------------------------------------
-- 3.4 torch.Tensor(storage, [storageOffset, sz1 [, st1 ... [, sz4 [, st4]]]])
----------------------------------------------------
-- Convenience constructor (for the previous constructor) assuming a number of dimensions inferior or equal to 4.
-- szi is the size in the i-th dimension, and sti it the stride in the i-th dimension.
