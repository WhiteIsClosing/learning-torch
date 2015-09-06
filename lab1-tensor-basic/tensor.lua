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

----------------------------------------------------
-- 4. Function Calls
----------------------------------------------------
-- [Tensor] clone()
-- [Tensor] contiguous
-- [Tensor or string] type(type)
-- [Tensor] typeAs(tensor)
-- [boolean] isTensor(object)
-- [Tensor] byte(), char(), short(), int(), long(), float(), double()
-- [number] nDimension()
-- [number] dim()
-- [number] size(dim)
-- [LongStorage] size()
-- [LongStorage] #self

-- [number] stride(dim)
-- Returns the jump necessary to go from one element to the next one in the specified dimension dim. Example:
--[[
  x = torch.Tensor(4,5):zero()
  > x
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
  [torch.DoubleTensor of dimension 4x5]

  -- elements in a row are contiguous in memory
  > x:stride(2)
  1

  -- to go from one element to the next one in a column
  -- we need here to jump the size of the row
  > x:stride(1)
  5
--]]

----------------------------------------------------
-- 5. Querying Elements
----------------------------------------------------
-- Elements of a tensor can be retrieved with the [index] operator.

-- If index is a number,
-- [index] operator is equivalent to a select(1, index) if the tensor has more than one dimension.
-- This operation returns a slice of the tensor that shares the same underlying storage.
-- If the tensor is a 1D tensor, it returns the value at index in this tensor.

-- If index is a table,
-- the table must contain n numbers,
-- where n is the number of dimensions of the Tensor.
-- It will return the element at the given position.

-- In the same spirit,
-- index might be a LongStorage, specifying the position (in the Tensor) of the element to be retrieved.

-- If index is a ByteTensor
-- in which each element is 0 or 1 then it acts as a selection mask used to extract a subset of the original tensor.
-- This is particularly useful with logical operators like torch.le.
x = torch.Tensor(3,3)
i = 0; x:apply(function() i = i + 1; return i end)
-- x[2] -- returns row 2
-- x[2][3] -- returns row 2, column 3
-- x[{2,3}] -- another way to return row 2, column 3
-- x[torch.LongStorage{2,3}] -- yet another way to return row 2, column 3
-- x[torch.le(x,3)] -- torch.le returns a ByteTensor that acts as a mask

----------------------------------------------------
-- 6. Referencing a tensor to an existing tensor or chunk of memory
----------------------------------------------------
-- A Tensor being a way of viewing a Storage,
-- it is possible to "set" a Tensor such that it views an existing Storage.
-- Note that if you want to perform a set on an empty Tensor like, do it with constructors
y = torch.Storage(10)
x = torch.Tensor(y, 1, 10)
----------------------------------------------------
-- [self] set(tensor)
----------------------------------------------------
x = torch.Tensor(2,5):fill(3.14)
y = torch.Tensor():set(x)
y:zero()
-- shared memory, all zero now
----------------------------------------------------
-- [self] set(storage, [storageOffset, sizes, [strides]])
----------------------------------------------------
-- The Tensor is now going to "view" the given storage,
-- starting at position storageOffset (>=1) with the given dimension sizes and the optional given strides.
-- As the result, any modification in the elements of the Storage will have a impact on the elements of the Tensor, and vice-versa.
-- This is an efficient method, as there is no memory copy!
-- If only storage is provided, the whole storage will be viewed as a 1D Tensor.
-- creates a storage with 10 elements
s = torch.Storage(10):fill(1)

-- we want to see it as a 2x5 tensor
sz = torch.LongStorage({2,5})
x = torch.Tensor()
x:set(s, 1, sz)
x:zero()
----------------------------------------------------
-- 7. Copying and initializing
----------------------------------------------------
-- [self] copy(tensor)
----------------------------------------------------
-- Replace the elements of the Tensor by copying the elements of the given tensor.
-- The number of elements must match, but the sizes might be different.
x = torch.Tensor(4):fill(1)
y = torch.Tensor(2,2):copy(x)
----------------------------------------------------
-- [self] fill(value)
----------------------------------------------------
-- Fill the tensor with the given value.
torch.DoubleTensor(4):fill(3.14)
----------------------------------------------------
-- [self] zero()
----------------------------------------------------
torch.Tensor(4):zero()


----------------------------------------------------
-- 8. Resizing
----------------------------------------------------
-- When resizing to a larger size, the underlying Storage is resized to fit all the elements of the Tensor.
-- When resizing to a smaller size, the underlying Storage is not resized.
-- Important note:
-- the content of a Tensor after resizing is undertermined as strides might have been completely changed.
-- In particular, the elements of the resized tensor are contiguous in memory.

-- [self] resizeAs(tensor)
-- Resize the tensor as the given tensor (of the same type).

-- [self] resize(sizes)
-- Resize the tensor according to the given LongStorage size.

-- [self] resize(sz1 [,sz2 [,sz3 [,sz4]]]])
-- Convenience method of the previous method, working for a number of dimensions up to 4.

----------------------------------------------------
-- 9. Extracting sub-tensors
----------------------------------------------------
-- Each of these methods returns a Tensor which is a sub-tensor of the given tensor,
-- with the same Storage. Hence, any modification in the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.

-- These methods are very fast, as they do not involve any memory copy.

----------------------------------------------------
-- [self] narrow(dim, index, size)
----------------------------------------------------
-- Returns a new Tensor which is a narrowed version of the current one: the dimension dim is narrowed from index to index+size-1.
x = torch.Tensor(5, 6):zero()
-- print(x)
y = x:narrow(1, 2, 3) -- narrow dimension 1 from index 2 to index 2+3-1
y:fill(1) -- fill with 1
-- print(y)
-- print(x)

-- x = torch.Tensor(5, 6):zero()
-- > x
--
-- 0 0 0 0 0 0
-- 0 0 0 0 0 0
-- 0 0 0 0 0 0
-- 0 0 0 0 0 0
-- 0 0 0 0 0 0
-- [torch.DoubleTensor of dimension 5x6]
--
-- y = x:narrow(1, 2, 3) -- narrow dimension 1 from index 2 to index 2+3-1
-- y:fill(1) -- fill with 1
-- > y
--  1  1  1  1  1  1
--  1  1  1  1  1  1
--  1  1  1  1  1  1
-- [torch.DoubleTensor of dimension 3x6]
--
-- > x -- memory in x has been modified!
--  0  0  0  0  0  0
--  1  1  1  1  1  1
--  1  1  1  1  1  1
--  1  1  1  1  1  1
--  0  0  0  0  0  0
-- [torch.DoubleTensor of dimension 5x6]

----------------------------------------------------
-- [Tensor] sub(dim1s, dim1e ... [, dim4s [, dim4e]])
----------------------------------------------------
-- This method is equivalent to do a series of narrow up to the first 4 dimensions.
-- It returns a new Tensor which is a sub-tensor going from index dimis to dimie in the i-th dimension.
-- Negative values are interpreted index starting from the end: -1 is the last index, -2 is the index before the last index, ...

-- x = torch.Tensor(5, 6):zero()
-- > x
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
-- [torch.DoubleTensor of dimension 5x6]
--
-- y = x:sub(2,4):fill(1) -- y is sub-tensor of x:
-- > y                    -- dimension 1 starts at index 2, ends at index 4
--  1  1  1  1  1  1
--  1  1  1  1  1  1
--  1  1  1  1  1  1
-- [torch.DoubleTensor of dimension 3x6]
--
-- > x                    -- x has been modified!
--  0  0  0  0  0  0
--  1  1  1  1  1  1
--  1  1  1  1  1  1
--  1  1  1  1  1  1
--  0  0  0  0  0  0
-- [torch.DoubleTensor of dimension 5x6]
--
-- z = x:sub(2,4,3,4):fill(2) -- we now take a new sub-tensor
-- > z                        -- dimension 1 starts at index 2, ends at index 4
--                            -- dimension 2 starts at index 3, ends at index 4
--  2  2
--  2  2
--  2  2
-- [torch.DoubleTensor of dimension 3x2]
--
-- > x                        -- x has been modified
--
--  0  0  0  0  0  0
--  1  1  2  2  1  1
--  1  1  2  2  1  1
--  1  1  2  2  1  1
--  0  0  0  0  0  0
-- [torch.DoubleTensor of dimension 5x6]
--
-- > y:sub(-1, -1, 3, 4)      -- negative values = bounds
--  2  2
-- [torch.DoubleTensor of dimension 1x2]

----------------------------------------------------
-- [Tensor] select(dim, index)
----------------------------------------------------
-- Returns a new Tensor which is a tensor slice at the given index in the dimension dim.
-- The returned tensor has one less dimension: the dimension dim is removed.
-- As a result, it is not possible to select() on a 1D tensor.
-- Note that "selecting" on the first dimension is equivalent to use the [] operator

-- x = torch.Tensor(5,6):zero()
-- > x
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
-- [torch.DoubleTensor of dimension 5x6]
--
-- y = x:select(1, 2):fill(2) -- select row 2 and fill up
-- > y
--  2
--  2
--  2
--  2
--  2
--  2
-- [torch.DoubleTensor of dimension 6]
--
-- > x
--  0  0  0  0  0  0
--  2  2  2  2  2  2
--  0  0  0  0  0  0
--  0  0  0  0  0  0
--  0  0  0  0  0  0
-- [torch.DoubleTensor of dimension 5x6]
--
-- z = x:select(2,5):fill(5) -- select column 5 and fill up
-- > z
--  5
--  5
--  5
--  5
--  5
-- [torch.DoubleTensor of dimension 5]
--
-- > x
--  0  0  0  0  5  0
--  2  2  2  2  5  2
--  0  0  0  0  5  0
--  0  0  0  0  5  0
--  0  0  0  0  5  0
-- [torch.DoubleTensor of dimension 5x6]

----------------------------------------------------
-- [Tensor] [{ dim1,dim2,... }] or [{ {dim1s,dim1e}, {dim2s,dim2e} }]
----------------------------------------------------
-- The indexing operator []
-- can be used to combine narrow/sub and select in a concise an efficient way.
-- It can also be used to copy, and fill (sub) tensors.

-- This operator also works with an input mask made of a ByteTensor with 0 and 1 elements, e.g with a logical operator.
-- x = torch.Tensor(5, 6):zero()
-- x[{ 1,3 }] = 1 -- sets element at (i=1,j=3) to 1
-- x[{ 2,{2,4} }] = 2  -- sets a slice of 3 elements to 2
-- x[{ {},4 }] = -1 -- sets the full 4th column to -1
-- x[{ {},2 }] = torch.range(1,5) -- copy a 1D tensor to a slice of x
-- x[torch.lt(x,0)] = -2 -- sets all negative elements to -2 via a mask
-- print(x)
-- x = torch.Tensor(5, 6):zero()
-- > x
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
--  0 0 0 0 0 0
-- [torch.DoubleTensor of dimension 5x6]
--
-- x[{ 1,3 }] = 1 -- sets element at (i=1,j=3) to 1
-- > x
--  0  0  1  0  0  0
--  0  0  0  0  0  0
--  0  0  0  0  0  0
--  0  0  0  0  0  0
--  0  0  0  0  0  0
-- [torch.DoubleTensor of dimension 5x6]
--
-- x[{ 2,{2,4} }] = 2  -- sets a slice of 3 elements to 2
-- > x
--  0  0  1  0  0  0
--  0  2  2  2  0  0
--  0  0  0  0  0  0
--  0  0  0  0  0  0
--  0  0  0  0  0  0
-- [torch.DoubleTensor of dimension 5x6]
--
-- x[{ {},4 }] = -1 -- sets the full 4th column to -1
-- > x
--  0  0  1 -1  0  0
--  0  2  2 -1  0  0
--  0  0  0 -1  0  0
--  0  0  0 -1  0  0
--  0  0  0 -1  0  0
-- [torch.DoubleTensor of dimension 5x6]
--
-- x[{ {},2 }] = torch.range(1,5) -- copy a 1D tensor to a slice of x
-- > x
--
--  0  1  1 -1  0  0
--  0  2  2 -1  0  0
--  0  3  0 -1  0  0
--  0  4  0 -1  0  0
--  0  5  0 -1  0  0
-- [torch.DoubleTensor of dimension 5x6]
--
-- x[torch.lt(x,0)] = -2 -- sets all negative elements to -2 via a mask
-- > x
--
--  0  1  1 -2  0  0
--  0  2  2 -2  0  0
--  0  3  0 -2  0  0
--  0  4  0 -2  0  0
--  0  5  0 -2  0  0
-- [torch.DoubleTensor of dimension 5x6]

----------------------------------------------------
-- [Tensor] index(dim, index)
----------------------------------------------------
-- Returns a new Tensor which indexes the original Tensor along dimension dim using the entries in torch.LongTensor index.
-- The returned Tensor has the same number of dimensions as the original Tensor.
-- The returned Tensor does not use the same storage as the original Tensor -- see below for storing the result in an existing Tensor.

-- [self] narrow(dim, index, size)
-- [Tensor] sub(dim1s, dim1e ... [, dim4s [, dim4e]])
-- [Tensor] select(dim, index)
-- [Tensor] [{ dim1,dim2,... }] or [{ {dim1s,dim1e}, {dim2s,dim2e} }]
-- [Tensor] index(dim, index)
