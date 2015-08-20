-- Two dashes start a one-line comment.

--[[
     Adding two ['s and ]'s makes it a
     multi-line comment.
--]]

----------------------------------------------------
-- 0. Installation
----------------------------------------------------
-- Get Started: http://torch.ch/docs/getting-started.html
-- Newbie Tutorial: http://tylerneylon.com/a/learn-lua/

----------------------------------------------------
-- 1. Variables and flow control.
----------------------------------------------------

s = 'walternate'  -- Immutable strings like Python.
t = "double-quotes are also fine"
u = [[ Double brackets
       start and end
       multi-line strings.]]
t = nil  -- Undefines t; Lua has garbage collection.

num = 0
-- Blocks are denoted with keywords like do/end:
while num < 50 do
  num = num + 1  -- No ++ or += type operators.
end

-- if clause
if num > 40 then
  print('over 40')
elseif s~='walternate' then -- ~= is not equal
  -- Equality check is == like python; ok for strs.
  io.write('not over 40\n') -- Defaults to stdout.
else
  -- Variables are global by defaults
  thisIsGlobal = 5 -- Camel case is common

  -- How to make variable local:
  local line = io.read() -- Read next stdin line.

  -- String concatenation uess the ... operator:
  print('Winter is coming '..line)
end

-- Undefined variables return nil.
-- This is not an error:
foo = anUnknownVariable  -- Now foo = nil.

aBoolValue = false

-- Only nil and false are falsy; 0 and '' are true!
if not aBoolValue then print('twas false') end

-- 'or' and 'and' are short-circuited.
-- This is similar to the a?b:c operator in C/js:
ans = aBoolValue and 'yes' or 'no'  --> 'no'

karlSum = 0
for i = 1, 100 do  -- The range includes both ends.
  karlSum = karlSum + i
end

-- Use "100, 1, -1" as the range to count down:
fredSum = 0
for j = 100, 1, -1 do
  fredSum = fredSum + j
end

-- In general, the range is begin, end[, step].

-- Another loop construct:
repeat
  print('the way of the future')
  num = num - 1
until num == 0

----------------------------------------------------
-- 2. Functions.
----------------------------------------------------
function fib(n)
  if n < 2 then return 1 end
  return fib(n - 2) + fib(n - 1)
end

-- Closures and anonymous functions are ok:
function adder(x)
  -- The returned function is created when adder is
  -- called, and remembers the value of x:
  return function (y) return x + y end
end

a1 = adder(9)
a2 = adder(36)
print(a1(16))  --> 25
print(a2(64))  --> 100
