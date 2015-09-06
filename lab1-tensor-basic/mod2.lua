-- Suppose the file mod.lua looks like this:
local M = {}
print('Hi')

local function sayMyName()
  print('Hrunkner')
end

function M.sayHello()
  print('Hi')
  sayMyName()
end

return M
