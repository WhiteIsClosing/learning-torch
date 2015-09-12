require 'nn'
require 'nngraph'

x1 = nn.Identity()()
x2 = nn.Identity()()
a = nn.CAddTable()({x1, x2})
m = nn.gModule({x1, x2}, {a})

-- draw graph (the forward graph, '.fg')
graph.dot(m.fg, 'add-table', 'add-table')
