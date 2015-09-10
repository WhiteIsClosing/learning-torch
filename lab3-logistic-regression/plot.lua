require 'gnuplot'
x = torch.linspace(-5,5)
y = torch.sin(x)
yp = y+0.3+torch.rand(x:size())*0.1
ym = y-(torch.rand(x:size())*0.1+0.3)
yy = torch.cat(x,ym,2)
yy = torch.cat(yy,yp,2)
gnuplot.pngfigure('test.png')
gnuplot.plot({yy,' filledcurves'},{x,yp,'lines ls 1'},{x,ym,'lines ls 1'},{x,y,'lines ls 1'})
gnuplot.plotflush()
