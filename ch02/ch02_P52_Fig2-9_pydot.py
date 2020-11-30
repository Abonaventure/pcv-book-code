import pydot

g = pydot.Dot(graph_type='graph')

g.add_node(pydot.Node(str(0), fontcolor='transparent'))
for i in range(5):
  g.add_node(pydot.Node(str(i + 1)))
  g.add_edge(pydot.Edge(str(0), str(i + 1)))
  for j in range(5):
    g.add_node(pydot.Node(str(j + 1) + '_' + str(i + 1)))
    g.add_edge(pydot.Edge(str(j + 1) + '_' + str(i + 1), str(j + 1)))
# g.write_png('../images/ch02/ch02_fig2-9_graph.png', prog='neato') 
# 源码中write_png函数已经改成write函数,默认生产图片format=raw所要重新设置图片格式format，不然普通图片工具打不开
g.write('graph.jpg', prog='neato', format='png', encoding=None)



from PIL import Image
from pylab import *
 
# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc", size=14)

figure()
im = array(Image.open('graph.jpg'))
imshow(im)
title(u'Pydot Sample', fontproperties=font)
axis('off')
show()