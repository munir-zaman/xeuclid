from euclid import *

A=col_vector([2,1])
B=col_vector([-1,4])
C=col_vector([-2,1])

bisector1=angle_bisector(A,B,C)
bisector2=angle_bisector(B,C,A)
#angle bisector of angle ABC and angle BCA

I=bisector1.intersection(bisector2)
# intersection of bisector1 and bisector2
# I is the incenter of trinagle ABC

tikz=Tikz('triangle.tex', preamble=tikz_config.standalone)

tikz.usepackage('ifthen')

tikz.begin('document')
tikz.begin('tikzpicture')

tikz.draw_grid(x_range=[-5,5],
               y_range=[-5,5],
               color='Black!50')

tikz.draw_axis(x_range=[-5,5], 
               y_range=[-5,5],
               tick_labels=None)

tikz.draw_angle(A,C,B, radius=0.3)
tikz.draw_angle(C,B,A, radius=0.3)
tikz.draw_angle(B,A,C, radius=0.3)

tikz.draw_path(A,B,C,cycle=True)
tikz.draw_path(I,A)
tikz.draw_path(I,B)
tikz.draw_path(I,C)

tikz.draw_points(A,B,C,I)

tikz.node(A, node_config="anchor=west",text=r"$A$")
tikz.node(B, node_config="anchor=south",text=r"$B$")
tikz.node(C, node_config="anchor=east",text=r"$C$")
tikz.node(I, node_config="anchor=north",text=r"$I$")

tikz.end('tikzpicture')
tikz.end('document')

tikz.pdf()
#This will compile the TeX file using pdfLaTeX