`euclid` is little project of mine that I've been working on for the past few weeks. It's a set of python scripts 
that lets you do analytic geometry in `python`. You can also draw TikZ diagrams using `tikz_draw.py`. 
**It's a work in progress and therefore it can't do much.** 
(Geometric objects are defined using *parametric equations*.)

# Example Usage

```python
from euclid2 import *
from tikz_draw import *

A=col_vector([2,1])
B=col_vector([-1,4])
C=col_vector([-2,1])

bisector1=angle_bisector(A,B,C)
bisector2=angle_bisector(B,C,A)
#angle bisector of angle ABC and angle BCA

I=bisector1.intersection(bisector2)
# intersection of bisector1 and bisector2
# I is the incenter of trinagle ABC

tikz=Tikz('triangle.tex')

tikz.begin('document')
tikz.begin('tikzpicture')

tikz.draw_grid(x_range=[-5,5], y_range=[-5,5])
tikz.draw_axis(x_range=[-5,5], y_range=[-5,5])

tikz.draw_angle(A,C,B, radius=0.3)
tikz.draw_angle(A,B,C, radius=0.3)
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
```

**Output:**

![Output](doc/example_usage_files/triangle.png)

```python
from euclid2 import *
from tikz_draw import *

A=origin
B=col_vector([3,3])

c1=Circle(A, 2)
c2=Circle(B, 1)

T=common_tangents(c1, c2)

tikz=Tikz('common_tangents.tex')

tikz.begin('document')
tikz.begin('tikzpicture')

tikz.draw_circle(c1, config="cyan!50!black, thick")
tikz.draw_circle(c2, config="cyan!50!black, thick")

tikz.draw_line(T[0], x_range=[-8,8], y_range=[-8,8])
tikz.draw_line(T[1], x_range=[-8,8], y_range=[-8,8])
tikz.draw_line(Line(c1.center, c2.center), config="gray", x_range=[-8,8], y_range=[-8,8])
tikz.draw_path(c1.center, T[0] & T[1])

c1T0=intersection_line_circle(T[0], c1)[0]
c2T0=intersection_line_circle(T[0], c2)[0]

tikz.draw_path(c1.center, c1T0)
tikz.draw_path(c2.center, c2T0)

tikz.draw_points(c1.center, c2.center, T[0] & T[1], c1T0, c2T0)

tikz.node(c1.center, node_config="anchor=north, inner sep=8pt", text="$C_2$")
tikz.node(c2.center, node_config="anchor=north, inner sep=8pt", text="$C_1$")

tikz.node(c2T0, node_config="anchor=south, inner sep=8pt", text="$T_1$")
tikz.node(c1T0, node_config="anchor=south, inner sep=8pt", text="$T_2$")

tikz.node(mid(c2T0, c2.center)-col_vector((0,0.25)), node_config="anchor=east, inner sep=2pt", text="$r_1$")
tikz.node(mid(c1T0, c1.center), node_config="anchor=east, inner sep=4pt", text="$r_2$")

tikz.node(T[0] & T[1], node_config="anchor=north, inner sep=5pt", text="$P$")

tikz.end('tikzpicture')
tikz.end('document')

tikz.pdf()
```

**Output:**

![Output](doc/example_usage_files/common_tangents.jpg)
