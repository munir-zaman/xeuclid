from xeuclid import *
import xeuclid.tikzrc as tikzrc

tikz=Tikz('lagrange_poly_plot.tex', preamble=tikzrc.standalone)

tikz.usepackage("pgfplots")
tikz.usepackage('ifthen')
tikz.begin('document')

tikz.begin('tikzpicture')

tikz.draw_grid(x_range=[-5,5], y_range=[-5,5], step=.2, style="solid", color="LightSteelBlue!70!white")
tikz.draw_grid(x_range=[-5,5], y_range=[-5,5], step=1, style="solid", color="gray")
tikz.draw_axis(x_range=[-5,5], y_range=[-5,5])

tikz.pgfplots_begin_axis(x_range=[-5,5], y_range=[-5,5], show_axis_lines=True)

tikz.pgfplots_lagrange_polynomial_from_points([(1,1), (2,3), (0, -1), (5, 4), (-5, -3), (4, 4.5)])

tikz.pgfplots_end_axis()

tikz.draw_points((1,1), (2,3), (0, -1), (5, 4), (-5, -3), (4, 4.5))

tikz.end('tikzpicture')
tikz.end('document')

tikz.pdf(shell_escape=True)