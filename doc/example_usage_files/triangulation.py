from euclid import *

tikz = Tikz('triangulation.tex', preamble=tikz_config.standalone)
tikz.begin('document')
tikz.begin('tikzpicture')

rng = np.random.default_rng()
points = rng.random((100, 2)) * 20

tikz.draw_triangulation(points, color='Pink!75!Red')

for p in points:
    tikz.draw_point(col_vector(p), color='Red!75', fill_color='Red')

tikz.end('tikzpicture')
tikz.end('document')

tikz.pdf()