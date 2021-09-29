from tikz_draw import *


def square_on_side(A, B):
    b = B - A
    a = origin
    d = dist(A, B)
    theta = angle_between_vectors(x_vect, b) - 90
    C = polar( sqrt(2)*d, theta + 45) + A
    D = polar(d, theta) + A
    return [A, B, C, D]

def square_center(A, B):
    C, D = square_on_side(A, B)[2:]
    AC = Line(A, C)
    BD = Line(B, D)
    return AC & BD

tikz=Tikz("sqr_center.tex")
tikz.begin("document")

tikz.begin("figure", config="!h")
tikz.begin("tikzpicture")

A = col_vector([0, 0])
B = col_vector([4, 0])
C = col_vector([6, 5])
D = col_vector([2, 5])

mAB = mid(A, B)
mBC = mid(B, C)
mCD = mid(C, D)
mDA = mid(D, A)

tikz.draw_path(*square_on_side(B, C), cycle=True)
tikz.draw_path(*square_on_side(A, B), cycle=True)
tikz.draw_path(*square_on_side(C, D), cycle=True)
tikz.draw_path(*square_on_side(D, A), cycle=True)

M, N, O, P = square_center(A, B), square_center(B, C), square_center(C, D), square_center(D, A)

tikz.draw_path(A, M, B)
tikz.draw_path(B, N, C)
tikz.draw_path(C, O, D)
tikz.draw_path(D, P, A)

tikz.draw_path(mAB, M)
tikz.draw_path(mBC, N)
tikz.draw_path(mCD, O)
tikz.draw_path(mDA, P)

tikz.draw_path(M, N, O, P, cycle=True, config="thick, cyan!40!black")

tikz.draw_angle(M, B, N, radius=.35)
tikz.draw_angle(O, C, N, radius=.35)
tikz.draw_angle(O, D, P, radius=.35)
tikz.draw_angle(M, A, P, radius=.35)
tikz.draw_angle(N, M, P, radius=.60, config="cyan!40!black")

tikz.draw_points(A, B, C, D)
tikz.draw_points(M, N, O, P)

tikz.node(A, node_config="anchor= north east", text="$A$")
tikz.node(B, node_config="anchor= south east", text="$B$")
tikz.node(C, node_config="anchor= south west", text="$C$")
tikz.node(D, node_config="anchor= north west", text="$D$")

tikz.node(M, node_config="anchor= north", text="$M$")
tikz.node(N, node_config="anchor= north west", text="$N$")
tikz.node(O, node_config="anchor= south", text="$O$")
tikz.node(P, node_config="anchor= south east", text="$P$")

tikz.end("tikzpicture")
tikz.end("figure")

tikz.begin("figure", config="!h")
tikz.begin("tikzpicture")

A = col_vector([0, 0])
B = col_vector([10, 0])
C = col_vector([15, 8])
D = col_vector([5, 8])

mAB = mid(A, B)
mBC = mid(B, C)
mCD = mid(C, D)
mDA = mid(D, A)

M, N, O, P = square_center(B, A), square_center(C, B), square_center(D, C), square_center(A, D)

tikz.draw_path(A, M, B)
tikz.draw_path(B, N, C)
tikz.draw_path(C, O, D)
tikz.draw_path(D, P, A)

tikz.draw_path(mAB, M)
tikz.draw_path(mBC, N)
tikz.draw_path(mCD, O)
tikz.draw_path(mDA, P)

tikz.draw_path(M, N, O, P, cycle=True, config="thick, cyan!40!black")

tikz.draw_angle(N, B, M, radius=.35)
tikz.draw_angle(M, C, O, radius=.35)
tikz.draw_angle(P, D, O, radius=.35)
tikz.draw_angle(P, A, M, radius=.35)
tikz.draw_angle(O, P, M, radius=.60, config="cyan!40!black")

tikz.draw_path(A, B, C, D, cycle=True)
tikz.draw_points(A, B, C, D)
tikz.draw_points(M, N, O, P)

tikz.node(A, node_config="anchor= north east", text="$A$")
tikz.node(B, node_config="anchor= north west", text="$B$")
tikz.node(C, node_config="anchor= south west", text="$C$")
tikz.node(D, node_config="anchor= south east", text="$D$")

tikz.node(M, node_config="anchor= south", text="$M$")
tikz.node(N, node_config="anchor= south east", text="$N$")
tikz.node(O, node_config="anchor= north", text="$O$")
tikz.node(P, node_config="anchor= north west", text="$P$")

tikz.end("tikzpicture")
tikz.end("figure")

tikz.end("document")
tikz.pdf()
