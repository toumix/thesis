from discopy import *

params = dict(to_tikz=True, textpad=(.25, .25))

u, v, w, s, t, s_, t_ = (Ty("${}$".format(x))
    for x in ("u", "v", "w", "s", "t", "s'", "t'"))
f, f_ = Box('$f$', s, t), Box("$f'$", s_, t_)
layer = Id(u) @ f @ Id(v)
layer.draw(path="layer.tex", **params)

interchange_left = Id(u) @ f @ Id(v @ s_ @ w) >> Id(u @ t @ v) @ f_ @ Id(w)
interchange_right = interchange_left.interchange(0, 1)
drawing.equation(interchange_left, interchange_right, symbol="$\\sim$",
                 path="interchange.tex", **params)

x, y, z = map(Ty, ("$x$", "$y$", "$z$"))
swap = lambda x, y: Box("SWAP", x @ y, y @ x)
drawing.equation(swap(x, y) >> swap(y, x), Id(x @ y),
                 symbol="$\\sim$", path='swap.tex', **params)

nat_left = f @ Id(z) >> swap(t, z)
nat_right = swap(s, z) >> Id(z) @ f
drawing.equation(nat_left, nat_right, path="naturality.tex",
                 symbol="$\\sim$", **params)

hexagon_left = Id(x) @ swap(y, z) >> swap(x, z) @ Id(y)
hexagon_right = swap(x, y) @ Id(z) >> Id(y) @ swap(x, z)
drawing.equation(hexagon_left, hexagon_right, path="hexagon.tex",
                 symbol="and", space=2, **params)

yang_baxter = swap(x, y) @ Id(z) >> Id(y) @ swap(x, z) >> swap(y, z) @ Id(x),\
    Id(x) @ swap(y, z) >> swap(x, z) @ Id(y) >> Id(z) @ swap(x, y)
drawing.equation(
    *yang_baxter, path="yang-baxter.tex", symbol="$\\sim$", **params)

copy = lambda x: Box('COPY', x, x @ x)
delete = lambda x: Box('DEL', x, Ty())

comonoid = copy(x) @ copy(y) >> Id(x) @ swap(x, y) @ Id(y),\
    delete(x) @ delete(y)
drawing.equation(
    *comonoid, path='comonoid.tex', symbol="and", space=2,
    color='black', draw_as_nodes=True, draw_box_labels=False, **params)

snake_equation = Id(x.r).transpose_l(), Id(x), Id(x.l).transpose_r()
drawing.equation(
    *snake_equation, path="snake-equation.tex", symbol="$\\sim$", **params)

s, n = Ty('s'), Ty('n')
one, two, three = map(lambda x: Word(x, n), ("one", "two", "three"))
plus, equals = Word("plus", n.r @ n @ n.l), Word("equals", n.r @ s @ n.l)
sentence = pregroup.eager_parse(one, plus, two, equals, three)
pregroup.draw(
    sentence, path="pregroup-reduction.tex", to_tikz=True, textpad=(.25, .25),
    textpad_words=(0, .25), draw_types=False, tikz_options="scale=0.666")

one_plus_two = one @ plus @ two >> Cup(n, n.r) @ Id(n) @ Cup(n.l, n)
wiring = Cap(n.r, n) @ Cap(n, n.l) >> Id(n.r) @ Box('plus', n @ n, n) @ Id(n.l)
W = rigid.Functor(
    ob={n: n, s: s},
    ar={one: one, two: two, plus: wiring})
drawing.equation(
    W(one_plus_two), W(one_plus_two).normal_form(), symbol="$\\mapsto$",
    scale=(2, 1), space=2, textpad=(.5, .2), to_tikz="controls",
    path="snake-removal.tex", tikz_options="scale=0.666", fontsize=0.8)
