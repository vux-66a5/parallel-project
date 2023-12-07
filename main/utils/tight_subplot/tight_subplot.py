

def tight_subplot(Nh, Nw, gap, marg_h, marg_w):

    if gap is None:
        gap = 0.02
    if marg_h is None:
        marg_h = 0.05
    if marg_w is None:
        marg_w = 0.05

    if len(gap) == 1:
        gap = [gap, gap]
    if len(marg_w) == 1:
        marg_w = [marg_w, marg_w]
    if len(marg_h) == 1:
        marg_h = [marg_h, marg_h]

    ha = []
    pos = []
    for i in range(Nh):
        for j in range(Nw):
            left = j / Nw + marg_w[0]
            bottom = 1 - (i + 1) / Nh + marg_h[0]
            width = 1 / Nw - marg_w[0] - marg_w[1] - (Nw - 1) * gap[1] / Nw
            height = 1 / Nh - marg_h[0] - marg_h[1] - (Nh - 1) * gap[0] / Nh
            ha.append((left, bottom, width, height))
            pos.append((left, bottom, width, height))
    return ha, pos

