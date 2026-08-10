# 03_percolation_zd — Site percolation on $\mathbb Z^d$, $d=2..6/7$

**Status: not started.**

$Y_i$ = number of open sites in a box of side $i$ **connected to a full face (side) of
the box**, at $p=p_c$ — not the cluster containing the origin. This is a deliberate
correction to `presentation18-05-2026`'s origin-anchored $V(r)$ (see `PLAN.md` ground
rule 7); rationale and exact face/connectivity convention to be pinned down before
coding.

This rung is also the first place $\cost(i)=i^d$ (Assumption 7) is a geometric claim
about a BFS/union-find simulation rather than an assumed formula — should be measured,
not assumed, before it's used in any allocation calculation.

$d=2$ has a known $d_f=91/48$ (Stauffer & Aharony) to check against.
