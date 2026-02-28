"""Textbook geometric programming examples from Duffin, Peterson, and Zener.

Five classic GP problems demonstrating the breadth of engineering design
problems that can be expressed in standard GP form. Each problem optimizes
a cost (or its reciprocal) subject to posynomial constraints.

Problems:
  1. Box transport: minimize ferry + box cost to move gravel across a river
  2. Fence: maximize rectangular plot area with a fixed fence length
  3. Beam cross-section: maximize bending stiffness from a log of radius R
  4. Box from sheet: maximize volume of a box cut from a tin sheet
  5. Work/sleep: maximize daily savings balancing work, sleep, and leisure

Extracted from docs/source/ipynb/Examples from Geometric Programming.ipynb
(nbformat-3, legacy notebook). Ported from Python 2 / gpkit.GP to current
gpkit API (Model).
"""

from gpkit import Model, Var


# ---------------------------------------------------------------------------
# 1. Box transport
#    How large a box and how many trips to move 400 cubic yards of gravel?
# ---------------------------------------------------------------------------
class BoxTransport(Model):
    "Minimize ferry and box cost to transport a fixed volume of gravel."

    V1 = Var("-", "volume of gravel to transport", value=400)
    c_sides = Var("-", "cost of box sides and bottom (per unit area)", value=10)
    c_ends = Var("-", "cost of box ends (per unit area)", value=20)
    c_ferry = Var("-", "cost per ferry trip", value=0.10)

    length = Var("-", "box length")
    w = Var("-", "box width")
    h = Var("-", "box height")

    def setup(self):
        V1, c_sides, c_ends, c_ferry = self.V1, self.c_sides, self.c_ends, self.c_ferry
        length, w, h = self.length, self.w, self.h
        self.cost = (
            c_ferry * V1 / (length * w * h)
            + c_ends * 2 * w * h
            + c_sides * (2 * length * h + w * length)
        )
        return []


# ---------------------------------------------------------------------------
# 2. Fence
#    Maximize rectangular plot area with one side along a river (no fence needed)
#    given total fence length L = 1 yard.
# ---------------------------------------------------------------------------
class FencePlot(Model):
    "Maximize rectangular plot area with a fixed fence length."

    L = Var("-", "total fence length", value=1)

    length = Var("-", "plot side parallel to river")
    w = Var("-", "plot side perpendicular to river")

    def setup(self):
        length, w, L = self.length, self.w, self.L
        self.cost = 1 / (length * w)
        return [2 * w + length <= L]


# ---------------------------------------------------------------------------
# 3. Beam cross-section
#    From a log of radius R = 1 inch, cut a rectangular beam (d x w) to
#    maximize the section modulus (proportional to w * d^3).
# ---------------------------------------------------------------------------
class BeamCrossSection(Model):
    "Maximize bending stiffness of a rectangular beam cut from a log."

    R = Var("-", "log radius", value=1)

    d = Var("-", "beam depth")
    w = Var("-", "beam width")

    def setup(self):
        d, w, R = self.d, self.w, self.R
        self.cost = 1 / (w * d**3)
        return [(d / 2) ** 2 + (w / 2) ** 2 <= R**2]


# ---------------------------------------------------------------------------
# 4. Box from sheet
#    Cut corner squares from a 1-inch tin sheet to fold into an open box;
#    maximize box volume.
# ---------------------------------------------------------------------------
class BoxFromSheet(Model):
    "Maximize open-top box volume cut from a square tin sheet."

    L = Var("-", "tin sheet side length", value=1)

    length = Var("-", "box length")
    w = Var("-", "box width")
    h = Var("-", "box height")
    cx = Var("-", "corner cutout length in x dimension")
    cy = Var("-", "corner cutout length in y dimension")

    def setup(self):
        length, w, h = self.length, self.w, self.h
        cx, cy, L = self.cx, self.cy, self.L
        self.cost = 1 / (length * w * h)
        return [
            L >= w + 2 * cx,
            L >= length + 2 * cy,
            cx >= h,
            cy >= h,
        ]


# ---------------------------------------------------------------------------
# 5. Work/sleep allocation
#    Maximize daily savings balancing work productivity, sleep, leisure,
#    and music (from Duffin et al.).
# ---------------------------------------------------------------------------
class WorkSleep(Model):
    "Maximize daily savings with work/sleep/leisure time allocation."

    p = Var("-", "pay rate per unit productivity", value=0.5)
    c = Var("-", "cost of new records per hour", value=5)
    te = Var("-", "hours required for eating", value=3)

    tw = Var("-", "hours spent working")
    ts = Var("-", "hours spent sleeping")
    to = Var("-", "hours spent eating and/or listening")
    tm = Var("-", "hours spent listening to music")
    s = Var("-", "savings accrued per day")

    def setup(self):
        p, c, te = self.p, self.c, self.te
        tw, ts, to, tm, s = self.tw, self.ts, self.to, self.tm, self.s
        self.cost = 1 / s
        return [
            p * tw**1.5 * ts**0.75 * tm**0.1 >= s + c * tm,
            tw + ts + to <= 24,
            to >= tm,
            to >= te,
        ]


# ---------------------------------------------------------------------------
# Solve all five and expose individual solutions for testing
# ---------------------------------------------------------------------------
m1 = BoxTransport()
sol1 = m1.solve(verbosity=0)

m2 = FencePlot()
sol2 = m2.solve(verbosity=0)

m3 = BeamCrossSection()
sol3 = m3.solve(verbosity=0)

m4 = BoxFromSheet()
sol4 = m4.solve(verbosity=0)

m5 = WorkSleep()
sol5 = m5.solve(verbosity=0)

if __name__ == "__main__":
    print("=== 1. Box Transport ===")
    print(sol1.summary())
    print("\n=== 2. Fence Plot ===")
    print(sol2.summary())
    print("\n=== 3. Beam Cross-Section ===")
    print(sol3.summary())
    print("\n=== 4. Box from Sheet ===")
    print(sol4.summary())
    print("\n=== 5. Work/Sleep Allocation ===")
    print(sol5.summary())
