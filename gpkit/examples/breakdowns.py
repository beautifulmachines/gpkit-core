"An example to show off Breakdowns"

from gpkit.breakdowns import Breakdowns
from gpkit.examples.uav import UAV

m = UAV()
sol = m.solve(verbosity=0)
bds = Breakdowns(sol)

print("Cost breakdown (as seen in solution tables)")
print("==============")
bds.show("cost")

print("Variable breakdowns (note the two methods of access)")
print("===================")
(varkey,) = m.varkeys.keys("UAV.Mission.Outbound.AircraftPerf.C_D")
bds.show(varkey)
bds.show("Wing.W")

print("Combining the two above by increasing maxwidth")
print("----------------------------------------------")
bds.show("Outbound.AircraftPerf.C_D", maxwidth=105)

print("Model sensitivity breakdowns (note the two methods of access)")
print("============================")
bds.show("model sensitivities")
bds.show("Aircraft")

print("Exhaustive variable breakdown traces (and configuration arguments)")
print("====================================")
# often useful as a reference point when reading traces
bds.show("Wing.W", height=12)
# includes factors, can be useful for reading traces as well
bds.show("Wing.W", showlegend=True)
print("\nPermissivity = 2 (the default)")
print("----------------")
bds.trace("Wing.W")
print("\nPermissivity = 1 (stops at M_rbar⋅tau instead of splitting it)")
print("----------------")
bds.trace("Wing.W", permissivity=1)
