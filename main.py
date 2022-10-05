from data.data import Track, Spline

tr = Track()
tr.load_track()

sp = Spline(tr)
traj = sp.get_traj()

print(traj)
