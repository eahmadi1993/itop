from data.data import Track

tr = Track()

print(tr.track)

tr.load_track()

for trc, tr_shape in tr.track.items():
    print(trc, tr_shape.shape)