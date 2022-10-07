from data.routes import Track, Spline, ThetaFinder

tr = Track(10, 2)
sp = Spline(tr)
tf = ThetaFinder(tr, sp)
tf.set_initial_conditions(-12.3125, 35.3)
print(tf.find_theta(-12.3125, 35.3))

