import numpy as np
import pylab

r_sac = np.loadtxt("dump_reward.txt-safe-sac-quad-01")
v_sac = np.loadtxt("dump_violation.txt-safe-sac-quad-01")
#l_sac = np.loadtxt("dump_lambda.txt")

steps = np.linspace(1,len(r_sac),len(v_sac))

pylab.figure()
pylab.plot(steps, r_sac, 'k', linewidth=3, label="safe sac with quadratic corr")
#pylab.plot(steps, r_safe_sac, 'r', linewidth=2, label="safe sac")
pylab.legend()
pylab.xlabel("steps")
pylab.ylabel("discounted reward")
pylab.show()

pylab.figure()
pylab.plot(steps, v_sac, 'k', linewidth=3, label="safe sac with quadratic corr")
#pylab.plot(steps, v_safe_sac, 'r', linewidth=2, label="safe sac")
pylab.legend()
pylab.xlabel("step")
pylab.ylabel("number of safety violations")
pylab.show()


pylab.figure()
pylab.plot(steps, l_sac, 'k', linewidth=3, label="safe sac with linear corr")
#pylab.plot(steps, r_safe_sac, 'r', linewidth=2, label="safe sac")
pylab.legend()
pylab.ylim(-10, 50)
pylab.xlabel("steps")
pylab.ylabel("value of lambda")
pylab.show()
