import numpy as np
import pylab

r_sac = np.loadtxt("dump_reward.txt-sac-03")
v_sac = np.loadtxt("dump_violation.txt-sac-03")

r_safe_sac = np.loadtxt("dump_reward.txt-sac-safe-02")
v_safe_sac = np.loadtxt("dump_violation.txt-sac-safe-02")

l_safe_sac = np.loadtxt("dump_lambda.txt-sac-safe-02")

steps = np.linspace(1,len(r_sac),len(v_sac))

pylab.figure()
pylab.plot(steps, r_sac, 'b', linewidth=3, label="sac")
pylab.plot(steps, r_safe_sac, 'r', linewidth=2, label="safe sac - linear")
pylab.legend()
pylab.xlabel("steps")
pylab.ylabel("discounted reward")
pylab.show()

pylab.figure()
pylab.plot(steps, v_sac, 'b', linewidth=3, label="sac")
pylab.plot(steps, v_safe_sac, 'r', linewidth=2, label="safe sac - linear")
pylab.legend()
pylab.xlabel("step")
pylab.ylabel("number of safety violations")
pylab.show()

pylab.figure()
pylab.plot(steps, l_safe_sac, 'r', linewidth=3, label="safe sac - linear")
pylab.legend()
pylab.ylim(-5, 50)
pylab.xlabel("steps")
pylab.ylabel("lambda")
pylab.show()
