import json
import matplotlib.pyplot as plt
import numpy as np
import sys


results = []
if len(sys.argv) == 2:
    with open(sys.argv[1], 'r') as j:
        results = json.load(j)
else:
    with open('result_datas/all_result.json', 'r') as j:
        results = json.load(j)

assemble = [float(x["ASSEMBLE"])/100 for x in results]
compute = [float(x["COMPUTE"])/100 for x in results]
solve = [float(x["SOLVE"])/100 for x in results]

# total_time =[assemble[i]+compute[i]+solve[i] for i in range(5)]

bars = np.add(solve, compute).tolist()

N = 5

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, solve, width)
p2 = plt.bar(ind, compute, width, bottom=solve)
p3 = plt.bar(ind, assemble, width, bottom=bars)

plt.ylabel('Time')
plt.title('Method')
plt.xticks(ind, [x["name"] for x in results])
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0], p3[0]), ('Solve', 'Compute', 'Assemble'))


for r1, r2, r3 in zip(p1, p2, p3):
    h1 = r1.get_height()
    h2 = r2.get_height()
    h3 = r3.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", color="white", fontsize=5, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="white", fontsize=5, fontweight="bold")
    plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., "%d" % h3, ha="center", va="center", color="white", fontsize=5, fontweight="bold")

plt.show()


print(results)
