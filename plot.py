import json
import matplotlib.pyplot as plt
import numpy as np

results = []
with open('result_datas/all_result.json', 'r') as j:
    results = json.load(j)

assemble = [float(x["ASSEMBLE"])/100 for x in results]
compute = [float(x["COMPUTE"])/100 for x in results]
solve = [float(x["SOLVE"])/100 for x in results]

bars = np.add(assemble, compute).tolist()

N = 5

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, assemble, width)
p2 = plt.bar(ind, compute, width, bottom=assemble)
p3 = plt.bar(ind, solve, width, bottom=bars)

plt.ylabel('Time')
plt.title('Method')
plt.xticks(ind, [x["name"] for x in results])
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0], p3[0]), ('Assemble', 'Compute', 'Solve'))

plt.show()


print(results)
