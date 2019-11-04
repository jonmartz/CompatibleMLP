import matplotlib.pyplot as plt

h1_x = [0.6, 1.0]
h1_y = [0.6, 0.6]
plt.plot(h1_x, h1_y, 'k--', label='h1')

h2_x = list((x/50+0.6 for x in range(20)))
h2_old_y = list((0.9 - (x / 40) ** 2 for x in range(20)))
h2_new_y = list((0.9 - (x / 60) ** 2 for x in range(20)))
plt.plot(h2_x, h2_old_y, 'b', marker='.', label='h2')
plt.plot(h2_x, h2_new_y, 'r', marker='.', label='h2')
plt.xlabel('compatibility')
plt.ylabel('accuracy')
plt.legend(('initial model', 'updated model', "new updated model"), loc='center left')
plt.title('Performance / Compatibility tradeoff for a specific user')
plt.show()