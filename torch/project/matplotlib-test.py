import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
print(matplotlib.get_backend())

plt.plot([1, 2, 3, 4],[10, 20, 25, 30])
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
plt.title("Пример графика")

plt.show()