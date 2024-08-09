import matplotlib.pyplot as plt

# Beispiel-Array
data = [10, 20, 15, 30, 25, 40, 35]

# Indizes für die x-Achse
indices = range(len(data))

# Plotten des Arrays

plt.plot(indices, data, marker='o')
plt.xlabel('Index')
plt.ylabel('Wert')
plt.title('Array Plot1')
plt.grid(True)
plt.savefig('test.png')
plt.show()

plt.plot(indices, data, marker='o')
plt.xlabel('Index')
plt.ylabel('Wert')
plt.title('Array Plot2')
plt.savefig('test2.png')
plt.show()

plt.plot(indices, data, marker='o')
plt.xlabel('Index')
plt.ylabel('Wert')
plt.title('Array Plot3')
plt.savefig('test3.png')
plt.show()
