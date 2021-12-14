import random
f1 = open("test.txt", "w")
for i in range(5):
    f1.write(str(random.random()) + "\n")
f1.close()