import sys


f = open('./data/train/train_source.txt')
lines = f.readlines()

for line in lines:
    elems = line.split(" ")
    for elem in elems:
        print(chr(int(elem)))