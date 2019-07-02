

import csv

with open("enem_2018_small.csv", "r") as result:

    foo = 0

    for line in result:

        foo += 1

    print(foo)