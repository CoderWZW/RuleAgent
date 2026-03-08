import random
train = []
test = []
val=[]

# test.append(line)

with open('Gowalla_train.txt') as f:
    for line in f:
        train.append(line)

with open('Gowalla_test.txt') as f:
    for line in f:
        test.append(line)

with open('Gowalla_tune.txt') as f:
    for line in f:
        val.append(line)

with open('train.txt','w') as f:
    for line in train:
        item = line.split('\t')
        f.write(item[0]+' '+item[1].replace("\n", "")+' '+'1'+'\n')

with open('test.txt','w') as f:
    for line in test:
        item = line.split('\t')
        f.write(item[0]+' '+item[1].replace("\n", "")+' '+'1'+'\n')

with open('val.txt','w') as f:
    for line in val:
        item = line.split('\t')
        f.write(item[0]+' '+item[1].replace("\n", "")+' '+'1'+'\n')