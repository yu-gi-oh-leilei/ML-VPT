import json

images = json.load(open('objects.json', 'r'))

DICT = {}
for image in images:
    for object in image['objects']:
        name = object['names']
        for n in name:
            count = DICT.get(n, 0)
            DICT[n] = count + 1

SYN = {}
with open('object_alias.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        l = line.strip().split(',')
        for a in l[1:]:
            SYN[a] = l[0]
print(len(DICT))

for k, v in SYN.items():
    if k in DICT.keys():
        count = DICT[k]
        del DICT[k]
        DICT[v] = DICT.get(v, 0) + count
print(len(DICT))

keys = sorted(DICT, key=lambda x: DICT[x], reverse=True)

keys = keys[0:200]
with open('vg_labels.txt', 'w') as f:
    for key in keys:
        f.write(f"{key}\n")