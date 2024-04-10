import json

data = json.load(open("tree.json", "r"))
trees_list = data["learner"]["gradient_booster"]["model"]["trees"]
print(len(trees_list))
tree_data = trees_list[0]

group_counter = 0
for tree in trees_list:
    for key, value in tree.items():
        print(key, ": ", value)
    print("----------------------------------------")
    group_counter += 1
    if group_counter > 5:
        group_counter = 0
        print("============= NEW GROUP =============")
