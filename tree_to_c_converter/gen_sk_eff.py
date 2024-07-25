import os
import pickle
import subprocess
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def arr2cstr(a):
    """
    Converts a numpy array to a C-style string.
    """
    return np.array2string(a,
        separator=',',
        floatmode='unique',
        threshold = 1e6,
        max_line_width = 1e6).replace('\n','').replace(' ', '').replace(',', ', ').replace('[','{ ').replace(']',' }')


def model_read(model_path):
    with open(model_path, 'rb') as f:
        model_big = pickle.load(f)

    model = {}
    for i, tree_group in enumerate(model_big.estimators_):
        model[f"class{i}"] = []
        for j, tree in enumerate(tree_group.estimators_):
            tree = {
                "left_children": tree.tree_.children_left,
                "right_children": tree.tree_.children_right,
                "split_indices": tree.tree_.feature,
                "split_conditions": tree.tree_.threshold,
                "output_value": tree.tree_.value.squeeze()
            }

            new_indices = []
            for k, node in enumerate(tree["left_children"]):
                if node >= 0:
                    new_indices.append(k)

            leaf_nodes = []
            parent_indices = []
            leaf_nodes_count = 0
            for k, node in enumerate(tree["left_children"]):
                if node >= 0 and tree["left_children"][node] < 0:
                    leaf_nodes.append(tree["output_value"][node])
                    leaf_nodes_count += 1
                    tree["left_children"][k] = leaf_nodes_count-1 + 2**14
                    parent_indices.append(k)

            for k, node in enumerate(tree["right_children"]):
                if node >= 0 and tree["right_children"][node] < 0:
                    leaf_nodes.append(tree["output_value"][node])
                    leaf_nodes_count += 1
                    tree["right_children"][k] = leaf_nodes_count-1 + 2**14
                    parent_indices.append(k)

            # Remove the under 0s
            tree["split_conditions"] = [x for k, x in enumerate(tree["split_conditions"]) if tree["left_children"][k] >= 0 and tree["right_children"][k] >= 0]
            tree["split_indices"] = [x for x in tree["split_indices"] if x >= 0]

            new_lefts = []
            for k, x in enumerate(tree["left_children"]):
                if x >= 0:
                    if not k in parent_indices:
                        new_lefts.append(new_indices.index(x))
                    else:
                        new_lefts.append(tree["left_children"][k])
            tree["left_children"] = new_lefts

            new_rights = []
            for k, x in enumerate(tree["right_children"]):
                if x >= 0:
                    if not k in parent_indices:
                        new_rights.append(new_indices.index(x))
                    else:
                        new_rights.append(tree["right_children"][k])
            tree["right_children"] = new_rights

            tree["output_value"] = leaf_nodes
            model[f"class{i}"].append(tree)
    
    return model


def exportTree(model, model_path):
    gen_file = open('tree.h', 'w')
    gen_file.write(f"// GENERATED FILE FROM MODEL {model_path}\n")
    gen_file.write("#ifndef __GEN_TREE__\n")
    gen_file.write("#define __GEN_TREE__\n\n")

    INPUT_SIZE = max(model["class0"][0]["split_indices"])+1
    OUTPUT_SIZE = len(model.keys())

    gen_file.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
    gen_file.write(f"#define OUTPUT_SIZE {OUTPUT_SIZE}\n\n")
    gen_file.write("void tree_forward(float input[INPUT_SIZE], float output[OUTPUT_SIZE]);\n\n")
    gen_file.write(f"#endif\n")

    exportname = os.path.splitext(os.path.basename(model_path))[0]
    gen_file = open('tree.c', 'w')
    gen_file.write(f"// GENERATED FILE FROM MODEL {model_path}\n")
    gen_file.write(f'#include "tree.h"\n')
    gen_file.write(f'#include "tree_utils.h"\n\n\n')

    values = ["left_children", "right_children", "split_indices", "split_conditions", "output_value"]
    result = "struct DecisionTreeEnsemble\n{\n"
    for i, tree_group in enumerate(model.keys()):
        result += f"    // Output {i}\n"
        for j, tree in enumerate(model[tree_group]):

            node_count = len(tree["split_conditions"])
            leaf_count = len(tree["output_value"])

            for v in values:
                if v == "split_conditions":
                    result += f"    float {v}_{i}_{j}[{node_count}];\n"
                elif v == "output_value":
                    result += f"    float {v}_{i}_{j}[{leaf_count}];\n"
                elif v == "split_indices":
                    result += f"    __UINT8_TYPE__ {v}_{i}_{j}[{node_count}];\n"
                else:
                    result += f"    __INT16_TYPE__ {v}_{i}_{j}[{node_count}];\n"
            
            if not (i == len(model.keys())-1 and j == len(model[tree_group])-1): result += "\n"
    result += "};\n\n"

    print("Loading weights...")
    result += f"struct DecisionTreeEnsemble {exportname} = {{\n"
    for i, tree_group in enumerate(model.keys()):
        result += f"    // Output {i}\n"
        for j, tree in enumerate(model[tree_group]):
            result += f"    .{values[0]}_{i}_{j} = {arr2cstr(np.array(tree['left_children']))},\n"
            result += f"    .{values[1]}_{i}_{j} = {arr2cstr(np.array(tree['right_children']))},\n"
            result += f"    .{values[2]}_{i}_{j} = {arr2cstr(np.array(tree['split_indices']))},\n"
            result += f"    .{values[3]}_{i}_{j} = {arr2cstr(np.array(tree['split_conditions']))},\n"
            result += f"    .{values[4]}_{i}_{j} = {arr2cstr(np.array(tree['output_value']))},\n"
            if not (i == len(model.keys())-1 and j == len(model[tree_group])-1): result += "\n"
        print(f"{((i+1)/len(model.keys())*100):.2f}% Done")
    result += "};\n\n"

    print("Building feed forward function...")
    result += "void tree_forward(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {\n"
    for i, tree_group in enumerate(model.keys()):
        result += f"    output[{i}] = 0;\n"

        for j, tree in enumerate(model[tree_group]):
            result += f"    output[{i}] += tranverse_compact_tree({exportname}.left_children_{i}_{j}, {exportname}.right_children_{i}_{j}, {exportname}.split_indices_{i}_{j}, {exportname}.split_conditions_{i}_{j}, {exportname}.output_value_{i}_{j}, input);\n"

        result += f"    output[{i}] /= {len(model[tree_group])}.f;\n"
        if not (i == len(model.keys())-1 and j == len(model[tree_group])-1): result += "\n"
        print(f"{((i+1)/len(model.keys())*100):.2f}% Done")
    result += "}\n"
    gen_file.write(result)


if __name__ == '__main__':
    model_path = "../new_model_gen/random_forest_regressor_small.pkl"
    model = model_read(model_path)
    exportTree(model, model_path)

    # Check if the model outputs match
    print("Comparing with original...")
    process = subprocess.Popen('gcc main.c tree.c tree_utils.c -o p', shell=True)
    process.wait()
    process = subprocess.Popen('./p', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    c_output, _ = process.communicate()
    c_output = c_output.decode('utf-8')
    c_output = np.array(eval(c_output))

    with open(model_path, 'rb') as f:
        sk_model = pickle.load(f)
    test_data = np.load(f"../pth_to_c_converter/test_data.npz")["array"]
    py_output = sk_model.predict(test_data)

    same = True
    output_size = len(c_output[0])
    for i in range(len(c_output)):
        for j in range(output_size):
            if np.abs(c_output[i][j]-py_output[i][j]) >= 1e-4:
                same = False
                break
        if not same:
            break

    if same:
        print("Models give the same outputs!")
    else:
        print("Something went wrong, outputs dont match!")
