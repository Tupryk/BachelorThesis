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


def exportTree(model_path):
    gen_file = open('tree.h', 'w')
    gen_file.write(f"// GENERATED FILE FROM MODEL {model_path}\n")
    gen_file.write("#ifndef __GEN_TREE__\n")
    gen_file.write("#define __GEN_TREE__\n\n")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    INPUT_SIZE = model.estimators_[0].estimators_[0].n_features_in_
    OUTPUT_SIZE = len(model.estimators_)

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
    for i, tree_group in enumerate(model.estimators_):
        result += f"    // Output {i}\n"
        for j, tree in enumerate(tree_group.estimators_):

            node_count = tree.tree_.node_count

            for v in values:
                if v == "split_conditions" or v == "output_value":
                    result += f"    float {v}_{i}_{j}[{node_count}];\n"
                else:
                    result += f"    int {v}_{i}_{j}[{node_count}];\n"
            
            if not (i == len(model.estimators_)-1 and j == len(tree_group.estimators_)-1): result += "\n"
    result += "};\n\n"

    print("Loading weights...")
    result += f"struct DecisionTreeEnsemble {exportname} = {{\n"
    for i, tree_group in enumerate(model.estimators_):
        result += f"    // Output {i}\n"
        for j, tree in enumerate(tree_group.estimators_):
            result += f"    .{values[0]}_{i}_{j} = {arr2cstr(tree.tree_.children_left)},\n"
            result += f"    .{values[1]}_{i}_{j} = {arr2cstr(tree.tree_.children_right)},\n"
            result += f"    .{values[2]}_{i}_{j} = {arr2cstr(tree.tree_.feature)},\n"
            result += f"    .{values[3]}_{i}_{j} = {arr2cstr(tree.tree_.threshold)},\n"
            result += f"    .{values[4]}_{i}_{j} = {arr2cstr(tree.tree_.value.squeeze())},\n"
            if not (i == len(model.estimators_)-1 and j == len(tree_group.estimators_)-1): result += "\n"
        print(f"{((i+1)/len(model.estimators_)*100):.2f}% Done")
    result += "};\n\n"

    print("Building feed forward function...")
    result += "void tree_forward(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {\n"
    for i, tree_group in enumerate(model.estimators_):
        result += f"    output[{i}] = 0;\n"

        for j, tree in enumerate(tree_group.estimators_):
            result += f"    output[{i}] += tranverse_tree_sklearn({exportname}.left_children_{i}_{j}, {exportname}.right_children_{i}_{j}, {exportname}.split_indices_{i}_{j}, {exportname}.split_conditions_{i}_{j}, {exportname}.output_value_{i}_{j}, input);\n"

        result += f"    output[{i}] /= {len(tree_group.estimators_)}.f;\n"
        if not (i == len(model.estimators_)-1 and j == len(tree_group.estimators_)-1): result += "\n"
        print(f"{((i+1)/len(model.estimators_)*100):.2f}% Done")
    result += "}\n"
    gen_file.write(result)


if __name__ == '__main__':
    model_path = "../new_model_gen/random_forest_regressor_small.pkl"
    exportTree(model_path)

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
    print(c_output[0])
    print(py_output[0])

    same = True
    for i in range(len(c_output)):
        for j in range(6):
            if np.abs(c_output[i][j]-py_output[i][j]) >= 1e-4:
                same = False
                break
        if not same:
            break

    if same:
        print("Models give the same outputs!")
    else:
        print("Something went wrong, outputs dont match!")
