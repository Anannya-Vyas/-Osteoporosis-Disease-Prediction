import json
import os

def inspect_notebook(nb_path, out_file):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    with open(out_file, 'w', encoding='utf-8') as out:
        out.write(f"INSPECTING NOTEBOOK: {nb_path}\n")
        out.write(f"Total cells: {len(nb['cells'])}\n\n")
        for idx, cell in enumerate(nb['cells']):
            src = "".join(cell.get('source', []))
            cell_type = cell.get('cell_type')
            lower_src = src.lower()
            keywords = ['feature', 'select', 'gridsearch', 'randomsearch', 'extratrees', 'hyperparameter', 'columns', 'cleaned_dataset', 'best_model', 'params', 'drop', 'train_test_split']
            found = [k for k in keywords if k in lower_src]
            if found:
                out.write(f"--- Cell {idx} ({cell_type}) [Matched: {found}] ---\n")
                out.write(src.strip() + "\n\n")

if __name__ == '__main__':
    inspect_notebook('models/ML_MODEL(WITH DEXA)/model_training.ipynb', 'inspect_dexa_training_out.txt')
    inspect_notebook('models/ML_MODEL(WITHOUTDEXA)/ml_model(withoutdexa).ipynb', 'inspect_withoutdexa_out.txt')
    print("Done inspecting.")

