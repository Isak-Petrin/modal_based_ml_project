import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pickle

from data_scripts.utils import *

if __name__ == "__main__":
    
    base = Path.cwd()
    train = np.load(base / 'deepwriting_dataset' / 'deepwriting_training.npz', allow_pickle=True)
    test = np.load(base / 'deepwriting_dataset' / 'deepwriting_validation.npz', allow_pickle=True)

    all_labels = np.concatenate(train["char_labels"])

    alphabet = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.,-()/")
    alphabet.insert(0, chr(0))

    le = LabelEncoder()
    le.fit(alphabet)

    char_to_id = {ch: i for i, ch in enumerate(le.classes_)}
    id_to_char = {ch: i for ch, i in enumerate(le.classes_)}

    ids = char_to_id["W"]
    id_mask = np.isin(all_labels, ids)
    id_indices = np.nonzero(id_mask)[0]

    all_strokes = np.concatenate(train["strokes"])

    strokes_extracted = np.full(all_strokes.shape, None, dtype=object)
    strokes_extracted[id_mask] = all_strokes[id_mask]

    starts_and_ends = []

    eocs = np.concatenate(train["eoc_labels"])
    eoc_indices = np.nonzero(eocs)[0]

    for i in range(len(eoc_indices)):
        if i == 0:
            arr = [0, eoc_indices[0]]
        else:
            arr = [eoc_indices[i-1] + 1, eoc_indices[i]]
        starts_and_ends.append(arr)

    starts_and_ends = np.array(starts_and_ends)

    char_splits = [strokes_extracted[start:end+1] for start, end in starts_and_ends]

    extracted_characters = []

    for i, x in enumerate(char_splits):
        if x.size == 0:
            print(f"{i}-array empty: {x}")
            continue

        if not np.all(np.equal(x[-1], None)):
            extracted_characters.append(x)

    lst = []
    for character in extracted_characters:
        chars = []
        for arr in character:
            if np.all(arr == None):
                continue
            chars.append(arr)
        lst.append(np.asarray(chars))
        
    print("saving file")
    with open(f"data/{id_to_char[ids]}.pkl", "wb") as f:
        pickle.dump(lst, f)
