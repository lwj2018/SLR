import pickle
import sys
sys.path.append('..')
from datasets import CSL_Isolated_Openpose_fl
out_file = '../obj/word2gloss.pkl'
out_file = open(out_file,'wb')
database = {}
dataset = CSL_Isolated_Openpose_fl('trainval')
for i in range(len(dataset)):
    print(f"{i}/{len(dataset)}")
    mat, lb = dataset[i]
    database[lb] = mat
pickle.dump(database,out_file)

