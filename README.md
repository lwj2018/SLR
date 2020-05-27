# SLR
Isolated & Continuous sign language recognition based on skeleton data

## Isolated Sign Language Recognition

### LSTM

1. **Top-1 Acc & Top-5 Acc on CSL_Isolated**
- After about 200 epoch

   | Methods                | Best Top-1 Acc     | Best Top-5 Acc    |
   | ---------------------- | ------------------ | ----------------- |
   | LSTM                   | 69.81%             | 91.42%            |
   | HCN                    | 73.56%             | 94.39%            |
   | HCN + HierarchicalConv | 84.24%             | 97.00%            |

2. **Comparison between normal and bad situation**

   | Methods                | normal             | bad               |
   | ---------------------- | ------------------ | ----------------- |
   | HCN + HierarchicalConv | 81.72%             | 66.60%            |


## Continuous Sign Language Recognition

### HCN+LSTM

1. **using original skeleton**

   | Dataset        | Split    | Samples | Best Wer     | Best Bleu     |
   | -------------- | -------- | ------- | ------------ | ------------- |
   | CSL_Continuous | Train    | 20,000  | 10.593%      | 97.910%       |
   | CSL_Continuous | Val      |  5,000  | 13.569%      | 93.343%       |

