"""
Computes the F1 score on BIO tagged data

@author: Nils Reimers
"""


#Method to compute the accruarcy. Call predict_labels to get the labels for the dataset
def compute_f1(predictions, dataset_y, idx2Label): 
    
    
    label_y = [idx2Label[element] for element in dataset_y]
    pred_labels = [idx2Label[element] for element in predictions]
    


    prec = compute_precision(pred_labels, label_y)
    rec = compute_precision(label_y, pred_labels)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1


def compute_precision(guessed, correct):
    correctCount = 0
    count = 0
    
    idx = 0
    while idx < len(guessed):
        if guessed[idx][0] == 'B': #A new chunk starts
            count += 1
            
            if guessed[idx] == correct[idx]:
                idx += 1
                correctlyFound = True
                
                while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False
                    
                    idx += 1
                
                if idx < len(guessed):
                    if correct[idx][0] == 'I': #The chunk in correct was longer
                        correctlyFound = False
                    
                
                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
        else:  
            idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision