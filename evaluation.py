import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR[:, 1:, :, :]
    GT = GT[:, 1:, :, :]
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR.byte()==GT.byte())
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    #print(f'acc: {SR.any()}, {GT.any()}')
    #print(f'acc: {corr}, {tensor_size}')

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR[:, 1:, :, :]
    GT = GT[:, 1:, :, :]
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).byte()+(GT==1).byte())==2
    FN = ((SR==0).byte()+(GT==1).byte())==2

    SE = float(torch.sum(TP))/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR[:, 1:, :, :]
    GT = GT[:, 1:, :, :]
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).byte()+(GT==0).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    SP = float(torch.sum(TN))/(float(torch.sum(torch.sum(TN)+torch.sum(FP))) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR[:, 1:, :, :]
    GT = GT[:, 1:, :, :]
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).byte()+(GT==1).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR[:, 1:, :, :]
    GT = GT[:, 1:, :, :]
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR[:, 1:, :, :]
    GT = GT[:, 1:, :, :]
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR.byte()+GT.byte())==2)
    Union = torch.sum((SR.byte()+GT.byte())>=1)
    #print(f'JS: {SR.any()}, {GT.any()}')
    #print(f'JS: {Inter}, {Union}')
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR[:, 1:, :, :]
    GT = GT[:, 1:, :, :]
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte()+GT.byte())==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



