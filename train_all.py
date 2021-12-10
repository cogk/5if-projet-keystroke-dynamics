from models.GradientBoosting import get_trained_model as GB_get_trained_model
from models.Manhattan import get_trained_model as M_get_trained_model
from models.SVM import get_trained_model as SVM_get_trained_model
from models.LDA import get_trained_model as LDA_get_trained_model

print('Gradient Boosting (can be slow to train)')
GB_get_trained_model()

print()
print()
print()

print('Manhattan')
M_get_trained_model()

print()
print()
print()

print('SVM')
SVM_get_trained_model()

print()
print()
print()

print('LDA')
LDA_get_trained_model()
