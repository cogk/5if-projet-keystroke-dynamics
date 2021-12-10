
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from models.LDA import get_trained_model
from data_loader import get_X_y

m = get_trained_model()

X, y = get_X_y()

confusion = confusion_matrix(y, m.predict(X.values))

# set zeros on diagonal
confusion[np.diag_indices_from(confusion)] = -1

# confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

f, axarr = plt.subplots(2, 2)

ax = axarr[0, 0]
plt.sca(ax)
ax.matshow(confusion)
# plt.xticks(range(len(m.classes_)), m.classes_, fontsize=10, rotation=90)
# plt.yticks(range(len(m.classes_)), m.classes_, fontsize=10)
# plt.annotate('Diagonal values are set to -1 to increase contrast',
#              xy=(0.5, 0.1), xycoords='figure fraction', horizontalalignment='center',)

ax = axarr[0, 1]
plt.sca(ax)
ax.matshow(confusion)
# plt.xticks(range(len(m.classes_)), m.classes_, fontsize=10, rotation=90)
# plt.yticks(range(len(m.classes_)), m.classes_, fontsize=10)
# plt.annotate('Diagonal values are set to -1 to increase contrast',
#              xy=(0.5, 0.1), xycoords='figure fraction', horizontalalignment='center',)

ax = axarr[1, 0]
plt.sca(ax)
ax.matshow(confusion)
# plt.xticks(range(len(m.classes_)), m.classes_, fontsize=10, rotation=90)
# plt.yticks(range(len(m.classes_)), m.classes_, fontsize=10)
# plt.annotate('Diagonal values are set to -1 to increase contrast',
#              xy=(0.5, 0.1), xycoords='figure fraction', horizontalalignment='center',)

ax = axarr[1, 1]
plt.sca(ax)
ax.matshow(confusion)
# plt.xticks(range(len(m.classes_)), m.classes_, fontsize=10, rotation=90)
# plt.yticks(range(len(m.classes_)), m.classes_, fontsize=10)
# plt.annotate('Diagonal values are set to -1 to increase contrast',
#              xy=(0.5, 0.1), xycoords='figure fraction', horizontalalignment='center',)

f.tight_layout()
plt.colorbar()
plt.show()
