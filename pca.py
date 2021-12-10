from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, figure, get_cmap, colorbar, show
from sklearn.decomposition import PCA
from data_loader import get_X_y, all_columns

X, y = get_X_y()
n_samples = X.shape[0]

X_H = X[filter(lambda x: x.startswith('H.'), all_columns)]
X_DD = X[filter(lambda x: x.startswith('DD.'), all_columns)]

X = X_H

pca = PCA(n_components=3)
X_transformed = pca.fit_transform(X)
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca.explained_variance_
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
    print('Eigenvalue: {}'.format(eigenvalue))
    # print('Eigenvector: {}'.format(eigenvector))
    print('\n'.join(eigenvector.astype(str)))
    # print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    # print(eigenvalue)
    print()
    print()
    print()
    print()

# exit()

# use LDA to reduce dimensionality
# lda_model = LinearDiscriminantAnalysis(n_components=2)
# lda_model.fit(X, y)
# row = X.iloc[16, :]
# y_pred = lda_model.predict([row])
# print('Predicted class: {}'.format(y_pred))
# print('Actual class: {}'.format(y.iloc[16]))
# exit()

class_num = y.max()

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
fig = figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(1, 1, 1)
c_map = get_cmap(name='jet', lut=class_num)
norm = plt.Normalize(vmin=y.min(), vmax=y.max())
scatter = ax.scatter(
    X_transformed[:, 0], X_transformed[:, 1], s=10, c=norm(y), cmap=c_map)

ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_title("PCA projection of {} people".format(class_num))
colorbar(mappable=scatter)
show()


# Find amount of variance explained by each of the selected components
pca2 = PCA().fit(X)
plot(pca2.explained_variance_, linewidth=2)
xlabel('Components')
ylabel('Explained Variaces')
show()
