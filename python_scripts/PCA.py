import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#import files
filenameSimple = "add_data_2.dat"
filenameComplex = "epi_data_2.dat"

ngenes = 400

# read file for simple data
with open(filenameSimple, "rb") as binary_file:
    data = binary_file.read()
all_data = np.frombuffer(data, np.float64)
all_data = all_data[:120000]

# read file for complex data
with open(filenameComplex, "rb") as binary_file:
    data = binary_file.read()
all_data2 =  np.frombuffer(data, np.float64)   


# append complex data to simple
all_data = np.append(all_data, all_data2)

# reshape so each row is one simulation
all_data = np.reshape(all_data, (600,ngenes))


# make label array, first half simple (0), second half complex (1)
all_labels = np.full((600,2), 0)
all_labels[:300 , 0].fill(1)
all_labels[300: , 1].fill(1)

#Principal component analysis
#Calculation of principal components and eigen values
centered_data = all_data - all_data.mean(0)
A = np.asmatrix(centered_data.T) * np.asmatrix(centered_data)
U, S, V = np.linalg.svd(A) 
eigvals = S**2 / np.sum(S**2)

#Construction of scree plot
fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(400) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.xlim(0,50)
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()