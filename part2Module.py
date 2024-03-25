import numpy as np
import scipy
import time
import matplotlib.pyplot as mpl

def matrix_transormation(matrix,threshold):
    transform =lambda x:1 if x>threshold else 0

    transformer=np.vectorize(transform)
    return(transformer(matrix))
def generate_random_matrix(minval,maxval,rows,columns):

    mat=np.random.random_integers(minval,maxval,(rows,columns))
    return mat
#Initialise random matrix where Rows>>Columns
#mat=generate_random_matrix(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
#Use matrix_transformation to convert to binary matrix
#ratingMatrix=matrix_transormation(mat,int(sys.argv[5]))

def comparison(ratingMatrix):
    times=[]
    first=time.time()
    U,s,Vh=np.linalg.svd(ratingMatrix)
    second=time.time()
    times.append(second-first)

    first=time.time()
    U,s,Vh=scipy.linalg.svd(ratingMatrix)
    second=time.time()
    times.append(second-first)
    objects=('Numpy','Scipy')
    ypos=np.arange(len(objects))
    mpl.bar(ypos,times,align='center')
    mpl.xticks(ypos,objects)
    mpl.ylabel('Times')
    mpl.title('NumpySVD vs ScipySVD')
    mpl.show()
    #From this graph we cans see that Scipy is much faster
    #at calculating SVD than Numpy


    mpl.scatter(range(0,20),s[:20],marker='o',label="Singular Values")
    mpl.xscale('linear')
    mpl.yscale('log')
    mpl.grid(True)
    mpl.ylabel('Singular Value')
    mpl.xlabel('Index')
    mpl.legend()
    mpl.show()
    return U,s,Vh
#From this graph we can see the singular value drop in magnitude
#After the first value so we will only use the first one
# U,s,Vh=comparison(ratingMatrix)



    
    
    
    

    

def outputSVD(s,U,Vh,target):
    #Look at the difference between the singular values and if the change is small,then it stops iterating
    for i in range(len(s)):
        if s[i+1]/s[i]>=target:
            index=i
            break
    number_of_singularValues=index
    print("Singular Values:")
    print(s[:number_of_singularValues]+1)
    print("U")
    print(U)
    print("Vh")
    print(Vh)
    return




def recommendationAlgorithm(liked,VT,Selected):
    recommended=[]
    for i in range(VT.shape[0]):
        if i != liked:
            recommended.append([i,VT[i].dot(VT[liked])])
    final=sorted(recommended)
    return final[:Selected]

