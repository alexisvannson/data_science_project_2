import part2Module as m
import sys
mat=m.generate_random_matrix(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
ratingMatrix=m.matrix_transormation(mat,int(sys.argv[5]))
U,s,Vh=m.comparison(ratingMatrix)
m.outputSVD(s,U,Vh,float(sys.argv[6]))
# print(m.recommendationAlgorithm(3,Vh,8))
