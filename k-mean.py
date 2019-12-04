import  numpy as np
from    sklearn.cluster import  KMeans
import  sys
import  glob
import  os

if(len(sys.argv)!=4):
    print("ERROR: please write \"python k-means.py <Your Dir> <Number of Boxes> <InputSize>\"")
    exit()
PATH                =   sys.argv[1]
CLASS               =   int(sys.argv[2])
INPUT_SIZE          =   int(sys.argv[3])
TEXT_NAME_LIST      =   sorted(glob.glob(os.path.join(PATH,'*')))
TEXT_LIST           =   []

for text in TEXT_NAME_LIST:
    with open(text) as f:
        s_line = f.readline()
        s_line  =   s_line.split(',')
        TEXT_LIST.append(s_line[3:])

TEXT_LIST   =   np.array(TEXT_LIST)



pred        =   KMeans(n_clusters = CLASS).fit_predict(TEXT_LIST)
TEXT_LIST   =   np.concatenate([TEXT_LIST , pred[:,np.newaxis]] , axis = 1)

result_list =   []
sort_list   =   []

for n in range(0 , CLASS):
    width_mean  =   0
    heigth_mean =   0
    num         =   0
    for i in range(TEXT_LIST.shape[0]):
        #print(int(TEXT_LIST[i,2]) , n)
        if(int(TEXT_LIST[i,2])   ==  n):
            width_mean  +=  float(TEXT_LIST[i,0])*INPUT_SIZE
            heigth_mean  +=  float(TEXT_LIST[i,1])*INPUT_SIZE
            num         +=  1
    
    #print("Box",n,"'s anchor is [",int(width_mean/num) ,",",int(heigth_mean / num),"]")
    result_list.append([int(width_mean/num) ,int(heigth_mean / num)])    
    sort_list.append(int(width_mean/num)*int(heigth_mean / num))

print_list  =   []
for point , _ in sorted(zip(result_list , sort_list)):
    print_list.append(point)
print(print_list)