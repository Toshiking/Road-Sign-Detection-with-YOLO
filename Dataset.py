import  torch
import  torchvision.datasets    as  dset
import  torchvision.transforms  as  transforms
import  model
import  torch.nn                as  nn
import  torch.nn.functional     as  F
import  torch.optim             as  optim
import  torchvision
import  os
import  glob
import  cv2
import  numpy                   as  np
import  math

class dataset(torch.utils.data.Dataset):
    def __init__(self , x_path , y_path , transform = None , limit = 0 , class_n = 6):
        self.transform  =   transform
        self.x_list     =  sorted(glob.glob(os.path.join(x_path,'*.bmp')))        
        self.y_list     =  sorted(glob.glob(os.path.join(y_path,'*.txt')))
        #print(len(self.x_list),len(self.y_list))
        self.class_n    =   class_n

    def __getitem__(self , index):
        x_image         =   self.x_list[index]
        X_image         =   cv2.imread(x_image)
        x_image         =   (X_image/255).astype('float32')
        y_image         =   self.y_list[index]
        with open(y_image) as f:
            s_line = f.readline()
        s_line  =   s_line.split(',')
        #print(self.y_list[index])
        
        class_idx       =   int(s_line[0])
        tx              =   np.float32(s_line[1])
       
        ty              =   np.float32(s_line[2])
        tw              =   np.float32(s_line[3])
        th              =   np.float32(s_line[4])
        ans_list        =   []
        ans_list    =   [tx,ty,tw,th,class_idx]
        ans_list    =   np.array(ans_list)
        if self.transform is not None:
            x_image     =   self.transform(X_image)
        return x_image , ans_list
    def __len__(self):
        return len(self.x_list)
