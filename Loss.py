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


class YOLO_LOSS(nn.Module):
    def __init__(self,THRESHOLD):
        super(YOLO_LOSS,self).__init__()
        self.THRESHOLD  =   THRESHOLD
        self.img_size   =   416
        self.class_n    =   5
    def IoU(self,target):
        anchor      =   [[30,61],[62,45],[59,119],[10,13],[16,30],[33,23],[116,90],[156,198],[373,326] ]
        iou_list    =   []
        for p in anchor:
            if(p[0] > target[0]):
                if(p[1] > target[1]):
                    BB_p =   np.zeros((p[1],p[0]))
                    BB_t =   np.zeros((p[1],p[0]))
                else:
                    BB_p =   np.zeros((target[1],p[0]))
                    BB_t =   np.zeros((target[1],p[0]))
            else:
                if(p[1] > target[1]):
                    BB_p =   np.zeros((p[1],target[0]))
                    BB_t =   np.zeros((p[1],target[0]))
                else:
                    BB_p =   np.zeros((target[1],target[0]))
                    BB_t =   np.zeros((target[1],target[0]))
            BB_p[0:p[1],0:p[0]]             =   1
            BB_t[0:target[1],0:target[0]]   =   1
            TP  =   np.sum(BB_p == BB_t)
            TN  =   np.sum((BB_p == 0) == (BB_t==0))
            FN  =   np.sum((BB_p == 0) == BB_t)
            iou =   TP /(TP + TN + FN)
            iou_list.append(iou)
        return iou_list
    def forward(self, y_scale1 , y_scale2 , y_scale3 ,t_list):
        y       =   [y_scale1 , y_scale2 , y_scale3]
        p       =   [[30,61],[62,45],[59,119],[10,13],[16,30],[33,23],[116,90],[156,198],[373,326] ]
        #print(y_scale1.shape , t_scale1.shape)
        eps     =   1e-2
        Loss    =   torch.tensor(0,dtype = torch.float32).to('cuda')
        L_obj   =   1
        L_noobj =   100
        L_coord =   100
        Loss_class  =   torch.tensor(0,dtype = torch.float32).to('cuda')
        Loss_noobj  =   torch.tensor(0,dtype = torch.float32).to('cuda')
        Loss_obj    =   torch.tensor(0,dtype = torch.float32).to('cuda')
        Loss_coord  =   torch.tensor(0,dtype = torch.float32).to('cuda')
        for batch in range(y_scale1.shape[0]):
            for y_scale in y:
                #print(t_scale.shape , y_scale.shape)
                y_boxes   =   []
                for n in range(3):
                    y_boxes.append(y_scale[batch,n*10:(n+1)*10 , :,:])
                target      =   np.array([int(t_list[batch][2] * self.img_size) , int(t_list[batch][3] * self.img_size)])
                #print(target.shape)     
                iou_list    =   self.IoU(target)
                for i , y_box in enumerate(y_boxes):
                    tx  =   t_list[batch,0]
                    ty  =   t_list[batch,1]
                    tw  =   t_list[batch,2]
                    th  =   t_list[batch,3]         
                    #print(tx,ty,tw,th)
                    N                               =   y_box.shape[2]
                    _tx                             =   tx * N
                    _ty                             =   ty * N
                    Cx                              =   int(math.modf(_tx)[1])
                    Cy                              =   int(math.modf(_ty)[1])
                    _tx                             =   (math.modf(_tx)[0])   
                    _ty                             =   (math.modf(_ty)[0])
                    #print(N,Cx,Cy,_tx,_ty)
                    class_box                       =   torch.zeros((self.class_n)).to('cuda')
                    class_box[int((t_list[batch,4]))]    =   1
                    if(iou_list[i] == max(iou_list)):
                        t_bb        =   torch.from_numpy(np.array([_tx,_ty,tw,th]).astype("float32")).to('cuda')
                        yx          =   torch.sigmoid(y_box[0,Cy,Cx])
                        yy          =   torch.sigmoid(y_box[1,Cy,Cx])
                        yw          =   y_box[2,Cy,Cx]
                        yh          =   y_box[3,Cy,Cx]
                        y_bb        =   torch.from_numpy(np.array([yx,yy,yw,yh]).astype("float32")).to('cuda')
                        #print(nn.MSELoss(reduction = 'sum')(y_bb,t_bb))
                        Loss_coord  +=   L_coord*nn.MSELoss(reduction = 'sum')(y_bb,t_bb)
                        #print(y_box[4,Cy,Cx])
                        Loss_obj    +=   L_obj * -1 * torch.sigmoid(y_box[4,Cy,Cx]).log() 
                        
                        Loss_class  +=   nn.BCEWithLogitsLoss(y_box[5:,Cy,Cx] , class_box ) 
                        Loss_noobj  +=   L_noobj*(torch.sum(-1 * (1 - torch.sigmoid(y_box[4,:,:] + eps)).log()) + torch.sum((1 -  torch.sigmoid(y_box[4,Cy,Cx])+eps).log()))
                        #print(Loss_coord,Loss_class,Loss_noobj,Loss_obj)

                    else:    
                        Loss_noobj  +=   L_noobj*(torch.sum(-1 * (1 - torch.sigmoid(y_box[4,:,:] + eps)).log()))
                        
        Loss        =  Loss_coord + Loss_class + Loss_noobj + Loss_obj
        #print(Loss)
        return Loss , Loss_coord , Loss_class , Loss_noobj , Loss_obj


class TINY_YOLO_LOSS(nn.Module):
    def __init__(self,THRESHOLD):
        super(TINY_YOLO_LOSS,self).__init__()
        self.THRESHOLD  =   THRESHOLD
        self.img_size   =   416
        self.class_n    =   5
        self.out        =   (4 + 1 + self.class_n)
    def IoU(self,target):
        anchor      =   [[40, 40], [98, 97], [150, 164], [199, 287], [225, 184], [311, 212]]
        iou_list    =   []
        for p in anchor:
            if(p[0] > target[0]):
                if(p[1] > target[1]):
                    BB_p =   np.zeros((p[1],p[0]))
                    BB_t =   np.zeros((p[1],p[0]))
                else:
                    BB_p =   np.zeros((target[1],p[0]))
                    BB_t =   np.zeros((target[1],p[0]))
            else:
                if(p[1] > target[1]):
                    BB_p =   np.zeros((p[1],target[0]))
                    BB_t =   np.zeros((p[1],target[0]))
                else:
                    BB_p =   np.zeros((target[1],target[0]))
                    BB_t =   np.zeros((target[1],target[0]))
            BB_p[0:p[1],0:p[0]]             =   1
            BB_t[0:target[1],0:target[0]]   =   1
            TP  =   np.sum(BB_p == BB_t)
            TN  =   np.sum((BB_p == 0) == (BB_t==0))
            FN  =   np.sum((BB_p == 0) == BB_t)
            iou =   TP /(TP + TN + FN)
            iou_list.append(iou)
        return iou_list
    def forward(self, y_scale1 , y_scale2 ,t_list):
        counter =   1
        y       =   [y_scale1 , y_scale2 ]
        p       =   [[40, 40], [98, 97], [150, 164], [199, 287], [225, 184], [311, 212]]
        #print(y_scale1.shape , t_scale1.shape)
        eps     =   1e-16
        Loss    =   torch.tensor(0,dtype = torch.float32).to('cuda')
        L_obj   =   1
        L_noobj =   1
        L_coord =   1
        Loss_class  =   torch.tensor(0,dtype = torch.float32).to('cuda')
        Loss_noobj  =   torch.tensor(0,dtype = torch.float32).to('cuda')
        Loss_obj    =   torch.tensor(0,dtype = torch.float32).to('cuda')
        Loss_coord  =   torch.tensor(0,dtype = torch.float32).to('cuda')
        for batch in range(y_scale1.shape[0]):
            target      =   np.array([int(t_list[batch][2] * self.img_size) , int(t_list[batch][3] * self.img_size)])     
            iou_list    =   self.IoU(target)
            y_boxes   =   []
            #print(batch,iou_list)
            for y_scale in y:
                for n in range(3):
                    y_boxes.append(y_scale[batch,n*self.out:(n+1)*self.out , :,:])

            for i , y_box in enumerate(y_boxes):
                tx  =   t_list[batch,0]
                ty  =   t_list[batch,1]
                tw  =   t_list[batch,2] 
                th  =   t_list[batch,3]          
                #print(tx,ty,tw,th)
                N                               =   y_box.shape[2]
                _tx                             =   tx * N
                _ty                             =   ty * N
                Cx                              =   int(math.modf(_tx)[1])
                Cy                              =   int(math.modf(_ty)[1])
                _tx                             =   (math.modf(_tx)[0])   
                _ty                             =   (math.modf(_ty)[0])
                class_box                       =   torch.zeros((self.class_n)).to('cuda')
                class_box[int((t_list[batch,4]))]    =   1
                if(iou_list[i] == max(iou_list) or iou_list[i] > self.THRESHOLD):
                    tw                          =   torch.log(tw*self.img_size / p[i][0] + eps)   
                    th                          =   torch.log(th*self.img_size / p[i][1] + eps)
                    t_bb        =   torch.from_numpy(np.array([_tx,_ty,tw,th]).astype("float32")).to('cuda')
                    yx          =   torch.sigmoid(y_box[0,Cy,Cx])
                    yy          =   torch.sigmoid(y_box[1,Cy,Cx])
                    yw          =   y_box[2,Cy,Cx]
                    yh          =   y_box[3,Cy,Cx]
                    y_bb        =   torch.from_numpy(np.array([yx,yy,yw,yh]).astype("float32")).to('cuda')
                    
                    obj_mask            =   torch.zeros((N,N)).to('cuda')
                    obj_mask[Cy,Cx]   =   1
                    

                    
                    Loss_coords =   L_coord*nn.MSELoss(reduction = 'none')(y_bb,t_bb)
                    x_loss      =   Loss_coords[0]
                    y_loss      =   Loss_coords[1]
                    w_loss      =   Loss_coords[2]
                    h_loss      =   Loss_coords[3]
                    #print(x_loss , y_loss , w_loss , h_loss)
                    Loss_coord  +=   x_loss  +   y_loss + w_loss + h_loss
                    Loss_obj    +=   L_obj*nn.BCELoss()(torch.sigmoid(y_box[4,:,:]) , obj_mask)
                    Loss_class  +=   nn.BCELoss()(torch.sigmoid(y_box[5:,Cy,Cx]) , class_box ) 
                    #Loss_noobj  +=   L_noobj*nn.BCELoss()(torch.sigmoid(y_box[4,:,:]) , noobj_mask)
                else:
                    noobj_mask  =   torch.zeros((N,N)).to('cuda')
                    Loss_noobj  +=   L_noobj*nn.BCELoss()(torch.sigmoid(y_box[4,:,:]) , noobj_mask)
            Loss        =  Loss_coord + Loss_class + Loss_noobj + Loss_obj
        #print(Loss)
        return Loss , Loss_coord , Loss_class , Loss_noobj , Loss_obj