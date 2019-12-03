import  torch
import  torchvision.datasets    as  dset
import  torchvision.transforms  as  transforms
from    model                   import YOLO_v3
import  torch.nn                as  nn
import  torch.nn.functional     as  F
import  torch.optim             as  optim
import  torchvision
import  Dataset
import  numpy                   as  np
import  os
import  glob
import  cv2
import  copy
import  time
import  resize
import  sys # モジュール属性 argv を取得するため
import  traceback
from    PIL                     import  Image
import  sys
import  pyautogui
import  Loss
import  math













argvs           =   sys.argv                #コマンドライン引数を格納したリストの取得
argc            =   len(argvs)              #引数の個数
batch_size      =   2                       #バッチサイズ
workers         =   2                       #使用するプロセッサの数
n_epoch         =   1000                    #学習回数
device          =   'cuda'                  #使用するデバイス
save            =   "./save/"               #画像を保存する場所
PATH            =   "./model/"              #学習済みモデルの係数を保存する場所
traindata_x_path=   "./x_train"             #学習画像の入力画像の場所の指定
traindata_y_path=   "./y_train"             #学習用画像のラベル画像の場所の指定
testdata_x_path =   "./x_test"              #テスト画像の入力画像の場所の指定
testdata_y_path =   "./y_test"              #テスト画像のラベル画像の場所の指定
evaldata_x_path =   "./x_eval"              #評価画像の入力画像の場所の指定
evaldata_y_path =   "./y_eval"              #評価画像のラベル画像の場所の指定
real_path       =   "./real"                #学習と共にとりあえず検査したい画像の場所の指定
model_path      =   "./model/best_val.pth"  #学習済みモデルをロードする場所の指定
img_size        =   512                     #画像のサイズ
lr_decay_rate   =   0.1                     #学習率減衰割合の指定
decay_limit     =   5                       #何回うまくいかなかったら減衰させるか
init_lr         =   0.001                   #最初の学習率
CLASS           =   5
real_list       =   sorted(glob.glob(os.path.join(real_path,'*')))
RESOLURION      =   2
PREDICT_LEVEL   =   0.5 
#引数による分岐
if(argc > 1):
    if(argvs[1] == '2'):
        mode    =   2
    elif(argvs[1] == '3'):
        mode    =   3
    else:
        mode    =   0
else:
    mode    =   0


color_list  =   [(0, 0 ,0) ,(255 , 0 ,0),(0 , 255 ,0),(0 , 0 ,255) , (255, 255 ,0),(255 , 0 ,255),(0 , 255 ,255)]
#各クラスに対しての色分け



#学習用画像の読み出し
trainset    =   Dataset.dataset(
                    traindata_x_path , 
                    traindata_y_path ,
                    transform   =   transforms.Compose([
                        transforms.ToTensor()
                    ]) ,
                    class_n =   CLASS
                )   
#テスト画像の読みだし
testset     =    Dataset.dataset(
                    testdata_x_path , 
                    testdata_y_path ,
                    transform   =   transforms.Compose([
                        transforms.ToTensor()
                    ]),
                    class_n =   CLASS
                )

#データセットの読み出し定義
train_loader    =   torch.utils.data.DataLoader(trainset , batch_size = batch_size , shuffle = True , num_workers = int(workers))
test_loader     =   torch.utils.data.DataLoader(testset , batch_size = batch_size , shuffle = True , num_workers = int(workers))
criterion       =   Loss.YOLO_LOSS(THRESHOLD = 0.8)    #バイナリクリスエントロピで計算。これはとりあえず動いたから採用している。

#モデルの読み出し。今のところU-Netを定義してる stage_numは段数。nch_gは最初のフィルタ数
model           =   YOLO_v3(class_n = CLASS).to(device)

#学習済みモデルの読み出し。
try:
    model.load_state_dict(torch.load(model_path))
except:
    #モデルが読み出せなかった場合はこの文字列が表示される。
    print("最初からの学習です")

#最適化関数の定義。今のところAdam
optimizer       =   optim.Adam(model.parameters(),lr = init_lr)

#学習の定義

best_loss       =   100000000000    #一番良かった時のlossを保存
decay_counter   =   0



#学習係数の変更
def lr_decay(optim):
    for opt in optim.param_groups:
        opt['lr']   *=  lr_decay_rate
    print("lr_decay : lr={}".format(opt['lr']) )



#学習する関数。
def train(epoch):
    model.train()
    total_loss          =   0
    total_loss_coord    =   0
    total_loss_class    =   0
    total_loss_noobj    =   0
    total_loss_obj      =   0
    for itr,(data,target) in enumerate(train_loader):
        x_image     =   data.to(device)
        optimizer.zero_grad()
        output_1,output_2,output_3          =   model(x_image) 
        #print(output.shape , t_image.shape)                                       
        loss,loss_coord,loss_class,loss_noobj,loss_obj  =   criterion(output_1 , output_2 , output_3,target)   
        loss.to(device)     
        loss.backward()
        optimizer.step()
        total_loss          =   ( total_loss        * itr + loss.item()       ) /(itr + 1 )
        total_loss_class    =   ( total_loss_class  * itr + loss_class.item() ) /(itr + 1 )
        total_loss_coord    =   ( total_loss_coord  * itr + loss_coord.item() ) /(itr + 1 )
        total_loss_noobj    =   ( total_loss_noobj  * itr + loss_noobj.item() ) /(itr + 1 )
        total_loss_obj      =   ( total_loss_obj    * itr + loss_obj.item()   ) /(itr + 1 )

        print('Train : Epoch[{:>3}/{:>3}]  Iterate [{:>5}/{:>5}] Loss : {:.6f} class :{:.3f} coord :{:.3f} noobj :{:.3f} obj :{:.3f} \r'.format(epoch + 1 ,n_epoch,itr+1,len(train_loader),total_loss,total_loss_class,total_loss_coord,total_loss_noobj,total_loss_obj ) ,end = "")
#テスト関数。
def test(epoch,best_loss,decay_counter):
    model.eval()
    total_loss          =   0
    total_loss_coord    =   0
    total_loss_class    =   0
    total_loss_noobj    =   0
    total_loss_obj      =   0
    for itr,(data,target) in enumerate(test_loader):
        x_image     =   data.to(device)
        output_1,output_2,output_3          =   model(x_image) 
        #print(output.shape , t_image.shape)                                       
        loss,loss_coord,loss_class,loss_noobj,loss_obj  =   criterion(output_1 , output_2 , output_3,target)       
        total_loss          =   ( total_loss        * itr + loss.item()       ) /(itr + 1 )
        total_loss_class    =   ( total_loss_class  * itr + loss_class.item() ) /(itr + 1 )
        total_loss_coord    =   ( total_loss_coord  * itr + loss_coord.item() ) /(itr + 1 )
        total_loss_noobj    =   ( total_loss_noobj  * itr + loss_noobj.item() ) /(itr + 1 )
        total_loss_obj      =   ( total_loss_obj    * itr + loss_obj.item()   ) /(itr + 1 )
        print('Test  : Epoch[{:>3}/{:>3}]  Iterate [{:>5}/{:>5}] Loss : {:.6f} class :{:.3f} coord :{:.3f} noobj :{:.3f} obj :{:.3f} \r'.format(epoch + 1 ,n_epoch,itr+1,len(test_loader),total_loss,total_loss_class,total_loss_coord,total_loss_noobj,total_loss_obj ) ,end = "")
    if(total_loss < best_loss):
        best_loss       =   total_loss
        decay_counter   =   0
        torch.save(model.state_dict(), "./model/best_val.pth")
    else:
        decay_counter   +=  1
    if(decay_counter > decay_limit):
        lr_decay(optimizer)
        decay_counter   =   0
    #torchvision.utils.save_image(output[:,1:,:,:].detach() > 0 ,'save/test.bmp',normalize = True , nrow = 8)
    return best_loss,decay_counter



def Bounding_Box(y_scale1 , y_scale2 , y_scale3, image,count):
    y       =   [y_scale1 , y_scale2 , y_scale3]
    counter =   0
    p      =   [[30,61],[62,45],[59,119],[10,13],[16,30],[33,23],[116,90],[156,198],[373,326] ]
    #print(y_scale1.shape , t_scale1.shape)
    for i,y_scale in enumerate(y):
        #print(t_scale.shape , y_scale.shape)
        y_boxes   =   []
        for n in range(3):
            y_boxes.append(y_scale[:,n*10:(n+1)*10 , :,:])
            
        for y_box in y_boxes :
            #print(y_box.shape , t_box.shape)
            for batch in range(y_box.shape[0]):
                N_size  =   y_box.shape[2]
                place   =   torch.argmax(y_box[batch,4,:,:])
                y       =   math.floor(place / N_size)
                x       =   place   %   N_size
                #print(N_size,place,y,x,torch.max(y_box[batch,4,:,:]))
                if(torch.sigmoid(y_box[batch,4,y,x] )> 0.3):
                    #bx      =   int((x + torch.sigmoid(y_box[batch ,0,y,x])) * (416/N_size))
                    bx      =   int((x * 416/N_size) + torch.sigmoid(y_box[batch ,0,y,x]).item() * (416/N_size))
                    by      =   int((y * 416/N_size) + torch.sigmoid(y_box[batch ,1,y,x]).item() * (416/N_size))
                    #bw      =   int(torch.exp(y_box[batch ,2,y,x]).item() * p[counter][0]) 
                    #bh      =   int(torch.exp(y_box[batch ,3,y,x]).item() * p[counter][1])
                    #bw      =   y_box[batch ,2,y,x] * (N_size)
                    #bh      =   y_box[batch ,3,y,x] * (N_size)
                    bw      =   10
                    bh      =   10
                    #print(bx,by,bw,bh) 
                    index   =   torch.argmax(y_box[batch,5:,y,x]).item()
                    cv2.rectangle(image , (bx - int(bw/2) , by - int(bh/2)) , (bx + int(bw/2) , by + int(bh/2)),color_list[index],1,1 )
                counter +=  1
    cv2.imwrite("./save/No{}.bmp".format(count+1),image)

    
def real(epoch):
    model.eval()
    total_loss          =   0
    for n,img_path in enumerate(real_list):
        data        =   cv2.imread(img_path)
        data        =   cv2.resize(data , (416 ,416))
        box         =   torch.zeros((1,3,416,416))
        data        =   transforms.ToTensor()(np.array(data, dtype = np.float32))
        box[0,:,:,:]=   data
        x_image     =   box.to(device)
        output_1,output_2,output_3          =   model(x_image)
        try:
            Bounding_Box(output_3 , output_2 , output_1 , cv2.resize(cv2.imread(img_path) , (416 , 416)),n)
        except Exception as e:
            print("例外args:", e.args)
            continue



        
for epoch in range(n_epoch):
    real(epoch)
    train(epoch)
    print()
    best_loss,decay_counter = test(epoch,best_loss,decay_counter)
    print()
    
    

