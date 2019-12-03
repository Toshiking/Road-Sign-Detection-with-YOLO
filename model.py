import  torch.nn            as  nn
import  torch.nn.functional as  F
import  torch
import  torchsummary        as summary

alpha   =   0.1

class ResBlock(nn.Module):
    def __init__(self,channel,kernel_size,stride,padding):
        super(ResBlock,self).__init__()
        self.Conv1      =   nn.Conv2d(channel , channel , kernel_size , stride ,padding)
        self.BatchNorm1 =   nn.BatchNorm2d(channel)
        self.Conv2      =   nn.Conv2d(channel , channel , kernel_size , stride ,padding)
        self.BatchNorm2 =   nn.BatchNorm2d(channel)
    def forward(self,x):
        xx  =   x
        x   =   F.leaky_relu(self.BatchNorm1(self.Conv1(x)),negative_slope=alpha) 
        x   =   self.BatchNorm2(self.Conv2(x))
        #x   =   F.leaky_relu(self.Conv1(x),negative_slope=alpha) 
        #x   =   F.leaky_relu(self.Conv2(x),negative_slope=alpha) 
        
        x   =   x + xx
        x   =   F.leaky_relu(x , negative_slope=alpha)
        return x

class BottleNeck(nn.Module):
    def __init__(self,channel):
        super(BottleNeck,self).__init__()
        self.Conv1      =   nn.Conv2d( channel  ,channel*2 , 3 , 2 ,1)
        self.BatchNorm1 =   nn.BatchNorm2d(channel*2)
        self.Conv2      =   nn.Conv2d( channel*2,channel   , 1 , 1 ,0)
        self.BatchNorm2 =   nn.BatchNorm2d(channel)
        self.Conv3      =   nn.Conv2d( channel  ,channel*2 , 3 , 1 ,1) 
        self.BatchNorm3 =   nn.BatchNorm2d(channel*2)
    def forward(self,x):
        x   =   F.leaky_relu(self.BatchNorm1(self.Conv1(x)),negative_slope=alpha) 
        x   =   F.leaky_relu(self.BatchNorm2(self.Conv2(x)),negative_slope=alpha)
        x   =   F.leaky_relu(self.BatchNorm3(self.Conv3(x)),negative_slope=alpha) 
        #x   =   F.leaky_relu(self.Conv1(x),negative_slope=alpha) 
        #x   =   F.leaky_relu(self.Conv2(x),negative_slope=alpha)
        #x   =   F.leaky_relu(self.Conv3(x),negative_slope=alpha) 
        
        return x

class LastLayer(nn.Module):
    def __init__(self,channel):
        super(LastLayer,self).__init__()
        self.Conv1      =   nn.Conv2d( channel  ,int(channel/2) , 1 , 1 ,0)
        self.BatchNorm1 =   nn.BatchNorm2d(int(channel/2))
        self.Conv2      =   nn.Conv2d(int(channel/2),channel   , 3 , 1 ,1)
        self.BatchNorm2 =   nn.BatchNorm2d(channel)
    def forward(self,x):
        x   =   F.leaky_relu(self.BatchNorm1(self.Conv1(x)),negative_slope=alpha) 
        x   =   F.leaky_relu(self.BatchNorm2(self.Conv2(x)),negative_slope=alpha)
        #x   =   F.leaky_relu(self.Conv1(x),negative_slope=alpha) 
        #x   =   F.leaky_relu(self.Conv2(x),negative_slope=alpha)
        
        return x


class Convolution_Block(nn.Module):
    def __init__(self , input_channel,output_channel , kernel_size , stride , padding ,activation   =   'leaky_relu' , isBatchNorm = True):
        super(Convolution_Block ,self).__init__()
        self.Conv       =   nn.Conv2d(input_channel , output_channel , kernel_size , stride , padding)
        self.BatchNorm  =   nn.BatchNorm2d(output_channel)
        self.activation =   activation
        self.isBatchNorm=   isBatchNorm    
    def forward(self,x):
        
        x                   =   self.Conv(x)
        if(self.isBatchNorm ==  True):
            x               =   self.BatchNorm(x)

        if(self.activation  ==   'leaky_relu'):
            x               =   F.leaky_relu(x,negative_slope=alpha) 
        return x

class YOLO_v3(nn.Module):
    def __init__(self,class_n):
        super(YOLO_v3,self).__init__()
        #入力層
        self.Input_Conv     =   Convolution_Block(  3, 32, 3, 1, 1)

        #第1層
        self.Conv1          =   Convolution_Block( 32, 32, 3, 1, 1)
        self.BN1_1          =   BottleNeck(32)

        #第2層
        self.RB2_1          =   ResBlock(64,3,1,1)
        self.BN2_1          =   BottleNeck(64)

        #第3層
        self.RB3_1          =   ResBlock(128,3,1,1)
        self.RB3_2          =   ResBlock(128,3,1,1)
        self.BN3_1          =   BottleNeck(128)

        #第4層
        self.RB4_1          =   ResBlock(256,3,1,1)
        self.RB4_2          =   ResBlock(256,3,1,1)
        self.RB4_3          =   ResBlock(256,3,1,1)
        self.RB4_4          =   ResBlock(256,3,1,1)
        self.RB4_5          =   ResBlock(256,3,1,1)
        self.RB4_6          =   ResBlock(256,3,1,1)
        self.RB4_7          =   ResBlock(256,3,1,1)
        self.RB4_8          =   ResBlock(256,3,1,1) 
        self.BN4_1          =   BottleNeck(256)

        #第5層
        self.RB5_1          =   ResBlock(512,3,1,1)
        self.RB5_2          =   ResBlock(512,3,1,1)
        self.RB5_3          =   ResBlock(512,3,1,1)
        self.RB5_4          =   ResBlock(512,3,1,1)
        self.RB5_5          =   ResBlock(512,3,1,1)
        self.RB5_6          =   ResBlock(512,3,1,1)
        self.RB5_7          =   ResBlock(512,3,1,1)
        self.RB5_8          =   ResBlock(512,3,1,1)
        self.BN5_1          =   BottleNeck(512)
        
        #第6層
        self.RB6_1          =   ResBlock(1024,3,1,1)
        self.RB6_2          =   ResBlock(1024,3,1,1)
        self.RB6_3          =   ResBlock(1024,3,1,1)
        self.RB6_4          =   ResBlock(1024,3,1,1)
        self.LL_1           =   LastLayer(1024)
        self.LL_2           =   LastLayer(1024)
        self.LL_3           =   LastLayer(1024)

        

        self.Scale_3        =   nn.Conv2d(1024 , (3*(4 + 1 + class_n)) , 1, 1, 0)

        self.Scale_2_0      =   Convolution_Block(1024 , 256 , 1, 1, 0)

        self.Scale_2_1_1    =   Convolution_Block( 768 , 256 , 1, 1, 0)
        self.Scale_2_1_2    =   Convolution_Block( 256 , 512 , 3, 1, 1)
        self.Scale2_2       =   LastLayer(512)
        self.Scale2_3       =   LastLayer(512)
        self.Scale2_4       =   nn.Conv2d( 512 , (3*(4 + 1 + class_n)) , 1, 1, 1)


        self.Scale_1_0      =   Convolution_Block( 512, 128, 1, 1, 0)
        self.Scale_1_1_1    =   Convolution_Block( 384, 128, 1, 1, 0)         
        self.Scale_1_1_2    =   Convolution_Block( 128, 256, 3, 1, 1)

        self.Scale1_2       =   LastLayer(256)
        self.Scale1_3       =   LastLayer(256)
        self.Scale1_4       =   nn.Conv2d( 256 , (3*(4 + 1 + class_n)) , 1, 1, 0)

        self.upsample       =   nn.Upsample(scale_factor = 2 , mode = 'nearest')
    def forward(self , x):
        #入力層
        x               =   self.Input_Conv(x)
        
        #第1層
        x               =   self.BN1_1(self.Conv1(x))
        
        #第2層
        x               =   self.BN2_1(self.RB2_1(x))
        
        #第３層
        x               =   self.BN3_1(self.RB3_2(self.RB3_1(x)))
        
        #第4層
        x               =   self.RB4_8(self.RB4_7(self.RB4_6(self.RB4_5(self.RB4_4(self.RB4_3(self.RB4_2(self.RB4_1(x))))))))
        Scale_1         =   x
        x               =   self.BN4_1(x)
        
        #第5層
        x               =   self.RB5_8(self.RB5_7(self.RB5_6(self.RB5_5(self.RB5_4(self.RB5_3(self.RB5_2(self.RB5_1(x))))))))
        Scale_2         =   x
        x               =   self.BN5_1(x)

        #第6層
        x               =   self.LL_3(self.LL_2(self.LL_1(self.RB6_4(self.RB6_3(self.RB6_2(self.RB6_1(x)))))))
        Scale_3         =   x
  

        #Scale3の出力
        Scale_3_Result  =   self.Scale_3(Scale_3)
        
        #Scale2の出力
        Scale_2_Result  =   torch.cat(( self.upsample(self.Scale_2_0(Scale_3)) , Scale_2 ), dim = 1)
        Scale_2_Result  =   self.Scale2_3(self.Scale2_2(self.Scale_2_1_2(self.Scale_2_1_1(Scale_2_Result))))
        To_Scale_1      =   Scale_2_Result
        Scale_2_Result  =   self.Scale2_4(Scale_2_Result)
        
        Scale_1_Result  =   torch.cat((Scale_1,self.upsample(self.Scale_1_0(To_Scale_1)) ),dim = 1)
        Scale_1_Result  =   self.Scale1_4(self.Scale1_3(self.Scale1_2(self.Scale_1_1_2(self.Scale_1_1_1(Scale_1_Result)))))


        return Scale_1_Result,Scale_2_Result,Scale_3_Result


class Tiny_YOLO(nn.Module):
    def __init__(self,class_n):
        super(Tiny_YOLO,self).__init__()

        self.MaxPooling_stride_1    =   nn.MaxPool2d(2,1,1) 
    
        out             =   3*(4 + 1 + class_n)
        self.MaxPooling =   nn.MaxPool2d(2,2) 
        
        self.Upsample   =   nn.Upsample(scale_factor = 2 , mode = 'nearest')
        
        self.Conv1      =   Convolution_Block(   3,  16,3,1,1)
        self.Conv2      =   Convolution_Block(  16,  32,3,1,1)
        self.Conv3      =   Convolution_Block(  32,  64,3,1,1)
        self.Conv4      =   Convolution_Block(  64, 128,3,1,1)
        self.Conv5      =   Convolution_Block( 128, 256,3,1,1)
        self.Conv6      =   Convolution_Block( 256, 512,3,1,1)
        self.Conv7      =   Convolution_Block( 512,1024,3,1,1)
        self.Conv8      =   Convolution_Block(1024, 256,1,1,0)
        self.Conv9      =   Convolution_Block( 256, 512,3,1,1)
        self.Conv10     =   Convolution_Block( 512, out,1,1,0,activation = 'linear' , isBatchNorm = False)
        
        self.Conv11     =   Convolution_Block( 256, 128,1,1,0)
        self.Conv12     =   Convolution_Block( 384, 256,3,1,1)
        self.Conv13     =   Convolution_Block( 256, out,1,1,0)
        
    def forward(self,x):
        x               =   self.MaxPooling(self.Conv1(x))
        #print("1",x.shape)
        x               =   self.MaxPooling(self.Conv2(x))
        #print("2",x.shape)
        x               =   self.MaxPooling(self.Conv3(x))
        #print("3",x.shape)
        x               =   self.MaxPooling(self.Conv4(x))
        #print("4",x.shape)
        x               =   self.Conv5(x)
        branch_1        =   x
        x               =   self.MaxPooling(x)
        #print("5",x.shape)
        x               =   self.Conv6(x)
        #print("6",x.shape)
        
        x               =   self.Conv7(x)
        #print("7",x.shape)
        
        x               =   self.Conv8(x)
        #print("8",x.shape)
        
        branch_2        =   x
    
        x               =   self.Conv9(x)
        scale2          =   self.Conv10(x)

        x               =   self.Upsample(self.Conv11(branch_2))
        #print(x.shape,branch_1.shape)
        
        x               =   torch.cat((x,branch_1) , dim = 1)
        x               =   self.Conv12(x)
        scale1          =   self.Conv13(x)

        return scale1 , scale2 
