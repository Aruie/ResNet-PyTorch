
import torch
import torch.nn as nn



def ResNet18(include_top = True) :
    return ResNet([3,4,6,3], bottleneck = False, include_top = include_top )

def ResNet34(include_top = True) :
    return ResNet([3,4,6,3], bottleneck = False, include_top = include_top )

def ResNet50(include_top = True) :
    return ResNet([3,4,6,3], bottleneck = True, include_top = include_top )

def ResNet101(include_top = True) :
    return ResNet([3,4,23,3], bottleneck = True, include_top = include_top )

def ResNet152(include_top = True) :
    return ResNet([3,8,36,3], bottleneck = True, include_top = include_top )



class ResNet(nn.Module) :
    def __init__(self, repeat_list, bottleneck, include_top = True) :
        super(ResNet, self).__init__()

        self.include_top = include_top

        self.conv_intro = nn.Conv2d(3, 64, (7,7), stride = 2, padding = 3)
        self.pool = nn.MaxPool2d(3, stride = 2, padding = 1 )
        self.bottleneck = bottleneck

        self.block1 = RepeatedBlock(self.bottleneck, 64, repeat_list[0], True)
        self.block2 = RepeatedBlock(self.bottleneck, 128, repeat_list[1])
        self.block3 = RepeatedBlock(self.bottleneck, 256, repeat_list[2])
        self.block4 = RepeatedBlock(self.bottleneck, 512, repeat_list[3])

        if self.include_top == True :
            self.gap = nn.AvgPool2d(7)

            if self.bottleneck == True :
                self.fc = nn.Linear(2048, 1000)
            else :
                self.fc = nn.Linear(512, 1000)

    def  forward(self, x) :

        x = self.conv_intro(x)
        x = self.pool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        if self.include_top == True :
            x = self.gap(x)
            channel = x.shape[1]
            x = x.view(-1,channel)
            x = self.fc(x)
        return x








# Repeat Block
class RepeatedBlock(nn.Module) :
    def __init__(self,  bottleneck, channel, repeat, is_first = False) :
        super(RepeatedBlock, self).__init__()
        
        self.chaanel = channel
        self.is_first = is_first
        self.repeat = repeat
        self.bottleneck = bottleneck

        self.blocks = nn.ModuleList()

        for i in range(self.repeat) :
            is_start = (True if i == 0 else False)
                
            if bottleneck == True :
                self.blocks.append( ResidualBlock(channel, is_start, is_first ))
            else :
                self.blocks.append( ResidualBlockS(channel, is_start, is_first ))
            
    def forward(self, x) :
        for layer in self.blocks :
            x = layer(x)
        return x


# Residual Block with BottleNeck
class ResidualBlock(nn.Module) :
    def __init__(self, channel, is_start, is_first = False) :
        super(ResidualBlock, self).__init__()
        
        self.channel = channel
        self.is_start = is_start
        self.is_first = is_first

        if self.is_start == True : 
            if self.is_first == True : 
                self.conv1 = nn.Conv2d(self.channel, self.channel, (1,1))
                self.convsc = nn.Conv2d(self.channel, self.channel * 4, (1,1))
                self.conv2 = nn.Conv2d(self.channel, self.channel, (3,3), padding = 1)

            else : 
                self.conv1 = nn.Conv2d(self.channel*2, self.channel, (1,1))
                self.convsc = nn.Conv2d(self.channel*2, self.channel * 4, (1,1), stride=2)
                self.conv2 = nn.Conv2d(self.channel, self.channel, (3,3), stride=2, padding = 1)        
            

        else :
            self.conv1 = nn.Conv2d(self.channel*4, self.channel, (1,1))
            self.conv2 = nn.Conv2d(self.channel, self.channel, (3,3), padding = 1)       
            
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.act1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(self.channel)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.channel, self.channel * 4 , (1,1))
        self.bn3 = nn.BatchNorm2d(self.channel * 4)
        self.act3 = nn.ReLU()

    def forward(self, x) :
        sc = x
        if self.is_start == True :
            sc = self.convsc(sc)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        x = x + sc

        x = self.act3(x)
        return x

# Residual Block without BottleNeck
class ResidualBlockS(nn.Module) :
    def __init__(self, channel, is_start, is_first = False) :
        super(ResidualBlockS, self).__init__()
        
        self.channel = channel
        self.is_start = is_start
        self.is_first = is_first

        if self.is_start == True :
            if self.is_first == True : 
                self.conv1 = nn.Conv2d(self.channel, self.channel, (3,3))
            else :
                self.conv1 = nn.Conv2d( int(self.channel / 2), self.channel, (3,3), stride = 2)
                self.convsc = nn.Conv2d( int(self.channel / 2), self.channel, (1,1), stride = 2)
        else :
            self.conv1 = nn.Conv2d(self.channel, self.channel, (3,3))
            
        self.conv2 = nn.Conv2d(self.channel, self.channel, (3,3))
            
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.act1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(self.channel)
        self.act2 = nn.ReLU()
        
    def forward(self, x) :
        if self.is_start == True and self.is_first == False :
            sc = self.convsc(x)
        else :
            sc = x

        x = nn.ZeroPad2d(1)(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = nn.ZeroPad2d(1)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = x + sc

        x = self.act2(x)

        return x






if __name__=="__main__" :
    
    from torchsummary import summary
    
    model = ResNet152(True)
    summary(model, (3,224,224))

    x_input = torch.randn(1, 3, 224, 224)
    print(x_input.shape)

    y_pred = model(x_input)
    
    print(y_pred.shape)

    

