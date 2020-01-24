
import torch
import torch.nn as nn

class ResNet50(nn.Module) :
    def __init__(self, include_top = True) :
        super(ResNet50, self).__init__()

        self.include_top = include_top

        self.conv_intro = nn.Conv2d(3, 64, (7,7))
        self.pool = nn.MaxPool2d(2, stride = 2, padding = 1 )

        self.block1 = nn.ModuleList()
        self.block1.append(ResidualBlock(64, is_start = True, is_first = True))
        self.block1.append(ResidualBlock(64, is_start = False))
        self.block1.append(ResidualBlock(64, is_start = False))
        
        self.block2 = nn.ModuleList()
        for i in range(4) :
            is_start = True if i == 0 else False
            self.block2.append( ResidualBlock(128, is_start ))

        self.block3 = nn.ModuleList()
        for i in range(6) :
            is_start = True if i == 0 else False
            self.block3.append( ResidualBlock(256, is_start ))

        self.block4 = nn.ModuleList()
        for i in range(3) :
            is_start = True if i == 0 else False
            self.block4.append( ResidualBlock(512, is_start ))

        if self.include_top == True :
            self.gap = nn.AvgPool2d(7)
            self.fc = nn.Linear(2048, 1000)


    def  forward(self, x) :

        x = nn.ZeroPad2d(2)(x)
        x = self.conv_intro(x)
        x = self.pool(x)

        for it in self.block1 :
            x = it(x)

        for it in self.block2 :
            x = it(x)

        for it in self.block3 :
            x = it(x)

        for it in self.block4 :
            x = it(x)

        if self.include_top == True :
            x = self.gap(x)
            x = self.fc(x)

        return x



class ResidualBlock(nn.Module) :
    def __init__(self, input_channel, is_start, is_first = False) :
        super(ResidualBlock, self).__init__()
        
        self.input_channel = input_channel
        self.is_start = is_start
        self.is_first = is_first

        if self.is_start == True : 
            if self.is_first == True : 
                self.conv1 = nn.Conv2d(self.input_channel, self.input_channel, (1,1))
                self.convsc = nn.Conv2d(self.input_channel, self.input_channel * 4, (1,1), stride=2)

            else : 
                self.conv1 = nn.Conv2d(self.input_channel*2, self.input_channel, (1,1))
                self.convsc = nn.Conv2d(self.input_channel*2, self.input_channel * 4, (1,1), stride=2)

            self.conv2 = nn.Conv2d(self.input_channel, self.input_channel, (3,3), stride=2)

        else :
            self.conv2 = nn.Conv2d(self.input_channel, self.input_channel, (3,3))
            self.conv1 = nn.Conv2d(self.input_channel*4, self.input_channel, (1,1))
            
        self.bn1 = nn.BatchNorm2d(self.input_channel)
        self.act1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(self.input_channel)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.input_channel, self.input_channel * 4 , (1,1))
        self.bn3 = nn.BatchNorm2d(self.input_channel * 4)
        self.act3 = nn.ReLU()

    def forward(self, x) :
        sc = x
        if self.is_start == True :
            sc = self.convsc(sc)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = nn.ZeroPad2d(1)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        x = x + sc

        x = self.act3(x)
        return x


if __name__=="__main__" :
    
    from torchsummary import summary

    
    model = ResNet50(include_top=False)
    summary(model, (3,224,224))

    x_input = torch.randn(1, 3, 224, 224)
    print(x_input.shape)

    y_pred = model(x_input)
    
    print(y_pred.shape)

    

