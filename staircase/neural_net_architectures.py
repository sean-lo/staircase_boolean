import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUResNetNormalized(torch.nn.Module):
    def __init__(self,input_length,num_layers,how_fat):
        super(ReLUResNetNormalized, self).__init__()
        self.num_layers = num_layers
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_layers-1):
            layer_width.append(how_fat)
        layer_width.append(1)

        # Do not give the last layer bias.
#        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=(i != num_layers-1)) for i in range(num_layers)])
        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=True) for i in range(num_layers)])

        # def xavier(param):
        #     torch.nn.init.xavier_uniform(param)
        # def weights_init(m):
        #     if isinstance(m, nn.Linear):
        #         # print(m)
        #         print('Xavier initialization')
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         torch.nn.init.zeros_(m.bias)
        # self.linears.apply(weights_init)

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        prevx = x;
        for i, l in enumerate(self.linears):
            prevx = x;
            x = l(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                if i > 0:
                    x = x + prevx
        x = torch.sum(x,axis=1)
        return x

class ReLUResNet(torch.nn.Module):
    def __init__(self,input_length,num_layers,how_fat):
        super(ReLUResNet, self).__init__()
        self.num_layers = num_layers
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_layers-1):
            layer_width.append(how_fat)
        layer_width.append(1)

        # Do not give the last layer bias.
#        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=(i != num_layers-1)) for i in range(num_layers)])
        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=True) for i in range(num_layers)])
    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        prevx = x;
        for i, l in enumerate(self.linears):
            prevx = x;
            x = l(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                if i > 0:
                    x = x + prevx
        x = torch.sum(x,axis=1)
        return x

    def set_closetozeroinit(self, val):
        for linear in self.linears:
            linear.weight.data = torch.randn(linear.weight.data.size()) * val;
            print(torch.norm(linear.weight.data))

class LeakyReLUResNet(torch.nn.Module):
    def __init__(self,input_length,num_layers,how_fat,negative_slope=0.1):
        super(LeakyReLUResNet, self).__init__()
        self.num_layers = num_layers
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_layers-1):
            layer_width.append(how_fat)
        layer_width.append(1)
        self.negative_slope = negative_slope

        # Do not give the last layer bias.
        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=(i != num_layers-1)) for i in range(num_layers)])
    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        prevx = x;
        for i, l in enumerate(self.linears):
            prevx = x;
            x = l(x)
            if i < self.num_layers - 1:
                x = F.leaky_relu(x,negative_slope=self.negative_slope)
                if i > 0:
                    x = x + prevx
        x = torch.sum(x,axis=1)
        return x

    def set_closetozeroinit(self, val):
        for linear in self.linears:
            linear.weight.data = torch.randn(linear.weight.data.size()) * val;
            print(torch.norm(linear.weight.data))

class QuadResNet(torch.nn.Module):
    def __init__(self,input_length,num_layers,how_fat):
        super(QuadResNet, self).__init__()
        self.num_layers = num_layers
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_layers-1):
            layer_width.append(how_fat)
        layer_width.append(1)

##        # Do not give the last layer bias.
##        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=(i != num_layers-1)) for i in range(num_layers)])
        # Do not give the last layer bias.
        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=True) for i in range(num_layers)])
    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        prevx = x;
        for i, l in enumerate(self.linears):
            prevx = x;
            x = l(x)
            if i < self.num_layers - 1:
                x = torch.square(x)
                if i > 0:
                    x = x + prevx
        return torch.sum(x,axis=1)

    def set_closetozeroinit(self, val):
        for linear in self.linears:
            linear.weight.data = torch.randn(linear.weight.data.size()) * val;
            print(linear.bias.data)
            print(torch.norm(linear.weight.data))


class QuadNetSumLayersSkips(torch.nn.Module):
    def __init__(self,input_length,num_layers,how_fat):
        super(QuadNetSumLayersSkips, self).__init__()
        self.num_layers = num_layers
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_layers-1):
            layer_width.append(how_fat)
        layer_width.append(1)


        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=True) for i in range(num_layers)])
        self.skips = nn.ModuleList([nn.Linear(layer_width[0], layer_width[i+1],bias=True) for i in range(num_layers)])

    def forward(self, x):
        inputx = x
        output = torch.zeros(x.shape[0])
#        print(output)
        # ModuleList can act as an iterable, or be indexed using ints
        prevx = x;
        for i, l in enumerate(self.linears):
            prevx = x;
            x = l(x)
            x = x + self.skips[i](inputx)
            if i < self.num_layers - 1:
                x = torch.square(x)
                if i > 0:
                    x = x + prevx
#            print(output)
#            print(torch.sum(x,1))
            output = output + torch.sum(x,1)
#            print(output)
        return output

    def set_closetozeroinit(self, val):
        for linear in self.linears:
            linear.weight.data = torch.randn(linear.weight.data.size()) * val;
            linear.bias.data = torch.randn(linear.bias.data.size()) * val;
            print(torch.norm(linear.weight.data))

        for skip in self.skips:
            skip.weight.data = torch.randn(skip.weight.data.size()) * val;
            skip.bias.data = torch.randn(skip.bias.data.size()) * val;


class QuadSumLayersNetVersion2(torch.nn.Module):

    # output = L1x + L2(L1x)^2 + L3(L2(L1x)^2)^2 + ...

    def __init__(self,input_length,num_squares,how_fat):
        super(QuadSumLayersNetVersion2, self).__init__()
        self.num_squares = num_squares
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_squares):
            layer_width.append(how_fat)
        layer_width.append(1)


        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=False) for i in range(num_squares+1)])


    def forward(self, x):
        inputx = x

        x = self.linears[0](x)
        output = torch.sum(x,axis=1)

        for i in range(self.num_squares):
#            print('Here')
            x = torch.square(x)
            x = self.linears[i+1](x)
            output = output + torch.sum(x,axis=1)

#        print(output)
        return output

    def set_closetozeroinit(self, val):
        for linear in self.linears:
            linear.weight.data = torch.randn(linear.weight.data.size()) * val;
            print(torch.norm(linear.weight.data))

class QuadSumLayersNetVersion1(torch.nn.Module):

    def __init__(self,input_length,num_squares,how_fat):
        super(QuadSumLayersNetVersion1, self).__init__()
        self.num_squares = num_squares
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_squares):
            layer_width.append(how_fat)
        layer_width.append(1)


        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1],bias=False) for i in range(num_squares+1)])
        self.presquares = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i],bias=False) for i in range(1,num_squares+1)])


    def forward(self, x):
        inputx = x

        x = self.linears[0](x)
        output = torch.sum(x,axis=1)

        for i in range(self.num_squares):
#            print('Here')
            x = self.presquares[i](x)
            x = torch.square(x)
            x = self.linears[i+1](x)
            output = output + torch.sum(x,axis=1)

#        print(output)
        return output

    def set_closetozeroinit(self, val):
        for linear in self.linears:
            linear.weight.data = torch.randn(linear.weight.data.size()) * val;
            print(torch.norm(linear.weight.data))


class FullyConnNet(nn.Module):
    def __init__(self,input_length,num_layers,how_fat):
        super(FullyConnNet, self).__init__()
        self.num_layers = num_layers
        layer_width = []
        layer_width.append(input_length)
        for i in range(num_layers-1):
            layer_width.append(how_fat)
        layer_width.append(1)
        self.linears = nn.ModuleList([nn.Linear(layer_width[i], layer_width[i+1]) for i in range(num_layers)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x

##class MeanField(torch.nn.Module):
##    def __init__(self,input_length,how_fat):
##        super(MeanField, self).__init__()
##        self.linear = nn.Linear(input_length,how_fat)
##        self.meanfieldweights = torch.ones(how_fat)
##        self.meanfieldweights[0,:how_fat//2] = -1

##    def forward(self, x):
##        x = self.linear(x)
###        x = F.relu(x)
###        x = F.leaky_relu(x,negative_slope=0.5)
###        print(x.shape)
###        print(self.meanfieldweights.shape)
##        x = torch.dot(x[0], self.meanfieldweights)
###        print(x)
##        return x

class OneLayer(torch.nn.Module):
    def __init__(self,input_length,how_fat):
        super(OneLayer, self).__init__()
        self.how_fat = how_fat
        self.linear1 = nn.Linear(input_length,how_fat)
#        self.linear1.weight.requires_grad = False
#        self.linear1.bias.requires_grad = False
        self.linear2 = nn.Linear(how_fat,1)
#        self.linear2.weight.requires_grad = False
#        self.linear2.bias.requires_grad = False

    def set_meanfield(self):
        layer2shape = self.linear2.weight.data.shape
        mf_weights = torch.ones(layer2shape)
        mf_weights[0,:self.how_fat//2] = -1
        mf_weights = mf_weights / math.sqrt(self.how_fat)
        self.linear2.weight.data = mf_weights
        self.linear2.weight.requires_grad = False
        print(self.linear2.weight)

    def set_closetozeroinit(self, val):
        self.linear1.weight.data = torch.randn(self.linear1.weight.data.size()) * val;
        print(torch.norm(self.linear1.weight.data))

    def print_ones_twos_weights_stats(self,tot_check=10):
        A = self.linear1.weight.data;
        for i in range(tot_check):
            print(i,torch.norm(A[:,i]))
        print('rest', torch.norm(A[:,tot_check:]))



    def forward(self, x):
        x = self.linear1(x)
#        x = F.relu(x)
        x = F.leaky_relu(x,negative_slope=0.01)
#        x = torch.square(x)
#        x = torch.pow(x, 10)
        x = self.linear2(x)
        return x
