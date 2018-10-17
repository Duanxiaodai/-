class ReluActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return max(0, weighted_input)
    def backward(self, output):
        return 1 if output > 0 else 0

class  IdentityActivator():
    def forward(self,weight_input):
        return weight_input
    def backward(self,weighted_input):
        return 1.0
