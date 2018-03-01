import utils

list = utils.get_vgg19_decoder_layers_detail('conv4_1')
number = len(list)
for i in range(number, 0, -1):
    for j in range(list[i - 1], 0, -1):
        print('deconv%d_%d'%(i,j))