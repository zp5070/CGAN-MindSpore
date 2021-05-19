
import numpy as np
import itertools
import matplotlib.pyplot as plt
import mindspore.ops as ops
from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore import context
import imageio

from src.cell import GenWithLossCell, DisWithLossCell, TrainOneStepCell
from src.dataset import create_dataset
from src.model import Generator, Discriminator

def main():
    batch_size = 128
    input_dim = 100
    epochs = 50
    lr = 0.001
    

    dataset = create_dataset('data/MNIST_Data/train',
                        flatten_size=28 * 28,
                        batch_size=batch_size,
                        num_parallel_workers=2)

    netG = Generator(input_dim)
    netD = Discriminator(batch_size)
    netG_with_loss = GenWithLossCell(netG, netD)
    netD_with_loss = DisWithLossCell(netG, netD)
    optimizerG = nn.Adam(netG.trainable_params(), lr)
    optimizerD = nn.Adam(netD.trainable_params(), lr)

    net_train = TrainOneStepCell(netG_with_loss, netD_with_loss, optimizerG,
                                optimizerD)

    netG.set_train()
    netD.set_train()
    
    latent_code_eval = Tensor(np.random.randn(200, input_dim), dtype=mstype.float32)
    # print(latent_code_eval)
    label_eval = np.zeros((200, 10))
    for i in range(200):
        j = i//20
        label_eval[i][j]=1
    label_eval = Tensor(label_eval, dtype=mstype.float32) 

    for epoch in range(epochs):
        step = 0
        for data in dataset:
            img = data[0]
            label = data[1]
            img=ops.Reshape()(img,(batch_size,1,28,28))
            latent_code = Tensor(np.random.randn(batch_size, input_dim), dtype=mstype.float32)
            dout, gout = net_train(img, latent_code, label)
            step += 1
            
            if step%100 == 0:
                print(
                    "epoch {} step {}, d_loss is {:.4f}, g_loss is {:.4f}".format(
                        epoch, step/100, dout.asnumpy(), gout.asnumpy()))

 
        fig, ax = plt.subplots(10, 20, figsize=(10, 5))
        for digit, num in itertools.product(range(10), range(20)):
            ax[digit, num].get_xaxis().set_visible(False)
            ax[digit, num].get_yaxis().set_visible(False)
            
        gen_imgs_eval = netG(latent_code_eval, label_eval)  
        for i in range(200):
            digit = i//20
            num = i%20
            img = gen_imgs_eval[i].asnumpy().reshape((28, 28))
            ax[digit, num].cla()
            ax[digit, num].imshow(img * 127.5 + 127.5, cmap="gray")
        
        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.01, label, ha='center')
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig("./imgs4/{}.png".format(epoch))
    
    images = []
    for epoch in range(epochs):
        img_name = "./imgs4/{}.png".format(epoch)
        images.append(imageio.imread(img_name))
    imageio.mimsave('./gif/lr_10e-4_rn.gif', images, fps=5)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=4)
    main()