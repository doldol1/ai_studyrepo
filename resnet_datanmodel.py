import torch
import glob
import os
# glob 결과 숫자 오름차순으로 정리해주는 라이브러리, 기능적으로 필요하지 않았음을 깨달았으나
# 정렬 작업이 유지보수를 가정했을 때 충분히 의미 있다고 생각해서 그냥 놔두기로 함
import natsort
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

import numpy as np
import time

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# transform을 적용한 커스텀 데이터셋
# 무조건 torch.utils.data.Dataset을 상속받아야 한다.
class cnd_data(torch.utils.data.Dataset):
    def __init__(self, file_path, train=True, transforms=None):

        self.train=train
        self.transforms=transforms

        # cat, dog 경로 설정
        # self.cat_img_path=os.path.join(file_path, 'data\kagglecatsanddogs\PetImages\Cat')
        # self.dog_img_path=os.path.join(file_path, 'data\kagglecatsanddogs\PetImages\Dog')
        self.cat_img_path=os.path.join(file_path, 'data/kagglecatsanddogs/PetImages/Cat')
        self.dog_img_path=os.path.join(file_path, 'data/kagglecatsanddogs/PetImages/Dog')

        # cat, dog 이미지 목록 불러오기
        self.cat_list=natsort.natsorted(glob.glob(self.cat_img_path + '/*.jpg'))
        self.dog_list=natsort.natsorted(glob.glob(self.dog_img_path + '/*.jpg'))

        # cat, dog 이미지 list 및 label 지정하기, 0은 cat이고, 1은 dog이다
        # cat, dog 각각 12500개의 이미지가 존재하며, 각각 12000개는 train, 500개는 test에 사용된다
        if self.train==True:
            self.imgn_list=self.cat_list[:12000]+self.dog_list[:12000]
            self.img_label=[0]*12000+[1]*12000

        else:
            self.imgn_list=self.cat_list[12000:]+self.dog_list[12000:]
            self.img_label=[0]*500+[1]*500

        # 한번에 모든 이미지를 메모리에 올리고 싶었지만 공간 부족으로 불가
        # getitem쪽에 올렸다.


    # __len__()은 데이터쌍의 개수를 의미한다.
    # 아마 __len__의 크기를 기준으로 Dataloader에서 batch 묶음의 수를 결정하고
    # __len__만큼의 데이터쌍을 가져오는 것 같다.
    def __len__(self):
        return len(self.img_label)

    # __getitem__()은 하나의 데이터쌍(보통 데이터, 레이블)을 가져오는데 사용된다.
    # __getitem__출력시 한 쌍의 데이터가 아니라 한 batch만큼을 한번에 불러오는 방식으로 짜고 싶었지만
    # (만약 그렇게 한다면 Dataloader에서 불러온 다음 중첩 for문을 사용하여 사용하게 될 것이다.)
    # 처음으로 짜는 커스텀 데이터셋이기 때문에 한 쌍의 데이터를 가져올 때마다
    def __getitem__(self, idx):

        # 원 데이터는 cat과 dog 폴더로 나뉘어 있으며, 각각 0~12499까지 숫자가 파일 이름으로 사용된다.
        # 또한, train은 0~11999, test는 12000~12499 를 파일 이름으로 사용한다.
        # train 기준 실존하는 imgn_list의 index는 0~23999까지 사용하게 되므로,
        # 0~11999 idx의 경우 cat폴더에서 가져와야 하며,
        # 12000~23999 idx의 경우 dog 폴더에서 가져와야 한다.(당연히 dog폴더의 train이미지는 0~11999이므로 숫자 변환도 필요하다)
        # 라고 처음에는 생각해 왔지만 헛생각이었다... 어차피 인덱스와 이에 해당하는 이미지 경로는 연결되어 있으니 추가적인 조치를 취하지 않고도
        # 문제를 해결할 수 있다.
        image_data=Image.open(self.imgn_list[idx]).convert('RGB')


        # if len(np.array(image_data).shape)==2:
        #     image_data=image_data.convert('RGB')
        #     print('변환 후 사이즈:',np.array(image_data).shape)

        if self.transforms:
            sample=self.transforms(image_data)

        # print('사이즈:', sample.size())
        # 이미지에서 channel이 3이 아닌 경우
        # if sample.size()[0] != 3:
        #     print(self.imgn_list[idx])
        #     print('변환 사이즈:', sample.size())

            # sample=sample.expand(3, -1, -1)
            # print(sample.size())

        return sample, self.img_label[idx]
    


# 모델을 정의할 때는 무조건 torch.nn.Module을 상속받아야 한다.
# block은 short connection이 있는 최소 단위이며
# group은 동일한 block 형성 패턴(논문 table 1의 conv2_x, conv3_x)을 의미한다.
class ResNet_compat(nn.Module):
    def __init__(self,
                 input_shape=(3, 224, 224),
                 blocks_in_model=[3, 4, 6, 3],
                 layers_in_block=[2, 2, 2, 2],
                 kernel_sizes=[(3,3), (3,3), (3,3), (3,3)],
                 channel_sizes=[(64,64), (128,128), (256,256), (512,512)],
                 class_size=2,
                 is_plain=False):

        super(ResNet_compat, self).__init__()

        self.input_shape=input_shape
        self.blocks_in_model=blocks_in_model
        self.layers_in_block=layers_in_block
        self.kernel_sizes=kernel_sizes
        self.channel_sizes=channel_sizes
        self.class_size=class_size
        self.is_plain=is_plain


        # pytorch에도 padding='same'이라는 옵션은 존재하지만, stride=1일
        # 경우만 사용 가능하다.
        # 아래 코드는 (W-F+2P)/S + 1 공식 적용한 코드로
        # 계산 결과가 소수점이 나오지만, pytorch에서 사용하는 resnet이 이렇게 설정하였기 때문에
        # 똑같이 진행한다.

        # conv1+conv2 maxpooling
        self.conv1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), # 왜 그런지 모르겠지만 preset에는 inplace=True(default: False)
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        )

        # conv2(maxpooling은 제외), short connection은 구현부에서 구현

        self.block_forms=nn.Sequential()
        for i in range(len(blocks_in_model)):
            for j in range(blocks_in_model[i]):

                # i는 group, j는 block
                if i==0 and j==0:
                    inp_channel=64
                elif i!=0 and j==0:
                    inp_channel=channel_sizes[i-1][-1]
                else:
                    inp_channel=False

                # # input_channel은 block의 맨 처음+group(group: block묶음)의 맨 처음인 경우만 사용됨
                # self.block_forms.add_module(nn.Sequential(*self.build_block(
                #     self.layers_in_block[i],
                #     kernel_sizes=self.kernel_sizes[i],
                #     channel_sizes=self.channel_sizes[i],
                #     input_channel=inp_channel,
                #     is_plain=False
                # )))
                # input_channel은 block의 맨 처음+group(group: block묶음)의 맨 처음인 경우만 사용됨
                self.block_forms.add_module(f'bblock{i, j}',self.build_block(
                    self.layers_in_block[i],
                    kernel_sizes=self.kernel_sizes[i],
                    channel_sizes=self.channel_sizes[i],
                    input_channel=inp_channel,
                    is_plain=False
                ))



        # summary가 안 먹혀서 새로 짠 코드
        # self.model_body=nn.Sequential(*self.block_forms)

        # 선언을 하려고 하면 추가 입력이 필요해서 forward에 설정해야 하는 상황...
        self.relu = nn.ReLU()

        self.end_avg2d=nn.AdaptiveAvgPool2d((1,1))
        self.end_linear=nn.Linear(in_features=self.channel_sizes[-1][-1], out_features=2, bias=True)
        self.end_softmax=nn.Softmax(-1)

    # input_channel은 block의 맨 처음+group(group: block묶음)의 맨 처음인 경우만 사용됨
    def build_block(self, layers, kernel_sizes, channel_sizes, input_channel, is_plain=False):

        full_block=nn.Sequential()
        for i in range(layers):
            if kernel_sizes[i]!= 1:
                layer_padding=(1,1)
            else:
                layer_padding=(0,0)

            if input_channel and i==0:
                if input_channel != channel_sizes[i]:
                    f_stride=2
                else:
                    f_stride=1
                full_block.add_module(f'conv{i}',nn.Conv2d(in_channels=input_channel,
                                            out_channels=channel_sizes[i],
                                            kernel_size=kernel_sizes[i],
                                            padding=layer_padding,
                                            stride=f_stride,
                                            bias=False
                                            ))
            else:
                # 50이상은 channel이 일시적으로 늘어나도 feature map의 크기가 그대로임음 명심할 것
                # padding만 어떻게 할지 고민해보자... kernel_size가 1일 때는 패딩 제외?? 아니면 3일 때만 padding 1??
                full_block.add_module(f'conv{i}',nn.Conv2d(in_channels=channel_sizes[i-1],
                                            out_channels=channel_sizes[i],
                                            kernel_size=kernel_sizes[i],
                                            padding=layer_padding,
                                            bias=False
                                            ))
            # batch_normalization
            full_block.add_module(f'batnorm{i}',nn.BatchNorm2d(channel_sizes[i], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

            # short connection은 구현부에서 만들기 때문에 block의 마지막 layer가 아니라면 relu 추가
            # inplace 옵션이 있고, pre-model에서는 사용하긴 사용하지만 왜 사용하는지 모르겠어서 사용 안함
            if i< layers-1:
                full_block.add_module(f'relu{i}',nn.ReLU())

        return full_block

    def forward(self, x):
        # print(x.shape)
        x=self.conv1(x)
        # print(x.shape)

        # model body
        for block in self.block_forms:
            block= block # 만약 구조적 문제가 해결되면 삭제 시도해볼 것
            identity=x
            x=block(x)

            if block[0].in_channels != block[0].out_channels:
                self.reduce=nn.Conv2d(
                    block[0].in_channels,
                    block[0].out_channels,
                    kernel_size=(1,1),
                    stride=2).to(device)
                identity=self.reduce(identity)

            x+=identity
            x=self.relu(x)


        # 끝단
        x=self.end_avg2d(x)
        x=torch.flatten(x, 1, -1)

        # x현재 shape는 [2, 512, 1, 1]
        x=self.end_linear(x)
        x=self.end_softmax(x)

        return x

        # return self.conv_temp(x)

        pass