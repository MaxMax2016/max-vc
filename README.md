# singing voice conversion based on latent f0

In this project, the reconstructed torchcrepe is used to extract F0 hidden features to solve the problem of F0 explosion.

本项目使用改造的torchcrepe提取F0隐藏特征，用于解决F0炸掉的问题

With a little training, the robustness of F0 is greatly improved, but F0 is not controllable.

浅浅训练一下，F0鲁棒性大大提高，但F0不具备可控性


## 测试
从release页面下面模型[max-vc-pretrain.pth](https://github.com/PlayVoice/max-vc/releases/tag/v0.1)

下载crepe模型[full.pth](https://github.com/maxrmorrison/torchcrepe/blob/master/torchcrepe/assets/full.pth)，放到crepe/assets/full.pth

> python svc_inference.py --config config/maxgan.yaml --model model_pretrain/max-vc-pretrain.pth --spk config/singers/singer0001.npy --wave test.wav

生成文件在当前目录svc_out.wav


## 训练（未完善）

- 1 数据准备，将音频切分小于30S（推荐10S左右/可以不依照句子结尾）， 转换采样率为16000Hz, 将音频数据放到 **./data_svc/waves**
    > 这个我想你会~~~

- 2 下载音色编码器: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), 解压文件，把 **best_model.pth** 和 **condif.json** 放到目录 **speaker_pretrain/**

    提取每个音频文件的音色
    
    > python svc_preprocess_speaker.py ./data_svc/waves ./data_svc/speaker
    
    取所有音频文件的音色的平均作为目标发音人的音色
    
    > python svc_preprocess_speaker_lora.py ./data_svc/

- 3 下载whisper模型 [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 确定下载的是**medium.pt**，把它放到文件夹**whisper_pretrain/**中，提取每个音频的内容编码

    > python svc_preprocess_ppg.py -w ./data_svc/waves -p ./data_svc/whisper

- 4 提取基音，同时生成训练文件 **filelist/train.txt**，剪切train的前5条用于制作**filelist/eval.txt**

    使用深度改造的torchcrepe提取F0隐藏特征，下载模型[full.pth](https://github.com/maxrmorrison/torchcrepe/blob/master/torchcrepe/assets/full.pth)

    > python svc_preprocess_f0.py

- 5 从release页面下载预训练模型maxgan_pretrain，放到model_pretrain文件夹中，预训练模型中包含了生成器和判别器

    > python svc_trainer.py -c config/maxgan.yaml -n lora -p model_pretrain/maxgan_pretrain.pth


你的文件目录应该长这个样子~~~

    data_svc/
    |
    └── lora_speaker.npy
    |
    └── pitch
    │     ├── 000001.pit.npy
    │     ├── 000002.pit.npy
    │     └── 000003.pit.npy
    └── speakers
    │     ├── 000001.spk.npy
    │     ├── 000002.spk.npy
    │     └── 000003.spk.npy
    └── waves
    │     ├── 000001.wav
    │     ├── 000002.wav
    │     └── 000003.wav
    └── whisper
          ├── 000001.ppg.npy
          ├── 000002.ppg.npy
          └── 000003.ppg.npy


## 代码来源和参考文献
[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/openai/whisper/ [[paper]](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [[paper]](https://arxiv.org/abs/2206.04658)

https://github.com/maxrmorrison/torchcrepe

https://github.com/chenwj1989/pafx

# 期望您的鼓励
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。
