# singing voice conversion without f0

In this project, the reconstructed torchcrepe is used to extract audio encodec but not F0 to solve the problem of F0 explosion.

本项目使用改造的torchcrepe提取音频编码，而不提取F0，用于解决F0炸掉的问题


## 测试
从release页面下面模型[max-vc-pretrain.pth](https://github.com/PlayVoice/max-vc/releases/tag/v0.1)

下载crepe模型[full.pth](https://github.com/maxrmorrison/torchcrepe/blob/master/torchcrepe/assets/full.pth)，放到crepe/assets/full.pth

> python svc_inference.py --config config/maxgan.yaml --model model_pretrain/max-vc-pretrain.pth --spk config/singers/singer0001.npy --wave test.wav

生成文件在当前目录svc_out.wav

💗由于没有提取F0，所以模型不具备音高迁移的能力；但是，可以对原始音频进行音高迁移（SoundTouch），然后在使用模型进行VC。

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
