# end_to_end-GPTSOVITS
端到端的声音克隆项目(对GPT-SoVITS工程化)
端到端的声音克隆：即输入原音频(1分钟就可以，时间越长越好)和结果输出的音频命名，执行相应脚本，等待程序执行完成，即可得到克隆结果和模型。

项目1：声音克隆的训练一体化整合项目以及API。
下载链接：https://pan.quark.cn/s/e66ae87e6075

(虚拟环境包脚本链接：https://pan.quark.cn/s/bf2511f7b1fc)

项目2：声音克隆的推理一体化整合项目以及API。
下载链接：https://pan.quark.cn/s/b1d347574219


![1717582180416](https://github.com/jiangyixing/end_to_end-GPTSOVITS/assets/130124955/794e4a3e-ed12-4867-afc1-1e8441c924f1)

项目1：项目简介

1.该项目是本人基于GPT-SoVITS做的工程化，为什么用GPT-SoVITS？因为经过调研，目前开源质量最好的声音克隆项目就是GPT-SoVITS。1分钟的音频，训练生成结果大概5-10分钟。

2.首先温馨提醒：该项目前提是需要会配置深度学习的虚拟环境(关键包的版本我会在后面声明，未声明就是正常安装就行)，因为本人不是专业做教程的，目的是将技术打通，自己使用的同时共享给大家一起使用，在修改项目的过程中，可能没有将项目整理的很干净，有些多余文件，但没关系他们不会影响项目的执行，由于以上原因可能不适合纯小白。

3.使用说明：该项目以及产品2下载解压，都要统一放在E:\project下，再配置好环境，即可成功。因为在测试过程中，有些流程对相对路径会报错，所以需要用到绝对路径。主目录是：E:\project\GPT-SoVITS-main\打包好\GPT-SoVITS

4.硬件说明：本人使用window11系统，单卡显卡4070ti，12G显存，64G内存进行测试并修改的。故项目里面的相关设置都是基于单卡的，代码的相关配置已经写死了，如果你想用多卡，可能需要自己调整。如果只是换显卡型号，不是多卡的话，就没问题，爆显存就调低两次训练的模型的batch_size即可。

5.软件说明：需要用到VPN。本项目是先跑通官方GPT-SoVITS，再进行修改的，所有官方需要下载的文件，本项目都有。第一次部署项目时，需要通过网络下载音频预处理的相关模型到C盘(官方最近更新采用在线下载的方式)，如：ASR等。如果内网下载缓慢，就尝试开VPN。

6.环境说明：尽量按照给的包版本来安装，没有的就默认。cuda11.8，torch==2.0.1

7.修改说明：
    
    a：将原版的半精度改成False，在主目录的config.py文件中,不然训练会报错

    b：切割音频时长参数-24

    c:项目一次一个进程，不支持多路。故官方模型存放的目录每次使用都会被刷新，即新生成的模型覆盖旧的模型。但历史模型和参考音频都存放在output

    d：E:\project\GPT-SoVITS-main\打包好\GPT-SoVITS\tools\asr\models中不能按原作者放入ASR魔塔模型文件，模型文件换成自动网络下载到了C:\Users\Administrator\.cache\modelscope\hub\iic中

    e：原onnxruntime换成onnxruntime-gpu，UVR5才能使用onnx_dereverb_By_FoxJoy显卡推理

    f：输入音频限制oss库链接，如果想本地上传删掉end_to_end.py的128行，download_file()，将输入写入“.\output\source”即可。

    g：项目流程

        （1）UVR5(python)，音频预处理。

            模型使用步骤：分离伴奏HP5_only_main_vocal->去和声onnx_dereverb_By_FoxJoy-VR(后来取消了，原因是用GPU        推理也很慢，占用了整个推理时间的30%)->去混响/去延迟DeEcho-Aggressive。

            转换存储过程为了加快速度，使用了mp3格式，比wav大小小10倍（即速度快很多），但是质量有所下降了（目前使用的mp3）

        （2）语音切分end_to_end.py。def slice(audio)。训练batch_size==18极限

        （3）ASR处理end_to_end.py。def asr(audio)

        （4）1A训练格式化工具 def GPT_SoVITS_1A(inp_text,inp_wav_dir)

        （5）1B微调end_to_end.py     def GPT_SoVITS_1B_step1()训练batch_size==18极限    def GPT_SoVITS_1B_step2(batch_size==21极限)

        （6）1C推理end_to_end.py    def inference()    

8.脚本说明：
（1）核心脚本为end_to_end.py,集成了训练过程中的每个流程。输入是原音频和结果命名。输出是克隆音频的demo。其次将模型文件保存再output文件夹中。

（2）api.py
            将ip改为本机ip

            接口文档件附件1，生成的内容做成了另一个回调接口，用户使用时直接接到参数就行，可以不进行回调直接获取结果。链接：https://pan.quark.cn/s/f1a42572d2e4

9.可能出现错误的说明：

（1）nltk_data错误解决办法：https://github.com/RVC-Boss/GPT-SoVITS/issues/848

（2）路径错误：注意绝对路径以及导包的绝对路径,UVR5的输入音频一定是绝对路径，这里已经改过

（3）训练的batch_size不要过高(建议15以内)，否则会出现保持不了模型的问题

（4）ffmeg的包可能要uninstall，再重装相应版本


项目2：
声音克隆的推理模块单独抽离，选择声音训练的历史模型文件，进行一键推理。前提是基于产品1的部署好。

1.根目录说明：同项目1相同，将项目放在E:\project下

2.脚本说明：

（1）process.py，输入是模型名称(E:\project\GPT-SoVITS-main\打包好\GPT-SoVITS\output\SOVITS_models)，需要克隆的文字，结果命名。输出是克隆的结果音频

（2）api.py

    将ip改为本机ip
    
    接口文档件附件2，生成的内容做成了另一个回调接口，用户使用可以不进行回调直接获取结果。链接：https://pan.quark.cn/s/4606f23229fe

                        
原文链接：https://blog.csdn.net/weixin_46412999/article/details/139474587
