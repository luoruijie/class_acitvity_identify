## 这个code中包含了数据预处理，模型训练（全参微调） ，加载训练好的模型推理等步骤。 其中训练代码和推理代码需要在Linux环境下才能运行成功。 Linux环境用的是
## autodl云平台，训练环境基本配置得是：gpu显存需要48G，推理环境基本配置是：gpu显存需要24G。其它包的安装请查看主目录下requirementst.txt。
使用指南
1. 数据获取
首先，请联系青岛组的陈麒麟，获取课程资料。课程资料是一个zip文件，包含teacher.mp4和students.mp4两个文件。获取后，将文件名更改为teacher-full.mp4和students-full.mp4，然后将它们存放在我这台电脑的WSL-Debian系统中，路径为/home/soikit/videos/LLM。
接下来，在WSL-Debian系统中，进入/home/soikit/bj20/bj20目录，运行以下脚本：./reame_run.sh。该脚本用于将音频转换为文本，成功运行后，每个课堂目录下会生成一个asr_voice.xlsx文件。请将每个课堂目录下的onnx-9999.db文件更名为origin.db。注：你也可以直接让陈麒麟生成这些asr_voice.xlsx文件。
完成这些修改后，请将更名后的asr_voice.xlsx文件交给苗艺萌。她会运行调用GPT-4接口的代码，最终生成combined_file_splitted.xlsx文件。该文件包含课堂活动的分析过程和对应的课堂活动类别。GPT-4接口的代码位于class_activity_identity代码仓库下，链接为：https://github.com/luoruijie/class_acitvity_identify，更多使用细节请查看该项目下的README.md。
至此，数据获取工作结束。
2. 模型训练
在获取到combined_file_splitted.xlsx文件后，我们将调用class_activity_identity/train目录下的data_process_for_train.ipynb文件，生成train.txt和dev.txt两个文件。接着，你需要在Huggingface上创建账户，并新建一个数据集，将train.txt和dev.txt上传至该数据集。上传完成后，请修改训练代码中加载数据集的名称，替换为你创建的dataset名称（如your_name/dataset_name）。然后，你可以启动模型训练，训练代码为train_and_save.py。如有需要，请咨询苗艺萌，她会协助你完成模型训练。
3. 模型推理
模型训练完成后，我们将加载推理代码。推理代码位于class_activity_identity/infer/infer.py，具体使用方法请参考文件中的注释。如果有疑问，可咨询苗艺萌。
4. 环境配置
当前的模型训练是在autodl.com网站上进行，选择的显卡为L20，这意味着所有的模型加载和保存都在L20专区内完成。由于我已经将训练和推理的环境保存为一个镜像，你在启动新实例时，需选择“我的镜像”，并选择名为train_and_infer的镜像。
启动后即进入训练环境。如果训练完成后需要进行推理，请激活demo这个conda环境，具体切换指令为：conda activate demo。在demo环境下，你可以按照推理指令进行模型推理操作。

