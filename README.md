# 菜谱小管家

## 介绍

使用食谱相关的数据集对InternLM大模型进行微调，让其实现问它一道菜的做法，它可以清晰地告诉大家怎么做。

## OpenXlab 模型

&emsp;&emsp;菜谱小管家使用的是InternLM 的 7B 模型，模型参数量为 7B

## 数据集

&emsp;&emsp;菜谱小管家 数据集采用的数据为OpenDataLab上的公开数据集XiaChuFang Recipe Corpus，共计150百万余条，在训练采用了15万条数据。

&emsp;&emsp;数据集地址：https://opendatalab.com/OpenDataLab/XiaChuFang_Recipe_Corpus

&emsp;&emsp;数据集样例：

```text
{
"system": "你是一个经验丰富的专业厨师，用户提出自己想要做的菜肴时，你可以精确地回答出做菜的原料和做菜的方法",
"input": "自制腌菜的做法",
"output": "您需要准备以下食材:\n1500克上海青\n45克盐\n按以下方法制作:\n上海青（不用洗）放阳台上晒干表面的水份，然后放进干净的盆子里，一片一片掰开（雪里蕻不用掰），再放盐揉，感觉微微出水后就可，用盖子腌制一晚上\n第二天装进密封盒里放冰箱腌制一周左右。\n炒的时候拿出来清洗再浸泡半个小时"
}
```

<details><summary>数据集处理过程</summary>

> 下载数据集到本地并进行解压
>
> 将数据转为Xtuner的数据格式
>
> 原数据集文件格式：
> 
> ```test
>{"name": "水果果冻｜健康卫生", "dish": "Unknown", "description": "绵软的戚风蛋糕底，搭配上Q弹滑溜的芝士慕斯，顶层的白巧克力淋面酱奶香浓郁。\n\n三种滋味在口中交融，糟糕，是恋爱的感觉。", "recipeIngredient": ["60g水", "150g雪碧", "10g吉利丁片", "20g细砂糖", "3g柠檬汁", "西瓜", "奇异果"], "recipeInstructions": ["水+细砂糖倒锅里，小火加热至糖融化，关火", "吉利丁片用冷水泡软后，放入加热好的糖水中，搅拌至完全融化", "趁温热，在混合好的吉利丁液体中加入柠檬汁，混合均匀", "待液体冷却后，加入雪碧混合，过滤后成果冻液", "水果洗净后，切成想要的形状", "将水果放入容器中，倒入果冻液，入冰箱冷冻至凝结", "将凝结的果冻取出，用热毛巾在容器外敷10秒左右，就可以脱模享用啦"], "author": "author_15776", "keywords": ["水果果冻｜健康卫生的做法", "水果果冻｜健康卫生的家常做法", "水果果冻｜健康卫生的详细做法", "水果果冻｜健康卫生怎么做", "水果果冻｜健康卫生的最正宗做法", "甜品", "蛋糕", "戚风蛋糕"]}
> ```
>
> 目标格式：
>
> ```test
> 
> ```
> 

</details>

## 微调

&emsp;&emsp;使用 XTuner 训练， XTuner 有各个模型的一键训练脚本，很方便。且对 InternLM2 的支持度最高。

### XTuner

&emsp;&emsp;使用 XTuner 进行微调，具体脚本可参考`configs`文件夹下的脚本，脚本内有较为详细的注释。

|基座模型|配置文件|
|:---:|:---:|
|internlm-chat-7b|[internlm_chat_7b_qlora_e3_chineseMed.py](configs/internlm_chat_7b_qlora_e3_chineseMed.py)|
|internlm2-chat-7b|[internlm2_chat_7b_qlora_e3_chineseMed.py](configs/internlm2_chat_7b_qlora_e3_chineseMed.py)|

<details><summary>微调方法如下：</summary>

1. 根据基座模型复制上面的配置文件，将模型地址`pretrained_model_name_or_path`和数据集地址`data_path`修改成自己的，propmt模板`prompt_template`需要根据基座模型是InternLM还是InternLM2选择`PROMPT_TEMPLATE.internlm_chat`还是`PROMPT_TEMPLATE.internlm2_chat`，其他参数根据自己的需求修改，然后就可以开始微调（微调时间长的推荐使用tmux，免得万一和机器断开连接导致微调中断）

   ```bash
   xtuner train ${YOUR_CONFIG} --deepspeed deepspeed_zero2
   ```

   `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

2. 将保存的 `.pth` 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace Adapter 模型，即：生成 Adapter 文件夹：

   ```bash
   export MKL_SERVICE_FORCE_INTEL=1
   xtuner convert pth_to_hf ${YOUR_CONFIG} ${PTH} ${LoRA_PATH}
   ```

3. 将 HuggingFace Adapter 模型合并入 HuggingFace 模型：

    ```bash
    xtuner convert merge ${Base_PATH} ${LoRA_PATH} ${MERGED_PATH}
    ```

4. 若真的出现意外导致微调中段，可以从最近的 checkpoint 继续微调

   ```bash
   xtuner train ${YOUR_CONFIG} --deepspeed deepspeed_zero2 --resume ${LATEST_CHECKPOINT}
   ```

</details>

### Chat

微调结束后可以使用xtuner查看对话效果

```shell
xtuner chat ${MERGED_PATH} [optional arguments]
```

<details><summary>参数：</summary>
    
- `--prompt-template`: 指定对话模板，一代模型使用 internlm_chat，二代使用  internlm2_chat。
- `--system`:  指定SYSTEM文本
- `--system-template`:  指定SYSTEM模板
- `--bits`:  LLM位数，{4,8,None}。默认为 fp16。
- `--bot-name`:  bot名称
- `--with-plugins`:  指定要使用的插件
- `--no-streamer`:  是否启用流式传输
- `--lagent`:  是否使用lagent
- `--command-stop-word`:  命令停止词
- `--answer-stop-word`:  回答停止词
- `--offload-folder`:  存放模型权重的文件夹（或者已经卸载模型权重的文件夹）
- `--max-new-tokens`:  生成文本中允许的最大 token 数量
- `--temperature`:  温度值，对于二代模型，建议为0.8。
- `--top-k`:  保留用于顶k筛选的最高概率词汇标记数
- `--top-p`:  如果设置为小于1的浮点数，仅保留概率相加高于 top_p 的最小一组最有可能的标记，对于二代模型，建议为0.8。
- `--repetition-penalty`: 防止文本重复输出，对于二代模型，个人建议1.01，对于一代模型可不填。
- `--seed`:  用于可重现文本生成的随机种子
- `-h`:  查看参数。
  
</details>

## OpenXLab 部署 中医药知识问答助手

&emsp;&emsp;仅需要 Fork [此仓库](https://github.com/xiaomile/medKnowledgeAssitant)，然后在 OpenXLab 上创建一个新的项目，将 Fork 的仓库与新建的项目关联，即可在 OpenXLab 上部署 中医药知识问答助手。

&emsp;&emsp;***OPenXLab 中医药知识问答助手  https://openxlab.org.cn/apps/detail/xiaomile/medKnowledgeAssitant***
&emsp;&emsp;***OPenXLab 中医药知识问答助手（4bit）  https://openxlab.org.cn/apps/detail/xiaomile/personal_assistant2_4bit***

![Alt text](images/openxlab.png)
![4bit](images/openxlab2.png)

## LmDeploy部署

- 首先安装LmDeploy

```shell
pip install -U 'lmdeploy[all]==v0.2.0'
```

- 然后转换模型为`turbomind`格式。使用 TurboMind 推理模型需要先将模型转化为 TurboMind 的格式，，目前支持在线转换和离线转换两种形式。TurboMind 是一款关于 LLM 推理的高效推理引擎，基于英伟达的 FasterTransformer 研发而成。它的主要功能包括：LLaMa 结构模型的支持，persistent batch 推理模式和可扩展的 KV 缓存管理器。
本项目采用离线转换，需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式，如下所示。

> --dst-path: 可以指定转换后的模型存储位置。

```shell
lmdeploy convert internlm-chat-7b  要转化的模型地址 --dst-path ./workspace 转换后模型的存放地址
```
执行完成后将会在当前目录生成一个 workspace 的文件夹。

- LmDeploy Chat对话。模型转换完成后，我们就具备了使用模型推理的条件，接下来就可以进行真正的模型推理环节。
1、本地对话（Bash Local Chat）模式，它是跳过API Server直接调用TurboMind。简单来说，就是命令行代码直接执行 TurboMind。
```shell
lmdeploy chat turbomind ./workspace #转换后的turbomind模型地址
```

- 网页Demo演示。本项目采用将TurboMind推理作为后端，将Gradio作为前端Demo演示。
```shell
# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace #转换后的turbomind模型地址
```
就可以直接启动 Gradio，此时没有API Server，TurboMind直接与Gradio通信。
- 原始模型运行，显存占用56%

## Lmdeploy&opencompass 量化以及量化评测  
> 进行量化决策流程
> Step1:尝试正常版本，评估效果。效果一般，启动量化。
> Step2:开展KV Cache量化，以减少中间过程计算结果对显存的占用。评估量化效果。
### `KV Cache`量化 
- 计算与获得量化参数
  >计算 minmax。主要思路是通过计算给定输入样本在每一层不同位置处计算结果的统计情况。
  >在计算minmax的命令行中，会选择128条输入样本，每条样本长度为 2048，数据集选择ptb，输入模型后就会得到上面的各种统计值。
```shell
# 计算 minmax
lmdeploy lite calibrate \
  ./internlm-chat-7b/  #模型绝对路径 \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --work-dir ./quant_output #参数保存路径
  --trust_remote_code=True
```
  >通过minmax获取量化参数。主要利用下面公式来获取每一层的KV中心值（zp）和缩放值（scale）。
```shell
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp
```
  >有了这两个值就可以进行量化和反量化操作。具体来说就是对历史存储中的K和V做量化，使用时再反量化。使用如下命令：
```shell
# 通过 minmax 获取量化参数
lmdeploy lite kv_qparams \
   ./quant_output #保存kv计算结果的路径 \
   workspace/triton_models/weights/ #转换后模型的存放路径 \
  --num_tp 1
```
- 修改配置。修改weights/config.ini文件，把quant_policy改为4，从而打开KV int8开关。
```shell
tensor_para_size = 1
session_len = 2056
max_batch_size = 64
max_context_token_num = 1
step_length = 1
cache_max_entry_count = 0.5
cache_block_seq_len = 128
cache_chunk_size = 1
use_context_fmha = 1
quant_policy = 4
max_position_embeddings = 2048
rope_scaling_factor = 0.0
use_logn_attn = 0
```
  >至此就完成了KV Cache量化。
  >开始对话
```shell
lmdeploy chat turbomind /root/chinesemedical/workspace --model-format hf  --quant-policy 4
```

- 评估量化效果。编写评测文件`configs/eval_turbomind.py`
```python
from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel

with read_base():
 # choose a list of datasets   
  from .datasets.ceval.ceval_gen import ceval_datasets 
 # and output the results in a choosen format
  from .summarizers.medium import summarizer

datasets = [*ceval_datasets]

internlm2_chat_7b = dict(
     type=TurboMindModel,
     abbr='internlm2-chat-7b-turbomind',
     path='转换后的模型地址',
     engine_config=dict(session_len=512,
         max_batch_size=2,
         rope_scaling_factor=1.0),
     gen_config=dict(top_k=1,
         top_p=0.8,
         temperature=1.0,
         max_new_tokens=100),
     max_out_len=100,
     max_seq_len=512,
     batch_size=2,
     concurrency=1,
     #  meta_template=internlm_meta_template,
     run_cfg=dict(num_gpus=1, num_procs=1),
)
models = [internlm2_chat_7b]
```
- 启动评测！
```shell
python run.py configs/eval_turbomind.py -w 指定结果保存路径
```
- 单独做KV Cache量化，显存占用55%，无明显优化！
  
> Step3:开展W4A16量化，以减少模型参数计算结果对显存的占用。评估量化效果。W4A16中的A是指Activation，保持FP16，只对部分权重参数进行4bit量化
### `W4A16`量化 
- 计算与获得量化参数
  >计算 minmax。主要思路是通过计算给定输入样本在每一层不同位置处计算结果的统计情况。
  >在计算minmax的命令行中，会选择128条输入样本，每条样本长度为 2048，数据集选择ptb，输入模型后就会得到上面的各种统计值。
```shell
# 计算 minmax
lmdeploy lite calibrate \
  ./internlm-chat-7b/  #模型绝对路径 \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --work-dir ./quant_output #参数保存路径
  --trust_remote_code=True
```
- 量化权重模型
  >利用上面得到的统计值对参数进行量化。
  >执行如下命令：
```shell
# 量化权重模型
lmdeploy lite auto_awq \
  ./internlm-chat-7b/   #未量化前模型的存放路径 \
  --calib-dataset 'ptb' \
  --calib-samples 128 \ 
  --calib-seqlen 2048 \
  --w-bits 4 \
  --w_group_size 128 \
  --work_dir ./internlm-chat-7b-4bit #量化后模型的存放路径
```
  >命令中 w_bits表示量化的位数，w_group_size表示量化分组统计的尺寸，work_dir是量化后模型输出的位置。
  >因为没有 torch.int4，所以实际存储时，8个4bit权重会被打包到一个int32值中。
- 转换成 TurboMind 格式（也可以跳过这一步，直接启动对话）
```shell
# 转换模型的layout，存放在默认路径 ./workspace 下
lmdeploy convert  internlm-chat-7b ./internlm-chat-7b-4bit/ #W4A16量化后的模型路径\
    --model-format awq \
    --group-size 128
    --dst_path ./workspace_4bit #转换后模型的存放路径
``` 
  >这个group-size就是那个w_group_size。可以指定输出目录：--dst_path。
  >至此就完成了W4A16量化。
- 启动对话
```shell
lmdeploy chat turbomind ./workspace_4bit --model-format awq
``` 
- 评估量化效果。评测文件`configs/eval_turbomind.py`如上
- 启动评测！
```shell
python run.py configs/eval_turbomind.py -w 结果保存路径
```
结果文件可在同目录文件[results](./results)中获取
- 单独做W4A16量化，显存占用64%，较未量化前模型占用内存更大！


> Step4:同步开启KV Cache量化和W4A16量化，以减少中间过程计算结果和模型参数计算结果对显存的占用。
- 获取对W4A16量化后模型的KV Cache量化参数
```shell
lmdeploy lite kv_qparams \
   ./quant_output/  \  # 存放之前kv cache计算结果的文件夹路径
  workspace_4bit/triton_models/weights/ \ # 存放本次kv cache量化后参数的文件夹路径
  --num-tp 1
```
- 修改参数
对workspace_4bit/triton_models/weights/config.ini文件进行参数修改
```shell
cache_max_entry_count = 0.2
quant_policy = 4
```
- 启动对话
```shell
lmdeploy chat turbomind ./workspace_4bit/  --model-format awq --quant-policy 4
```
- 同步开启KV Cache量化和W4A16量化，显存占用34%，有明显优化效果！


## OpenCompass 评测

- 安装 OpenCompass

```shell
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

- 在opencompass/configs目录下新建自定义数据集测评配置文件 `eval_internlm_7b_custom.py` 和 `eval_internlm_chat_turbomind_api_custom.py`

```python
from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    from .summarizers.medium import summarizer

datasets = [
    {"path": "/root/ChineseMedicalAssistant/test_qa.jsonl", "data_type": "qa", "infer_method": "gen"}, # your custom dataset
]

internlm_chat_7b = dict(
       type=HuggingFaceCausalLM,
       # `HuggingFaceCausalLM` 的初始化参数
       path='/root/ChineseMedicalAssistant/merged', # your model path
       tokenizer_path='/root/ChineseMedicalAssistant/merged', # your model path
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto',trust_remote_code=True),
       # 下面是所有模型的共同参数，不特定于 HuggingFaceCausalLM
       abbr='internlm_chat_7b',               # 结果显示的模型缩写
       max_seq_len=2048,             # 整个序列的最大长度
       max_out_len=100,              # 生成的最大 token 数
       batch_size=64,                # 批量大小
       run_cfg=dict(num_gpus=1),     # 该模型所需的 GPU 数量
    )

models=[internlm_chat_7b]
```

```python
from mmengine.config import read_base
from opencompass.models.turbomind_api import TurboMindAPIModel

with read_base():
    from .summarizers.medium import summarizer

datasets = [
    {"path": "/root/ChineseMedicalAssistant/test_qa.jsonl", "data_type": "qa", "infer_method": "gen"}, # your custom dataset
]


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)

models = [
    dict(
        type=TurboMindAPIModel,
        abbr='internlm-chat-7b-turbomind',
        path="./model/workspace_4bit",
        api_addr='http://0.0.0.0:23333',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
```

- 评测启动！

```shell
python run.py configs/eval_internlm_7b_custom.py
```

- 量化评测，先启动turbomind作为服务端

```shell
lmdeploy serve api_server ./workspace_4bit --server_name 0.0.0.0 --server_port 23333 --instance_num 64 --tp 1
```

```shell
python run.py eval_internlm_chat_turbomind_api_custom.py
```


## 致谢

<div align="center">

***感谢上海人工智能实验室组织的 书生·浦语实战营 学习活动~***

***感谢 OpenXLab 对项目部署的算力支持~***

***感谢 浦语小助手 对项目的支持~***
</div>
