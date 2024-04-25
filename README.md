# 菜谱小管家

## 目录

- [菜谱小管家](#菜谱小管家)
  - [目录](#目录)
  - [介绍](#介绍)
  - [模型](#模型)
  - [数据集](#数据集)
  - [微调](#微调)
     - [XTuner](#xtuner)
  - [部署](#部署)
     - [模型转换](#模型转换)
     - [TurboMind推理+API服务](#TurboMind推理+API服务)
     - [网页demo演示](#网页demo演示)
  - [OpenCompass评测](#OpenCompass评测)

## 介绍

&emsp;&emsp;使用食谱相关的数据集对InternLM大模型进行微调，让其实现问它一道菜的做法，它可以清晰地告诉大家怎么做。

## 模型

&emsp;&emsp;菜谱小管家使用的是InternLM 的 7B 模型，模型参数量为 7B

## 数据集

&emsp;&emsp;菜谱小管家 数据集采用的数据为OpenDataLab上的公开数据集XiaChuFang Recipe Corpus，共计150百万余条，在训练采用了15万条数据。

&emsp;&emsp;数据集地址：https://opendatalab.com/OpenDataLab/XiaChuFang_Recipe_Corpus

&emsp;&emsp;由于处理后的数据集依旧过大，无法放入github中，我们附上处理过的使用的数据集放置地址：https://jrcd8xtxn1.feishu.cn/wiki/TCeRwPZaAiPW6PkWIYJcbftwnqG?from=from_copylink

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
> [{
>    "conversation":[
>       {
>          "system": "xxx",
>          "input": "xxx",
>          "output": "xxx"
>       }
>    ]
> }]
> ```
>
> 通过python脚本进行转换，python代码如下：
>
> ```python
>import json
>
>input_file = 'recipe_corpus_full.json'
>output_prefix = 'tran_'
>records_per_file = 100000
>
>start_index = 0
>
>
>with open(input_file, 'r', encoding='utf-8') as file:
>    lines = file.readlines()
>
> 
>    row_num = 0
>    conversations = []
>
>
>with open(input_file, 'r', encoding='utf-8') as file:
>    lines = file.readlines()
>
>    row_num = 0
>    conversations = []
>
>    for line in lines:
>        data = json.loads(line) 
>        keywords = data['keywords'] 
>        recipeIngredient = data['recipeIngredient']  
>        recipeInstructions = data['recipeInstructions'] 
>        author = data['author']  
>        system_message = f"你是一个经验丰富的专业厨师，用户提出自己想要做的菜肴时，你可以精>确地回答出做菜的原料和做菜的方法" 
>        input_message = keywords[0]  
>
>        
>        formatted_ingredients = "\n".join(recipeIngredient)
>        formatted_instructions = "\n".join(recipeInstructions)
>
>        output_message = f"您需要准备以下食材:\n{formatted_ingredients}\n按以下方法制作:\n{formatted_instructions}" 
>        new_record = {
>            "conversation": [
>                {
>                    "system": system_message,
>                    "input": input_message,
>                    "output": output_message
>                }
>            ]
>        }
>        conversations.append(new_record)  
>        row_num += 1  
>        if row_num % records_per_file == 0:  
>            with open(f'{output_prefix}{start_index}.json', 'w', encoding='utf-8') as >file:
>                json.dump(conversations, file, ensure_ascii=False,
>                          indent=4)  
>            start_index += 1 
>            conversations = []  
> ```
>
> 此处生成10个.json文件，本项目中随机抽取一个.json文件作为微调的训练集
</details>

## 微调

&emsp;&emsp;使用 XTuner 训练， XTuner 有各个模型的一键训练脚本，很方便。且对 InternLM2 的支持度最高。

### XTuner

&emsp;&emsp;使用 XTuner 进行微调，具体操作如下：

- 安装：

```bash
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 2.0.1 的环境：
/root/share/install_conda_env_internlm_base.sh xtuner0.1.9
# 如果你是在其他平台：
conda create --name xtuner0.1.9 python=3.10 -y

# 激活环境
conda activate xtuner0.1.9
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir xtuner019 && cd xtuner019


# 拉取 0.1.9 的版本源码
git clone -b v0.1.9  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.9 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd xtuner

# 从源码安装 XTuner
pip install -e '.[all]'
```
- 安装结束后，创建文件夹

 ```bash
mkdir RecipeAssistant && cd /root/RecipeAssistant
 ```

- 准备配置文件

 ```bash
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
 ```

- 下载模型文件

 ```bash
ln -s /share/temp/model_repos/internlm-chat-7b .
 ```

- 导入训练集

 ```bash
mkdir data 
#将本地的.json文件直接拖到该文件夹下即可
 ```

- 修改配置文件

 ```bash
# 复制配置文件到当前目录
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
# 改个文件名
mv internlm_chat_7b_qlora_oasst1_e3_copy.py internlm_chat_7b_qlora_medqa2019_e3.py

# 修改配置文件内容
vim internlm_chat_7b_qlora_medqa2019_e3.py
 ```

减号代表要删除的行，加号代表要增加的行

 ```bash
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据为 MedQA2019-structured-train.jsonl 路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = 'MedQA2019-structured-train.jsonl'

# 修改 train_dataset 对象
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
 ```

- 开始调试

 ```bash
xtuner train internlm_chat_7b_qlora_medqa2019_e3.py --deepspeed deepspeed_zero2
 ```

- pth转huggingface

 ```bash
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm_chat_7b_qlora_medqa2019_e3.py ./work_dirs/internlm_chat_7b_qlora_medqa2019_e3/epoch_1.pth ./hf
 ```

- 部署与测试

  将HuggingFace adapter合并到大语言模型

```bash
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```

  与合并后的模型对话

```bash
# 4 bit 量化加载
xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```

- 部署智能对话Demo

```bash
conda activate internlm-demo
cd /root/RecipeAssistant
#clone代码
mkdir code &&cd code
git clone https://gitee.com/internlm/InternLM.git
#切换 commit 版本，与教程 commit 版本保持一致
cd InternLM
git checkout 3028f07cb79e5b1d7342f4ad8d11efad3fd13d17
```

- Web demo运行

```bash
cd root/RecipeAssistant/code/InternLM
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
```

  需要配置本地端口进行访问
  

## 部署 

### 模型转换

转换模型为`turbomind`格式。使用 TurboMind 推理模型需要先将模型转化为 TurboMind 的格式，目前支持在线转换和离线转换两种形式。TurboMind 是一款关于 LLM 推理的高效推理引擎，基于英伟达的 FasterTransformer 研发而成。它的主要功能包括：LLaMa 结构模型的支持，persistent batch 推理模式和可扩展的 KV 缓存管理器。
本项目采用离线转换，需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式

- 离线转换

```bash
lmdeploy convert internlm-chat-7b  ~/RecipeAssistant/merged
```

转换成功，并在RecipeAssistant文件夹下生成workplace文件夹

### TurboMind推理+API服务

```bash
lmdeploy serve api_server ./workspace \
        --server_name 0.0.0.0 \
        --server_port 23333 \
        --instance_num 64 \
        --tp 1
```

### 网页Demo演示

- 分窗口输入命令

```bash
lmdeploy serve gradio http://0.0.0.0:23333 \
        --server_name 0.0.0.0 \
        --server_port 6006 \
        --restful_api True
```
 
- 配置本地端口

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38793
```

- 打开localhost:6006,在页面中就能与模型进行对话

## OpenCompass评测

- 安装 OpenCompass

```shell
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

- 下载解压数据集

```shell
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/RecipeAssistant/opencompass/
unzip OpenCompassData-core-20231110.zip
#将会在RecipeAssistant/opencompass下看到data文件夹
```

- 启动评测

```bash
export MKL_SERVICE_FORCE_INTEL=1
python run.py 
--datasets ceval_gen 
--hf-path /root/RecipeAssistant/merged/ 
--tokenizer-path /root/RecipeAssistant/merged/ 
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True 
--model-kwargs trust_remote_code=True device_map='auto' 
--max-seq-len 1024 
--max-out-len 16 
--batch-size 2  
--num-gpus 1  
--debug
```

- 评测结果格式如下

```bash
dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_RecipeAssistant_merged
----------------------------------------------  ---------  -------------  ------  -------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                   21.05
ceval-operating_system                          1c2571     accuracy       gen                                                                   36.84                                                             30.11
```

结果文件可在同目录文件[results](./results)中获取
