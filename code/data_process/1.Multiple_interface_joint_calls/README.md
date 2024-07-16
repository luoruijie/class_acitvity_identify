# 多接口联合调用

## data_for_train.py来实现对gpt4-o,qwen,doubao这几个api的处理。该处理包括对并行api的调用，每个接口的响应时间统计,是把三个接口跑出的程序写到一个文件里。

    data_for_train.py代码注释详细说明
    1.process_and_time 函数：
    
        功能：调用指定的API函数并统计其运行时间。
        参数：
            api_func：要调用的API函数。
            df：输入的DataFrame。
            api_name：API名称，用于打印和日志记录。
        返回值：
            result_df：API调用后的结果DataFrame。
            elapsed_time：API调用的运行时间。

    2.main 函数：
    
        功能：读取输入文件，调用指定的API处理数据，并将结果保存到输出文件。
        参数：
            input_file：输入的Excel文件路径。excel是已经分割好的excel
            output_file：输出的Excel文件路径。
            models：要运行的API模型列表，例如 ['qwen', 'doubao', 'gpt4o']。
    3.命令行参数解析：
            input_file：输入文件路径。
            output_file：输出文件路径。
            models：要运行的API模型列表，通过检查 models 列表中的字符串来创建 futures 任务。
    ## 命令行使用：python data_for_train.py --input_file input.xlsx --output_file output.xlsx --models qwen doubao


## main.py 的用途及功能

`main.py` 是一个用于处理Excel文件并通过调用不同API来处理数据的脚本。它可以并发调用多个API，将处理后的结果保存到指定的输出目录下。其主要功能包括：

1. **process_and_time**：调用指定的API函数并统计其运行时间，将API调用的结果添加到DataFrame的新列中，并记录运行时间和错误信息。
2. **load_api_function**：动态加载指定的API函数，从模块名称和函数名称字符串中解析出模块和函数，并返回对应的函数对象。
3. **generate_output_filename**：根据输入文件名称和模型名称生成输出文件名，并将其保存在指定的输出目录中。
4. **main**：主函数，负责协调整个流程，包括读取输入文件、读取配置文件、并发调用各个API处理数据，并根据输入文件名称和模型名称生成输出文件名，最后将处理结果保存到对应的输出文件中。

## config.json 在 main.py 中的作用

`config.json` 文件在 `main.py` 中起到以下作用：

1. **模型编号与模型名称的映射**：通过 `model_mapping` 对象，将模型编号映射到实际的模型名称。例如，编号 `1` 映射到 `qwen`，编号 `2` 映射到 `doubao`。
2. **API函数和列名的配置**：通过 `models` 对象，为每个模型名称配置对应的API函数名称和结果列名。这样在主程序中可以根据模型名称动态加载和调用对应的API函数，并将结果存储在指定的列中。

通过这种方式，`main.py` 可以灵活地根据配置文件调用不同的API，并处理输入的Excel数据，生成带有处理结果的输出文件。配置文件的使用使得新增或修改模型时，无需更改主程序代码，只需更新配置文件即可。

`main.py` 的使用:python your_script.py --input_file input.xlsx --output_dir ./data/调api跑出的数据 --models 1 2 --config_file config/config.json



## 其中对gpt4-o的调用和处理写到user_gpt4o_api.py中。

## 其中对qwen的调用和处理写到user_qwen_api.py中。

## 其中对doubao的调用和处理写到user_doubao_api.py中。

