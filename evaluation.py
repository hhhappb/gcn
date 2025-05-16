# 文件名: evaluation.py (通用版 - 支持动态 Feeder 加载)
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import logging

# --- 导入你的模型 ---
from model.SDT_GRUs_Gesture import SDT_GRU_Classifier # 假设模型是固定的

# --- 导入工具函数 ---
from utils import StandardScaler, move2device, get_logger, collate_fn_filter_none, import_class # <<<--- 确保 import_class 可用

# 获取一个 logger 实例
logger = logging.getLogger(__name__)

def evaluate_gesture_model(model_cfg_path, dataset_cfg_path, weight_path, log_dir, device='cuda'):
    eval_logger = get_logger(log_dir, name='evaluation_process')
    eval_logger.info(f"Evaluating model specified in {model_cfg_path} with weights {weight_path}")
    eval_logger.info(f"Using dataset specified in {dataset_cfg_path}")

    # --- 加载模型配置 ---
    try:
        with open(model_cfg_path, "r") as f:
            model_cfg_full = yaml.load(f, Loader=yaml.FullLoader)
            model_params = model_cfg_full.get('model', model_cfg_full.get('model_args', {}))
            if not model_params:
                 eval_logger.error(f"模型配置文件 {model_cfg_path} 中未找到 'model' 或 'model_args' 部分。")
                 return
    except FileNotFoundError:
        eval_logger.error(f"模型配置文件未找到: {model_cfg_path}"); return
    except Exception as e:
        eval_logger.error(f"加载或解析模型配置文件 {model_cfg_path} 失败: {e}"); return

    # --- 加载数据集配置 ---
    try:
        with open(dataset_cfg_path, "r") as f:
            full_cfg = yaml.load(f, Loader=yaml.FullLoader)
            feeder_args = full_cfg.get('test_feeder_args',
                                       full_cfg.get('dataset',
                                       full_cfg.get('feeder_args', {})))
            if not feeder_args:
                eval_logger.error(f"数据集配置文件 {dataset_cfg_path} 中未找到 feeder 参数块。")
                return

            # <<<--- 新增：获取 Feeder 类字符串 --- >>>
            feeder_class_str = full_cfg.get('feeder_class', feeder_args.get('feeder_class'))
            if not feeder_class_str:
                # 尝试从顶层 'feeder' 或 'processor_cfg' 获取，兼容旧格式
                feeder_class_str = full_cfg.get('feeder', full_cfg.get('processor_cfg',{}).get('feeder'))
                if not feeder_class_str:
                    eval_logger.error(f"数据集配置文件 {dataset_cfg_path} 中未指定 'feeder_class'。")
                    return
            eval_logger.info(f"将使用的 Feeder 类: {feeder_class_str}")

    except FileNotFoundError:
        eval_logger.error(f"数据集配置文件未找到: {dataset_cfg_path}"); return
    except Exception as e:
        eval_logger.error(f"加载或解析数据集配置文件 {dataset_cfg_path} 失败: {e}"); return


    # --- 参数协调与校验 (与之前类似，但更通用) ---
    # 1. Feeder 参数准备
    if 'split' not in feeder_args: # 大多数feeder都需要split参数
        feeder_args['split'] = 'test'
        eval_logger.info(f"Feeder_args 中未指定 'split', 默认为 'test'")

    # 检查 feeder_args 中是否有必要的参数 (这些是比较通用的，具体 Feeder 可能有更多)
    # root_dir, data_path, label_path (如果feeder需要) 是常见的
    # modalities, num_nodes, base_channel 对于骨骼数据 feeder 很重要
    # SHREC17 可能用 list_file 代替 label_path
    # DHG14-28 用 subject_idx
    # 建议每个Feeder的__init__自行做更严格的参数检查

    # 2. 模型参数与 Feeder/数据集参数同步
    # num_classes
    if 'num_classes' not in model_params:
        if 'num_classes' in feeder_args: model_params['num_classes'] = feeder_args['num_classes']
        else: eval_logger.error("未找到 'num_classes'"); return
    feeder_args['num_classes'] = model_params['num_classes']

    # max_len / max_seq_len
    if 'max_seq_len' not in model_params:
        if 'max_len' in feeder_args: model_params['max_seq_len'] = feeder_args['max_len']
        else: eval_logger.error("未找到序列长度参数"); return
    feeder_args['max_len'] = model_params['max_seq_len']

    # num_nodes (如果 Feeder 提供，则同步到模型)
    if 'num_nodes' in feeder_args:
        if 'num_nodes' not in model_params or model_params['num_nodes'] != feeder_args['num_nodes']:
            eval_logger.info(f"同步模型 'num_nodes' 为 Feeder 的: {feeder_args['num_nodes']}")
        model_params['num_nodes'] = feeder_args['num_nodes']
    elif 'num_nodes' not in model_params: # 如果 Feeder 也没有，模型也没有，则报错
        eval_logger.error("模型和数据配置中均未找到 'num_nodes'"); return
    # 如果 Feeder 没有 num_nodes 但模型有，则假设模型配置的是对的，Feeder 内部可能会使用它

    # num_input_dim (根据 feeder 的 modalities 和 base_channel 计算)
    if 'modalities' in feeder_args and 'base_channel' in feeder_args:
        try:
            num_modalities = len(str(feeder_args['modalities']).split(','))
            base_channel = int(feeder_args['base_channel'])
            calculated_num_input_dim = base_channel * num_modalities
            if model_params.get('num_input_dim') != calculated_num_input_dim:
                eval_logger.warning(f"模型配置 'num_input_dim' (原: {model_params.get('num_input_dim')}) "
                                   f"将被根据 feeder_args 计算的值 ({calculated_num_input_dim}) 覆盖。")
            model_params['num_input_dim'] = calculated_num_input_dim
        except Exception as e:
            eval_logger.error(f"从 feeder_args 计算 num_input_dim 失败: {e}"); return
    elif 'num_input_dim' not in model_params: # 如果 feeder 没有提供计算依据，模型也没有，则报错
        eval_logger.error("无法确定 'num_input_dim' (模型配置无，且feeder_args缺少modalities/base_channel)"); return

    eval_logger.info(f"最终模型参数 (部分): num_classes={model_params.get('num_classes')}, "
                   f"max_seq_len={model_params.get('max_seq_len')}, num_nodes={model_params.get('num_nodes')}, "
                   f"num_input_dim={model_params.get('num_input_dim')}")
    eval_logger.info(f"最终Feeder参数 (部分): { {k: feeder_args[k] for k in ['split', 'max_len', 'num_classes', 'root_dir', 'data_path'] if k in feeder_args} }")


    # --- 加载数据 ---
    try:
        # <<<--- 动态导入并实例化 Feeder --- >>>
        FeederClass = import_class(feeder_class_str)
        eval_logger.info(f"使用 {FeederClass.__name__} 实例化测试数据集，参数: {feeder_args}")
        test_set = FeederClass(**feeder_args)

        batch_size = int(full_cfg.get('test_batch_size', feeder_args.get('batch_size', 32)))
        num_workers = int(feeder_args.get('num_worker', full_cfg.get('num_worker', 0)))

        test_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn_filter_none, # 通用 collate_fn
                                 pin_memory=True)
        eval_logger.info(f"测试数据加载器 ({FeederClass.__name__}) 创建成功，批次大小: {batch_size}, "
                       f"工作进程数: {num_workers}, 样本数: {len(test_set)}")
    except ImportError:
        eval_logger.error(f"无法导入 Feeder 类: {feeder_class_str}", exc_info=True); return
    except Exception as e:
        eval_logger.error(f"创建测试数据集 ({feeder_class_str}) 或加载器失败: {e}", exc_info=True); return


    class DummyScaler: # 对于骨骼数据，通常不需要额外缩放
        def transform(self, data): return data
    scaler = DummyScaler()
    eval_logger.info("使用 DummyScaler (不进行额外的数据缩放)。")


    # --- 加载模型 ---
    try:
        model = SDT_GRU_Classifier(model_cfg=model_params)
    except Exception as e:
        eval_logger.error(f"实例化模型 SDT_GRU_Classifier 失败: {e}\n模型配置: {model_params}", exc_info=True); return

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        eval_logger.info(f"模型权重从 {weight_path} 加载成功。")
    except FileNotFoundError:
        eval_logger.error(f"模型权重文件未找到: {weight_path}"); return
    except Exception as e:
        eval_logger.error(f"加载模型权重失败: {e}", exc_info=True); return

    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    if not test_loader or len(test_loader) == 0:
        eval_logger.error("测试数据加载器为空或未正确初始化。"); return

    dl = tqdm(test_loader, desc="Testing", unit="batch")
    processed_samples_count = 0
    for batch_idx, batch in enumerate(dl):
        if batch is None:
            eval_logger.warning(f"批次 {batch_idx} 为 None (可能所有样本都被过滤)，跳过。")
            continue

        try:
            # 不同 Feeder 返回的元组长度和内容可能略有不同
            # 假设标准返回是 (data, label, mask, index) 或 (data, label, mask)
            # Feeder_NTU, Feeder_SHREC17, Feeder_UCLA, Feeder_DHG1428 都应该是 (data, label, mask, index)
            if len(batch) == 4:
                x, labels, mask_tensor, _ = batch # 忽略索引
            elif len(batch) == 3: # 兼容可能只返回 data, label, mask 的 feeder
                x, labels, mask_tensor = batch
            else:
                eval_logger.error(f"批次 {batch_idx} 数据元组长度 ({len(batch)}) 不符合预期 (3或4)。跳过此批次。")
                continue
            processed_samples_count += x.size(0)
        except ValueError as e: # 解包错误
            eval_logger.error(f"解包批次 {batch_idx} 数据失败: {e}. 跳过此批次。", exc_info=True)
            if isinstance(batch, (list, tuple)):
                 eval_logger.error(f"批次长度: {len(batch)}, 各元素类型: {[type(b) for b in batch]}")
            continue
        except Exception as e_unpack:
            eval_logger.error(f"解包批次 {batch_idx} 时发生意外错误: {e_unpack}", exc_info=True); continue


        x = scaler.transform(x)
        x, labels, mask_tensor = move2device([x, labels, mask_tensor], device)

        with torch.no_grad():
            try:
                # 假设模型输入是 (B, T, V, C_final) 和 mask (B, T)
                logits, _ = model(x, mask_tensor)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
            except Exception as e_forward:
                eval_logger.error(f"模型前向传播失败 (批次 {batch_idx}): {e_forward}", exc_info=True)
                eval_logger.error(f"输入 x 形状: {x.shape}, mask_tensor 形状: {mask_tensor.shape if mask_tensor is not None else 'None'}")
                continue

    eval_logger.info(f"总共处理了 {processed_samples_count} 个样本进行评估。")

    # --- 计算指标 ---
    if not all_logits:
         eval_logger.error("评估过程中没有成功处理任何 logits。")
         return

    try:
        all_logits_tensor = torch.cat(all_logits, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
    except Exception as e_cat:
        eval_logger.error(f"拼接 logits 或 labels 时失败: {e_cat}", exc_info=True); return

    if all_logits_tensor.shape[0] == 0 or all_labels_tensor.shape[0] == 0:
        eval_logger.error("评估后 all_logits_tensor 或 all_labels_tensor 为空。"); return
    if all_logits_tensor.shape[0] != all_labels_tensor.shape[0]:
        eval_logger.error(f"评估后 logits ({all_logits_tensor.shape[0]}) 和 labels ({all_labels_tensor.shape[0]}) 数量不匹配。"); return

    all_preds_tensor = torch.argmax(all_logits_tensor, dim=1)

    accuracy = accuracy_score(all_labels_tensor.numpy(), all_preds_tensor.numpy())
    # 确保 target_names 的数量与 num_classes 匹配
    num_classes_for_report = model_params.get('num_classes', torch.max(all_labels_tensor).item() + 1)
    class_names_from_cfg = model_params.get('class_names')
    if class_names_from_cfg and len(class_names_from_cfg) == num_classes_for_report:
        target_names = class_names_from_cfg
    else:
        if class_names_from_cfg: # 数量不匹配
             eval_logger.warning(f"模型配置中的 class_names 数量 ({len(class_names_from_cfg)}) 与 num_classes ({num_classes_for_report}) 不符。将使用默认名称。")
        target_names = [f'C{i}' for i in range(num_classes_for_report)]

    unique_present_labels = np.unique(all_labels_tensor.numpy())
    # labels_for_report 应该是从0到num_classes-1的完整列表，以确保报告结构完整
    labels_param_for_report = np.arange(num_classes_for_report)


    report = classification_report(all_labels_tensor.numpy(), all_preds_tensor.numpy(),
                                   labels=labels_param_for_report,
                                   target_names=target_names,
                                   zero_division=0)
    cm = confusion_matrix(all_labels_tensor.numpy(), all_preds_tensor.numpy(), labels=labels_param_for_report)

    eval_logger.info(f"Test Accuracy: {accuracy:.4f}")
    eval_logger.info("Classification Report:\n" + report)
    eval_logger.info("Confusion Matrix:\n" + np.array2string(cm))

    results_df = pd.DataFrame({'true': all_labels_tensor.numpy(), 'pred': all_preds_tensor.numpy()})
    results_df.to_csv(os.path.join(log_dir, 'test_predictions.csv'), index=False)
    with open(os.path.join(log_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    eval_logger.info(f"评估结果已保存到目录: {log_dir}")

if __name__ == "__main__":
    # --- 设置评估所需的参数 ---
    # **重要**: 你需要创建这些配置文件和目录结构，并提供真实的模型权重路径

    # --- 示例：评估 NTU 数据集 ---
    print("\n--- 示例：评估 NTU 数据集 ---")
    # 1. 创建虚拟 NTU 数据集和模型配置文件 (如果不存在)
    os.makedirs('./config', exist_ok=True)
    os.makedirs('./log/NTU_Eval', exist_ok=True)
    os.makedirs('./log/NTU_Train_Dummy', exist_ok=True) # 假设有一个训练日志目录
    os.makedirs('./debug_data', exist_ok=True) # feeder_ntu.py 的测试块会在这里创建数据

    ntu_model_yaml_content = """
model:
  num_classes: 60
  max_seq_len: 100
  # num_input_dim: 6 # 会被 evaluation.py 根据 feeder_args 动态计算和覆盖
  num_nodes: 25      # NTU 有25个节点
  # SDT_GRU_Classifier 的其他特定参数
  num_rnn_layers: 2
  num_rnn_units: 128
  n_heads: 4
  ffn_dim: 256
  st_layers: 1
  st_dropout_rate: 0.1
  use_global_spatial_bias: true
  zoneout_rate: 0.05
"""
    ntu_model_yaml_file = './config/ntu_model_config.yaml'
    with open(ntu_model_yaml_file, 'w') as f: f.write(ntu_model_yaml_content)

    ntu_dataset_yaml_content = f"""
feeder_class: "feeders.feeder_ntu.Feeder_NTU" # <<<--- 指定 Feeder 类

test_feeder_args:
  root_dir: './debug_data'
  data_path: 'dummy_ntu_tdgcn_style.npz' # 由 feeder_ntu.py 的 if __name__ == '__main__' 创建
  label_path: 'dummy_ntu_tdgcn_style.npz'
  label_source: 'from_data_npz'
  split: 'test'
  max_len: 100        # 与模型配置的 max_seq_len 匹配
  modalities: "joint,bone"
  num_nodes: 25       # 与模型配置的 num_nodes 匹配
  base_channel: 3
  num_classes: 60     # 与模型配置的 num_classes 匹配
  num_worker: 0

test_batch_size: 8
"""
    ntu_dataset_yaml_file = './config/ntu_dataset_config.yaml'
    with open(ntu_dataset_yaml_file, 'w') as f: f.write(ntu_dataset_yaml_content)

    # 创建虚拟模型权重 (与之前的 evaluation.py __main__ 类似)
    temp_model_cfg_dict_ntu = yaml.load(ntu_model_yaml_content, Loader=yaml.FullLoader)['model']
    _temp_feeder_args_ntu = yaml.load(ntu_dataset_yaml_content, Loader=yaml.FullLoader)['test_feeder_args']
    _num_modalities_ntu = len(str(_temp_feeder_args_ntu['modalities']).split(','))
    _base_channel_ntu = int(_temp_feeder_args_ntu['base_channel'])
    temp_model_cfg_dict_ntu['num_input_dim'] = _num_modalities_ntu * _base_channel_ntu
    # temp_model_cfg_dict_ntu['num_nodes'] 已在yaml中

    dummy_model_ntu = SDT_GRU_Classifier(model_cfg=temp_model_cfg_dict_ntu)
    ntu_weight_path = './log/NTU_Train_Dummy/best_ntu.pt'
    torch.save(dummy_model_ntu.state_dict(), ntu_weight_path)
    print(f"已创建虚拟 NTU 模型权重文件: {ntu_weight_path}")
    del dummy_model_ntu

    # 确保 feeder_ntu.py 的测试块已运行以创建虚拟数据
    dummy_ntu_datafile = os.path.join('./debug_data', 'dummy_ntu_tdgcn_style.npz')
    if not os.path.exists(dummy_ntu_datafile):
        print(f"警告: 虚拟 NTU 数据文件 ({dummy_ntu_datafile}) 未找到。")
        print("请确保已运行 `python feeders/feeder_ntu.py` 来生成它，或者脚本会自动尝试生成。")
        # 尝试调用 feeder_ntu.py 的主函数来生成数据 (这是一种 hacky 的方式)
        try:
            print("尝试自动生成虚拟 NTU 数据...")
            # 这种方式可能因环境和 feeder_ntu.py 的 __main__ 结构而失败
            # from feeders import feeder_ntu
            # if hasattr(feeder_ntu, '__main__') and callable(feeder_ntu.__main__):
            #     feeder_ntu.__main__() # 不推荐这种直接调用
            # 最好是手动运行或通过 subprocess
            import subprocess
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'feeders', 'feeder_ntu.py')], check=True)
            print("虚拟 NTU 数据生成命令已执行。")

        except Exception as e_gen:
            print(f"自动生成虚拟 NTU 数据失败: {e_gen}")
            print("请手动运行 `python feeders/feeder_ntu.py`")


    if os.path.exists(dummy_ntu_datafile):
        print(f"找到虚拟 NTU 数据，开始评估 NTU 模型...")
        evaluate_gesture_model(ntu_model_yaml_file, ntu_dataset_yaml_file, ntu_weight_path, './log/NTU_Eval', 'cpu')
    else:
        print(f"跳过 NTU 评估，因为虚拟数据文件 {dummy_ntu_datafile} 未找到。")


    # --- 示例：评估 SHREC'17 数据集 ---
    print("\n--- 示例：评估 SHREC'17 数据集 ---")
    os.makedirs('./log/SHREC17_Eval', exist_ok=True)
    os.makedirs('./log/SHREC17_Train_Dummy', exist_ok=True)
    # SHREC17 的数据需要特定的 JSON 结构和列表文件，这里不自动创建虚拟数据，假设用户已准备
    # 创建虚拟 SHREC17 配置文件
    shrec_model_yaml_content = """
model:
  num_classes: 14 # 或 28，取决于 label_type
  max_seq_len: 180
  num_nodes: 22     # SHREC'17 有22个节点
  # SDT_GRU_Classifier 的其他特定参数 (可以与NTU不同)
  num_rnn_layers: 2
  num_rnn_units: 96 # 调整以适应 SHREC 特征
  n_heads: 3
  ffn_dim: 192
  st_layers: 1
  st_dropout_rate: 0.1
  use_global_spatial_bias: false # 示例：SHREC 可能不需要
  zoneout_rate: 0.0
"""
    shrec_model_yaml_file = './config/shrec_model_config.yaml'
    with open(shrec_model_yaml_file, 'w') as f: f.write(shrec_model_yaml_content)

    shrec_dataset_yaml_content = f"""
feeder_class: "feeders.feeder_shrec17.Feeder" # <<<--- 指定 SHREC Feeder

test_feeder_args:
  root_dir: 'data/shrec/shrec17_jsons'  # <<<--- 用户需要提供SHREC数据路径
  list_file: 'data/shrec/shrec17_jsons/test_samples.json' # <<<--- 用户需要提供列表文件
  split: 'test'
  label_type: 'label_14'
  max_len: 180
  modalities: "joint"
  num_nodes: 22
  base_channel: 3
  num_classes: 14
  num_worker: 0

test_batch_size: 8
"""
    shrec_dataset_yaml_file = './config/shrec_dataset_config.yaml'
    with open(shrec_dataset_yaml_file, 'w') as f: f.write(shrec_dataset_yaml_content)

    # 创建虚拟SHREC模型权重
    temp_model_cfg_dict_shrec = yaml.load(shrec_model_yaml_content, Loader=yaml.FullLoader)['model']
    _temp_feeder_args_shrec = yaml.load(shrec_dataset_yaml_content, Loader=yaml.FullLoader)['test_feeder_args']
    _num_modalities_shrec = len(str(_temp_feeder_args_shrec['modalities']).split(','))
    _base_channel_shrec = int(_temp_feeder_args_shrec['base_channel'])
    temp_model_cfg_dict_shrec['num_input_dim'] = _num_modalities_shrec * _base_channel_shrec
    # temp_model_cfg_dict_shrec['num_nodes'] 已在yaml中

    dummy_model_shrec = SDT_GRU_Classifier(model_cfg=temp_model_cfg_dict_shrec)
    shrec_weight_path = './log/SHREC17_Train_Dummy/best_shrec.pt'
    torch.save(dummy_model_shrec.state_dict(), shrec_weight_path)
    print(f"已创建虚拟 SHREC'17 模型权重文件: {shrec_weight_path}")
    del dummy_model_shrec

    # 检查用户是否已准备 SHREC 数据
    shrec_root_dir_check = 'data/shrec/shrec17_jsons' # 与上面配置一致
    shrec_list_file_check = 'data/shrec/shrec17_jsons/test_samples.json'
    if os.path.isdir(shrec_root_dir_check) and os.path.exists(shrec_list_file_check):
        print(f"找到 SHREC'17 数据目录和列表文件，开始评估 SHREC'17 模型...")
        evaluate_gesture_model(shrec_model_yaml_file, shrec_dataset_yaml_file, shrec_weight_path, './log/SHREC17_Eval', 'cpu')
    else:
        print(f"跳过 SHREC'17 评估，因为数据目录 '{shrec_root_dir_check}' 或列表文件 '{shrec_list_file_check}' 未找到。")
        print("请确保已按 SHREC17 Feeder 期望的结构准备数据。")

    # 可以添加更多数据集的评估示例，例如 UCLA, DHG14-28
    # 确保你的 feeder_ucla.py 和 feeder_dhg14_28.py 文件存在于 feeders/ 目录下
    # 并创建相应的配置文件和虚拟权重。

    # --- 示例：评估 UCLA 数据集 (需要 feeder_ucla.py) ---
    print("\n--- 示例：评估 UCLA 数据集 ---")
    os.makedirs('./log/UCLA_Eval', exist_ok=True)
    os.makedirs('./log/UCLA_Train_Dummy', exist_ok=True)

    # 检查 feeder_ucla.py 是否存在
    ucla_feeder_path = os.path.join(os.path.dirname(__file__), 'feeders', 'feeder_ucla.py')
    if not os.path.exists(ucla_feeder_path):
        print(f"警告: feeder_ucla.py 未在 {ucla_feeder_path} 找到。跳过 UCLA 评估示例。")
    else:
        ucla_model_yaml_content = """
model:
  num_classes: 10 # UCLA 有10个类别
  max_seq_len: 64  # UCLA Feeder v1.7 默认 max_len 或 window_size
  num_nodes: 20     # UCLA 有20个节点
  num_rnn_layers: 2; num_rnn_units: 128; n_heads: 4; ffn_dim: 256; st_layers: 1
"""
        ucla_model_yaml_file = './config/ucla_model_config.yaml'
        with open(ucla_model_yaml_file, 'w') as f: f.write(ucla_model_yaml_content)

        ucla_dataset_yaml_content = f"""
feeder_class: "feeders.feeder_ucla.Feeder"

test_feeder_args:
  root_dir: 'data/ucla_data_json'  # <<<--- 用户需要提供 UCLA JSON 数据路径
  split: 'val' # UCLA feeder 使用 'train'/'val'
  data_path: 'joint,bone' # 模态
  max_len: 64
  # label_path: 可以不提供，因为它用硬编码列表
  num_classes: 10
  num_nodes: 20 # Feeder 内部定义为20
  base_channel: 3
  num_worker: 0
test_batch_size: 8
"""
        ucla_dataset_yaml_file = './config/ucla_dataset_config.yaml'
        with open(ucla_dataset_yaml_file, 'w') as f: f.write(ucla_dataset_yaml_content)

        temp_model_cfg_dict_ucla = yaml.load(ucla_model_yaml_content, Loader=yaml.FullLoader)['model']
        _temp_feeder_args_ucla = yaml.load(ucla_dataset_yaml_content, Loader=yaml.FullLoader)['test_feeder_args']
        _num_modalities_ucla = len(str(_temp_feeder_args_ucla['data_path']).split(',')) # UCLA feeder用data_path作modalities
        _base_channel_ucla = int(_temp_feeder_args_ucla['base_channel'])
        temp_model_cfg_dict_ucla['num_input_dim'] = _num_modalities_ucla * _base_channel_ucla

        dummy_model_ucla = SDT_GRU_Classifier(model_cfg=temp_model_cfg_dict_ucla)
        ucla_weight_path = './log/UCLA_Train_Dummy/best_ucla.pt'
        torch.save(dummy_model_ucla.state_dict(), ucla_weight_path)
        print(f"已创建虚拟 UCLA 模型权重文件: {ucla_weight_path}")
        del dummy_model_ucla

        ucla_root_dir_check = 'data/ucla_data_json'
        if os.path.isdir(ucla_root_dir_check):
            print(f"找到 UCLA 数据目录，开始评估 UCLA 模型...")
            evaluate_gesture_model(ucla_model_yaml_file, ucla_dataset_yaml_file, ucla_weight_path, './log/UCLA_Eval', 'cpu')
        else:
            print(f"跳过 UCLA 评估，因为数据目录 '{ucla_root_dir_check}' 未找到。")
            print("请确保已按 UCLA Feeder 期望的结构准备 JSON 数据。")