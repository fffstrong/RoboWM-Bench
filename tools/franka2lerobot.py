import pickle
import numpy as np

def inspect_large_pkl(file_path):
    """
    查看大容量pkl文件的结构，递归输出第一个元素的 keys 和对应的 size 或元素的内容。
    :param file_path: pkl文件路径
    """
    # 1. 加载pkl文件（带异常处理）
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("✅ 成功加载pkl文件")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {file_path}")
        return
    except Exception as e:
        print(f"❌ 加载文件失败：{str(e)}")
        return

    # 2. 检查数据类型
    if not isinstance(data, list):
        print(f"❌ 数据类型为 {type(data)}，不是列表，无法处理")
        return

    print(f"数据类型: {type(data)}")
    print(f"数据长度: {len(data)}")

    # 3. 递归输出第一个元素的 keys 和对应的 size 或内容
    first_elem = data[0]
    print("\n===== 第一个元素的结构和 size =====")

    def print_structure(element, depth=0):
        indent = "  " * depth  # 缩进
        if isinstance(element, dict):
            print(f"{indent}字典，键值对数量: {len(element)}")
            for key, value in element.items():
                if isinstance(value, (list, dict, tuple)):
                    print(f"{indent}键: {key}, 元素类型: {type(value)}, 大小: {len(value)}")
                    print_structure(value, depth + 1)
                elif isinstance(value, np.ndarray):
                    if value.size <= 20:  # 如果数组大小小于等于10，直接输出内容
                        print(f"{indent}键: {key}, 元素类型: {type(value)}, 内容: {value}")
                    else:  # 否则输出大小
                        print(f"{indent}键: {key}, 元素类型: {type(value)}, 大小: {value.shape}")
                elif isinstance(value, (int, float, bool, str)):
                    print(f"{indent}键: {key}, 元素类型: {type(value)}, 内容: {value}")
                else:
                    print(f"{indent}键: {key}, 元素类型: {type(value)}, 大小: 无法计算")
        elif isinstance(element, list):
            print(f"{indent}列表，长度: {len(element)}")
            if len(element) > 0:
                print(f"{indent}第一个元素类型: {type(element[0])}")
                print_structure(element[0], depth + 1)
        elif isinstance(element, tuple):
            print(f"{indent}元组，长度: {len(element)}")
            if len(element) > 0:
                print(f"{indent}第一个元素类型: {type(element[0])}")
                print_structure(element[0], depth + 1)
        else:
            print(f"{indent}值类型: {type(element)}")

    if isinstance(first_elem, dict):
        print_structure(first_elem)
    else:
        print(f"第一个元素不是字典，类型为: {type(first_elem)}")

# ===================== 调用示例 =====================
if __name__ == "__main__":
    # 替换成你的pkl文件路径
    pkl_file_path = "/home/feng/lehome_1/open_success_20_demos_2026-01-28_09-32-40.pkl"
    # 调用函数
    inspect_large_pkl(pkl_file_path)