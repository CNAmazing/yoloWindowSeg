import numpy as np

# 初始化窗口的参数（宽度和高度）
window_width = 1.0  # 窗口宽度
window_height = 1.5  # 窗口高度
window_spacing = 0.5  # 窗口之间的间距

# 初始窗口位置（假设有五个窗口，其中第四个窗口缺失）
initial_positions = [0.0, 1.4, 3.1, 4.5,None]

# 定义先验概率函数（假设宽度和高度在某范围内为合理）
def prior_probability(width, height, expected_width=1.0, expected_height=1.5):
    return np.exp(-abs(width - expected_width) - abs(height - expected_height))

# 定义似然函数（根据相邻窗口的位置判断合理性）
def likelihood_function(position, expected_position):
    return np.exp(-abs(position - expected_position))

# EM-MAP算法主循环
def EM_MAP_missing_window(positions, expected_width=1.0, expected_height=1.5, max_iterations=10, tolerance=1e-4):
    # 初始化缺失窗口的位置、宽度和高度
    missing_index = positions.index(None)
    current_position = positions[missing_index - 1] + window_spacing + expected_width
    current_width = expected_width
    current_height = expected_height

    # 开始迭代
    for i in range(max_iterations):
        # E步：计算缺失窗口的期望位置、宽度和高度
        expected_position = positions[missing_index - 1] + window_spacing + expected_width

        # M步：计算后验概率并更新参数
        prior = prior_probability(current_width, current_height)
        likelihood = likelihood_function(current_position, expected_position)
        posterior = prior * likelihood

        # 更新参数，使得后验概率最大
        new_position = expected_position
        new_width = expected_width
        new_height = expected_height

        # 检查收敛条件
        if abs(new_position - current_position) < tolerance and abs(new_width - current_width) < tolerance:
            print(f"Converged after {i + 1} iterations.")
            break

        # 更新当前值
        current_position = new_position
        current_width = new_width
        current_height = new_height

    # 返回推断的缺失窗口的最终参数
    return current_position, current_width, current_height

# 运行算法
inferred_position, inferred_width, inferred_height = EM_MAP_missing_window(initial_positions)
print(f"Inferred missing window position: {inferred_position}")
print(f"Inferred missing window width: {inferred_width}")
print(f"Inferred missing window height: {inferred_height}")
