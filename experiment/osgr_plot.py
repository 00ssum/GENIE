import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

def read_tensorboard_values(folder_list, tag):
    data = {}
    for folder in folder_list:
        event_acc = EventAccumulator(folder)
        event_acc.Reload()
        if tag not in event_acc.Tags()['scalars']:
            print(f"Tag '{tag}' not found in folder '{folder}'")
            continue

        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        data[folder] = (steps, values)

    return data

def calculate_mean_per_interval(steps, values, interval=100):
    """
    100 iteration 단위로 평균을 계산합니다.
    
    Args:
        steps (list): 원본 스텝 리스트.
        values (list): 원본 값 리스트.
        interval (int): 평균 계산 간격.
        
    Returns:
        tuple: 평균화된 스텝 리스트와 값 리스트.
    """
    averaged_steps = []
    averaged_values = []

    for i in range(0, len(steps), interval):
        interval_steps = steps[i:i + interval]
        interval_values = values[i:i + interval]

        if len(interval_values) > 0:
            averaged_steps.append(np.mean(interval_steps))
            averaged_values.append(np.mean(interval_values))

    return averaged_steps, averaged_values

def plot_tensorboard_data(data, xlabel, ylabel, label_map=None, output_file=None):
    plt.figure(figsize=(5, 5))

    for folder, (steps, values) in data.items():
        # 100 iteration마다 평균 계산
        averaged_steps, averaged_values = calculate_mean_per_interval(steps, values, interval=200)

        # 폴더 이름을 매핑된 레이블로 변경
        label = label_map[folder] if label_map and folder in label_map else folder

        # 평균값 플롯
        plt.plot(averaged_steps, averaged_values, label=label, linewidth=1.5)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlim(0, 15000)  # x축 범위를 0~15000으로 고정
    plt.legend(loc='upper left', fontsize=12)  # 범례를 왼쪽 상단에 배치
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()

# 사용 예시
if __name__ == "__main__":
    # 사용자 입력
    # dataset = "VLCS"
    # env = "[1]"
    # tag = "te_L/OSGR/OSGR" 
    # output_file = str(dataset) + "_" + str(env) + ".png"

    # GENIE = "250119_04-26-21_B_VLCS1_iter"
    # SGD = "250120_22-16-36_ERM1_SGD_iter"
    # Adam = "250120_22-14-14_ERM1_Adam_iter"
    # SAM = "250125_10-29-56_resnet50_sgd"
    
    
    dataset = "VLCS"
    env = "[0]"
    tag = "te_C/OSGR/OSGR" 
    output_file = "output/"+str(dataset) + "_" + str(env) + ".png"

    GENIE = "250118_21-07-58_B_VLCS0_iter"
    SGD = "250120_14-31-58_ERM0_SGD_iter"
    Adam = "250120_14-32-48_ERM0_Adam_iter"
    SAM = "250125_02-32-15_resnet50_sgd"

    path = os.path.join("/jsm0707/Large-scale/train_output", dataset)

    folder_list = [ 
        os.path.join(path, "gsnr1224", env, GENIE),
        os.path.join(path, "ERM", env, SGD), 
        os.path.join(path, "ERM", env, Adam), 
        os.path.join(path, "SAM", env, SAM)

    ] 

    label_map = {
        os.path.join(path, "gsnr1224", env, GENIE): "GENIE",
        os.path.join(path, "ERM", env, SGD): "SGD",
        os.path.join(path, "ERM", env, Adam): "Adam",
        os.path.join(path, "SAM", env, SAM): "SAM"
    }

    data = read_tensorboard_values(folder_list, tag)
    plot_tensorboard_data(data, "Iteration", "OSGR", label_map=label_map, output_file=output_file)
