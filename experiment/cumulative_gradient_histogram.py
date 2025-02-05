import numpy as np
import matplotlib.pyplot as plt
import os

def plot_saved_cumulative_gradient_change(npy_file_path, save_path, vmin=0, vmax=1.0):
    sum_cumulative_g_change = np.load(npy_file_path)
    plt.figure(figsize=(10, 1))
    heatmap = plt.imshow(
        sum_cumulative_g_change.reshape(1, -1),  # 1행으로 변환
        cmap="Reds",  # 흰색 → 빨간색 컬러맵
        aspect="auto",
        vmin=vmin,  # 최소값
        vmax=vmax   # 최대값
    )
    cbar = plt.colorbar(heatmap)
    cbar.ax.yaxis.set_ticks_position('right')  # 컬러바 위치 조정
    cbar.set_ticks(np.linspace(vmin, vmax, num=3))  # 균등 간격으로 5개의 tick 설정
    #cbar.set_label("Gradient Change", rotation=270, labelpad=15)  # 컬러바 레이블 추가
    plt.xlabel("Parameter Index")  # X축 레이블
    plt.gca().get_yaxis().set_visible(False)  # 세로축 숨김
    #plt.title("Cumulative Gradient Change", pad=20)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()



def plot_saved_cumulative_gradient_change_with_normalization(npy_file_path, save_path):
    sum_cumulative_g_change = np.load(npy_file_path)
    
    min_val = np.min(sum_cumulative_g_change)
    max_val = np.max(sum_cumulative_g_change)
    
    if max_val == min_val:
        normalized_data = np.zeros_like(sum_cumulative_g_change)
    else:
        normalized_data = (sum_cumulative_g_change - min_val) / (max_val - min_val)
    
    plt.figure(figsize=(5, 1))
    heatmap = plt.imshow(
        normalized_data.reshape(1, -1),  # 1행으로 변환
        cmap="Reds",  # 흰색 → 빨간색 컬러맵
        aspect="auto",
        vmin=0,  # 정규화 후 최소값은 0
        vmax=0.005  # 정규화 후 최대값은 1
    )
    # 컬러바를 가로로 아래 배치
    # cbar = plt.colorbar(heatmap, orientation='horizontal', pad=0.3)  # pad로 간격 조정
    # cbar.set_ticks(np.linspace(0, 0.005, num=3))  # tick 설정
    
    plt.xlabel("Parameter Index")  # X축 레이블
    plt.gca().get_yaxis().set_visible(False)  # 세로축 숨김
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def save_colorbar_only(vmin, vmax, cmap, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    # 별도의 Figure로 컬러바 생성
    fig, ax = plt.subplots(figsize=(5, 0.15))  # 가로로 길게 설정
    norm = plt.Normalize(vmin=vmin, vmax=vmax)  # 정규화
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 컬러바 추가 (가로 방향)
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_ticks(np.linspace(vmin, vmax, num=3))  # tick 설정
    cbar.ax.xaxis.set_ticks_position('bottom')  # tick 위치 조정

    # 컬러바 저장
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

#------------------------------------------------------------------------------------------------------------------
# dataset = "VLCS"
# alg = "ERM"
# envs = ["[0]", "[1]","[2]", "[3]"]
# files = [
#     "250120_14-31-58_ERM0_SGD_iter",
#     "250120_22-16-36_ERM1_SGD_iter",
#     "250121_02-28-17_ERM2_SGD_iter",
#     "250121_10-15-58_ERM3_SGD_iter"]

# for env, file in zip(envs, files):
#     file_path = os.path.join("/jsm0707/Large-scale/train_output/", dataset, alg, env, file)
#     input_path = os.path.join(file_path, "sum_cumulative_g_change.npy")
#     save_path = os.path.join(file_path, "output",  "cumulative_gradient_change_nor.png")
#     plot_saved_cumulative_gradient_change_with_normalization(input_path, save_path)

# dataset = "VLCS"
# alg = "ERM"
# envs = ["[0]", "[1]","[2]", "[3]"]
# files = [
#     "250120_14-32-48_ERM0_Adam_iter",
#     "250120_22-14-14_ERM1_Adam_iter",
#     "250121_02-26-00_ERM2_Adam_iter",
#     "250121_10-08-38_ERM3_Adam_iter"]
# for env, file in zip(envs, files):
#     file_path = os.path.join("/jsm0707/Large-scale/train_output/", dataset, alg, env, file)
#     input_path = os.path.join(file_path, "sum_cumulative_g_change.npy")
#     save_path = os.path.join(file_path, "output",  "cumulative_gradient_change_nor.png")
#     plot_saved_cumulative_gradient_change_with_normalization(input_path, save_path)

# dataset = "VLCS"
# alg = "SAM"
# envs = ["[0]", "[1]","[2]", "[3]"]
# files = [
#     "250125_02-32-15_resnet50_sgd",
#     "250125_10-29-56_resnet50_sgd",
#     "250125_15-12-40_resnet50_sgd",
#     "250125_23-14-38_resnet50_sgd"]
# for env, file in zip(envs, files):
#     file_path = os.path.join("/jsm0707/Large-scale/train_output/", dataset, alg, env, file)
#     input_path = os.path.join(file_path, "sum_cumulative_g_change.npy")
#     save_path = os.path.join(file_path, "output",  "cumulative_gradient_change_nor.png")
#     plot_saved_cumulative_gradient_change_with_normalization(input_path, save_path)

# dataset = "VLCS"
# alg = "gsnr1224"
# envs = ["[0]", "[1]","[2]", "[3]"]
# files = [
#     "250118_21-07-58_B_VLCS0_iter",
#     "250119_04-26-21_B_VLCS1_iter",
#     "250119_08-20-34_B_VLCS2_iter",
#     "250119_14-19-26_B_VLCS3_iter"]
# for env, file in zip(envs, files):
#     file_path = os.path.join("/jsm0707/Large-scale/train_output/", dataset, alg, env, file)
#     input_path = os.path.join(file_path, "sum_cumulative_g_change.npy")
#     save_path = os.path.join(file_path, "output",  "cumulative_gradient_change_nor.png")
#     plot_saved_cumulative_gradient_change_with_normalization(input_path, save_path)

#------------------------------------------------------------------------------------------------------------------
# dataset = "PACS"
# alg = "ERM"
# envs = ["[0]", "[1]","[2]", "[3]"]
# files = [
#     "250116_15-09-50_resnet50_adam (iter15000)",
#     "250116_16-33-38_resnet50_adam (iter15000)",
#     "250116_17-58-06_resnet50_adam (iter15000)",
#     "250116_19-23-12_resnet50_adam (iter15000)"]
# for env, file in zip(envs, files):
#     file_path = os.path.join("/jsm0707/Large-scale/train_output/", dataset, alg, env, file)
#     input_path = os.path.join(file_path, "sum_cumulative_g_change.npy")
#     save_path = os.path.join(file_path, "output",  "cumulative_gradient_change_nor.png")
#     plot_saved_cumulative_gradient_change_with_normalization(input_path, save_path)

# dataset = "PACS"
# alg = "SAM"
# envs = ["[0]", "[1]","[2]", "[3]"]
# files = [
#     "250125_02-30-11_resnet50_sgd",
#     "250125_04-46-06_resnet50_sgd",
#     "250125_07-02-14_resnet50_sgd",
#     "250125_09-20-05_resnet50_sgd"]
# for env, file in zip(envs, files):
#     file_path = os.path.join("/jsm0707/Large-scale/train_output/", dataset, alg, env, file)
#     input_path = os.path.join(file_path, "sum_cumulative_g_change.npy")
#     save_path = os.path.join(file_path, "output",  "cumulative_gradient_change_nor.png")
#     plot_saved_cumulative_gradient_change_with_normalization(input_path, save_path)

# dataset = "PACS"
# alg = "gsnr1224"
# envs = ["[0]", "[1]","[2]", "[3]"]
# files = [
#     "250118_21-07-58_B_VLCS0_iter",
#     "250119_04-26-21_B_VLCS1_iter",
#     "250119_08-20-34_B_VLCS2_iter",
#     "250119_14-19-26_B_VLCS3_iter"]
# for env, file in zip(envs, files):
#     file_path = os.path.join("/jsm0707/Large-scale/train_output/", dataset, alg, env, file)
#     input_path = os.path.join(file_path, "sum_cumulative_g_change.npy")
#     save_path = os.path.join(file_path, "output",  "cumulative_gradient_change_nor.png")
#     plot_saved_cumulative_gradient_change_with_normalization(input_path, save_path)
#
#------------------------------------------------------------------------------------------------------------------
dataset = "PACS"
alg = "ERM"
envs = ["[0]", "[1]","[2]", "[3]"]
files = [
    "250125_11-35-17_resnet50_sgd",
    "250125_13-05-29_resnet50_sgd",
    "250125_14-37-28_resnet50_sgd",
    "250125_16-09-14_resnet50_sgd"]

for env, file in zip(envs, files):
    file_path = os.path.join("/jsm0707/Large-scale/train_output/", dataset, alg, env, file)
    input_path = os.path.join(file_path, "sum_cumulative_g_change.npy")
    save_path = os.path.join(file_path, "output",  "cumulative_gradient_change_nor.png")
    plot_saved_cumulative_gradient_change_with_normalization(input_path, save_path)

save_colorbar_only(vmin=0, vmax=0.005, cmap="Reds", save_path="output/colorbar_only.png")