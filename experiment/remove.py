import os
import shutil

def delete_folders_without_jsonl(target_dir, dry_run=True):
    if not os.path.exists(target_dir):
        print(f"❌ 경로가 존재하지 않습니다: {target_dir}")
        return

    folders = [f.path for f in os.scandir(target_dir) if f.is_dir()]

    for folder in folders:
        jsonl_files = [f for f in os.listdir(folder) if f.endswith('.jsonl')]
        
        if not jsonl_files:  # `.jsonl` 파일이 없으면 삭제 대상
            print(f"❌ 삭제 대상: {folder}")

            if not dry_run:  # 실제 삭제 실행
                try:
                    shutil.rmtree(folder)
                    print(f"✅ 삭제 완료: {folder}")
                except Exception as e:
                    print(f"⚠ 삭제 실패: {folder} → {e}")

# 실행할 디렉토리 리스트 생성
base_directory = "/jsm0707/GENIE/train_output/VLCS/CORAL"
sub_dirs = [f"[{i}]" for i in range(0, 4)]  # ["[1]", "[2]", "[3]", "[4]"]

for sub_dir in sub_dirs:
    target_directory = os.path.join(base_directory, sub_dir)
    print(f"\n🔍 대상 디렉토리 확인: {target_directory}")
    delete_folders_without_jsonl(target_directory, dry_run=False)
