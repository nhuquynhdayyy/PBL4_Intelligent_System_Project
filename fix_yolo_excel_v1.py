import pandas as pd
import os

def fix_charts():
    # 1. Đường dẫn tới các file kết quả bạn đã có (Đảm bảo đúng tên thư mục nhé)
    old_csv_path = "runs/detect/train/results.csv"   # 30 epoch đầu
    new_csv_path = "runs/detect/train4/results.csv"  # 20 epoch sau (train4 theo ảnh của bạn)
    output_excel = "yolo_action_training_report_FINAL.xlsx"

    if not os.path.exists(old_csv_path) or not os.path.exists(new_csv_path):
        print("[LỖI] Không tìm thấy file CSV. Hãy kiểm tra lại tên thư mục train/train3 trong runs/detect/")
        return

    # 2. Đọc và nối dữ liệu
    df_old = pd.read_csv(old_csv_path)
    df_new = pd.read_csv(new_csv_path)
    
    df_old.columns = [c.strip() for c in df_old.columns]
    df_new.columns = [c.strip() for c in df_new.columns]
    
    df_new['epoch'] = df_new['epoch'] + 30
    df_final = pd.concat([df_old, df_new], ignore_index=True)
    num_epochs = len(df_final)

    # 3. Vẽ lại biểu đồ với CỘT CHUẨN
    print(f"[INFO] Đang vẽ lại biểu đồ cho {num_epochs} Epoch...")
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    df_final.to_excel(writer, sheet_name='Training_Log', index=False)
    
    workbook  = writer.book
    log_sheet = writer.sheets['Training_Log']

    # --- BIỂU ĐỒ 1: LOSS (PHẢI ĐI XUỐNG) ---
    chart_loss = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_loss.add_series({
        'name':       'Train Box Loss',
        'categories': ['Training_Log', 1, 0, num_epochs, 0], # Cột A: Epoch
        'values':     ['Training_Log', 1, 2, num_epochs, 2], # Cột C: train/box_loss (Index 2)
    })
    chart_loss.add_series({
        'name':       'Val Box Loss',
        'categories': ['Training_Log', 1, 0, num_epochs, 0], # Cột A: Epoch
        'values':     ['Training_Log', 1, 9, num_epochs, 9], # Cột J: val/box_loss (Index 9 theo file của bạn)
    })
    chart_loss.set_title({'name': 'Biểu đồ Training & Validation (Loss)'})
    chart_loss.set_x_axis({'name': 'Epoch', 'min': 0, 'max': 50})
    chart_loss.set_y_axis({'name': 'Loss Value'})
    log_sheet.insert_chart('L2', chart_loss)

    # --- BIỂU ĐỒ 2: mAP50 (PHẢI ĐI LÊN) ---
    chart_map = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_map.add_series({
        'name':       'mAP50 (Accuracy)',
        'categories': ['Training_Log', 1, 0, num_epochs, 0], # Cột A: Epoch
        'values':     ['Training_Log', 1, 7, num_epochs, 7], # Cột H: metrics/mAP50(B) (Index 7)
        'line':       {'color': 'green'},
    })
    chart_map.set_title({'name': 'Biểu đồ Độ chính xác qua các Epoch (mAP50)'})
    chart_map.set_x_axis({'name': 'Epoch', 'min': 0, 'max': 50})
    log_sheet.insert_chart('L18', chart_map)
    
    writer.close()
    print(f"[SUCCESS] Đã tạo xong file Excel chuẩn: {output_excel}")

if __name__ == "__main__":
    fix_charts()