import pandas as pd
import os
from ultralytics import YOLO # Cần thêm cái này để lấy điểm Final

def fix_charts_and_add_final_report():
    # 1. Đường dẫn
    old_csv_path = "runs/detect/train/results.csv"   # 30 epoch đầu
    new_csv_path = "runs/detect/train4/results.csv"  # 20 epoch sau
    best_model_path = "runs/detect/train4/weights/best.pt" # Model xịn nhất 50 Epoch
    output_excel = "yolo_action_training_report_FINAL.xlsx"

    if not os.path.exists(old_csv_path) or not os.path.exists(new_csv_path):
        print("[LỖI] Không tìm thấy file CSV. Hãy kiểm tra lại thư mục train/train4")
        return

    # 2. Đọc và nối dữ liệu (Giữ nguyên logic của bạn)
    df_old = pd.read_csv(old_csv_path)
    df_new = pd.read_csv(new_csv_path)
    
    df_old.columns = [c.strip() for c in df_old.columns]
    df_new.columns = [c.strip() for c in df_new.columns]
    
    df_new['epoch'] = df_new['epoch'] + 30
    df_final_log = pd.concat([df_old, df_new], ignore_index=True)
    num_epochs = len(df_final_log)

    # --- PHẦN THÊM MỚI: LẤY DỮ LIỆU CHO SHEET FINAL TESTING RESULT ---
    print("[INFO] Đang chạy đánh giá model 50 Epoch để lấy bảng điểm...")
    model = YOLO(best_model_path)
    metrics = model.val(split='val') # Chạy đánh giá trên tập val/test

    final_metrics_data = {
        "Metric": ["Precision", "Recall", "mAP50", "mAP50-95"],
        "Test_Value": [
            metrics.results_dict['metrics/precision(B)'],
            metrics.results_dict['metrics/recall(B)'],
            metrics.results_dict['metrics/mAP50(B)'],
            metrics.results_dict['metrics/mAP50-95(B)']
        ]
    }
    df_testing_result = pd.DataFrame(final_metrics_data)

    # 3. Xuất Excel với 2 Sheet
    print(f"[INFO] Đang tạo file Excel với 2 sheet và biểu đồ...")
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    
    # Ghi sheet 1: Nhật ký training
    df_final_log.to_excel(writer, sheet_name='Training_Validation_Log', index=False)
    # Ghi sheet 2: Kết quả test cuối cùng (Đúng yêu cầu của bạn)
    df_testing_result.to_excel(writer, sheet_name='Final_Testing_Result', index=False)
    
    workbook  = writer.book
    log_sheet = writer.sheets['Training_Validation_Log']

    # --- BIỂU ĐỒ 1: LOSS (Sử dụng sheet Training_Validation_Log) ---
    chart_loss = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_loss.add_series({
        'name':       'Train Box Loss',
        'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Validation_Log', 1, 2, num_epochs, 2],
    })
    chart_loss.add_series({
        'name':       'Val Box Loss',
        'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Validation_Log', 1, 9, num_epochs, 9],
    })
    chart_loss.set_title({'name': 'Biểu đồ Training & Validation (Loss)'})
    chart_loss.set_x_axis({'name': 'Epoch'})
    chart_loss.set_y_axis({'name': 'Loss Value'})
    log_sheet.insert_chart('L2', chart_loss)

    # --- BIỂU ĐỒ 2: mAP50 ---
    chart_map = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_map.add_series({
        'name':       'mAP50 (Accuracy)',
        'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Validation_Log', 1, 7, num_epochs, 7],
        'line':       {'color': 'green'},
    })
    chart_map.set_title({'name': 'Biểu đồ Độ chính xác qua các Epoch (mAP50)'})
    chart_map.set_x_axis({'name': 'Epoch'})
    log_sheet.insert_chart('L18', chart_map)
    
    writer.close()
    print(f"[SUCCESS] Đã tạo xong file Excel FINAL có đầy đủ 2 Sheet: {output_excel}")

if __name__ == "__main__":
    fix_charts_and_add_final_report()