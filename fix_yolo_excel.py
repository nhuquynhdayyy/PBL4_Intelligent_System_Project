import pandas as pd
import os
from ultralytics import YOLO

def fix_charts_pro():
    # 1. Đường dẫn
    old_csv_path = "runs/detect/train/results.csv"
    new_csv_path = "runs/detect/train4/results.csv"
    best_model_path = "runs/detect/train4/weights/best.pt"
    output_excel = "yolo_action_training_report_FINAL.xlsx"

    if not os.path.exists(old_csv_path) or not os.path.exists(new_csv_path):
        print("[LỖI] Không tìm thấy file CSV. Hãy kiểm tra lại thư mục train/train4")
        return

    # 2. Đọc và nối dữ liệu
    df_old = pd.read_csv(old_csv_path)
    df_new = pd.read_csv(new_csv_path)
    
    df_old.columns = [c.strip() for c in df_old.columns]
    df_new.columns = [c.strip() for c in df_new.columns]
    
    df_new['epoch'] = df_new['epoch'] + 30
    df_final = pd.concat([df_old, df_new], ignore_index=True)
    num_epochs = len(df_final)

    # --- LẤY DỮ LIỆU TEST CUỐI CÙNG ---
    print("[INFO] Đang chạy đánh giá model để lấy thông số Test...")
    model = YOLO(best_model_path)
    metrics = model.val(split='val', verbose=False)
    df_test = pd.DataFrame({
        "Metric": ["Precision", "Recall", "mAP50", "mAP50-95"],
        "Test_Value": [
            metrics.results_dict['metrics/precision(B)'],
            metrics.results_dict['metrics/recall(B)'],
            metrics.results_dict['metrics/mAP50(B)'],
            metrics.results_dict['metrics/mAP50-95(B)']
        ]
    })

    # 3. Ghi file Excel
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    df_final.to_excel(writer, sheet_name='Training_Log', index=False)
    df_test.to_excel(writer, sheet_name='Final_Testing_Result', index=False)
    
    workbook  = writer.book
    log_sheet = writer.sheets['Training_Log']
    test_sheet = writer.sheets['Final_Testing_Result']

    # --- ĐỊNH DẠNG: Freeze Panes & Autosize ---
    header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
    
    for sheet in [log_sheet, test_sheet]:
        sheet.freeze_panes(1, 0) # Cố định hàng 1
        # Autosize đơn giản: đặt độ rộng cột cố định hoặc tính toán
        sheet.set_column('A:Z', 15)

    # --- BIỂU ĐỒ 1: LOSS (PHẢI ĐI XUỐNG) ---
    chart_loss = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_loss.add_series({
        'name': 'Train Box Loss',
        'categories': ['Training_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Log', 1, 2, num_epochs, 2],
    })
    chart_loss.add_series({
        'name': 'Val Box Loss',
        'categories': ['Training_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Log', 1, 9, num_epochs, 9],
    })
    chart_loss.set_title({'name': 'Biểu đồ Training & Validation (Loss)'})
    chart_loss.set_x_axis({'name': 'Epoch'})
    log_sheet.insert_chart('L2', chart_loss)

    # --- BIỂU ĐỒ 2: mAP50 (PHẢI ĐI LÊN) ---
    chart_map = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_map.add_series({
        'name': 'mAP50 (Accuracy)',
        'categories': ['Training_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Log', 1, 7, num_epochs, 7],
        'line': {'color': 'green'},
    })
    chart_map.set_title({'name': 'Biểu đồ Độ chính xác (mAP50)'})
    log_sheet.insert_chart('L18', chart_map)

    # --- BIỂU ĐỒ 3: PRECISION & RECALL (PHẢI ĐI LÊN) ---
    chart_pr = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_pr.add_series({
        'name': 'Precision',
        'categories': ['Training_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Log', 1, 5, num_epochs, 5], # Cột F
        'line': {'color': 'blue'},
    })
    chart_pr.add_series({
        'name': 'Recall',
        'categories': ['Training_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Log', 1, 6, num_epochs, 6], # Cột G
        'line': {'color': 'orange'},
    })
    chart_pr.set_title({'name': 'Biểu đồ Precision & Recall qua các Epoch'})
    log_sheet.insert_chart('L34', chart_pr)

    # --- BIỂU ĐỒ 4: BAR CHART (Ở SHEET TESTING) ---
    chart_bar = workbook.add_chart({'type': 'column'})
    chart_bar.add_series({
        'name':       'Giá trị kiểm thử',
        'categories': ['Final_Testing_Result', 1, 0, 4, 0],
        'values':     ['Final_Testing_Result', 1, 1, 4, 1],
        'data_labels': {'value': True, 'position': 'outside_end'},
    })
    chart_bar.set_title({'name': 'Tổng hợp kết quả kiểm thử cuối cùng'})
    chart_bar.set_y_axis({'min': 0, 'max': 1})
    test_sheet.insert_chart('D2', chart_bar)

    writer.close()
    print(f"[SUCCESS] Đã tạo xong báo cáo FINAL chuyên nghiệp: {output_excel}")

if __name__ == "__main__":
    fix_charts_pro()