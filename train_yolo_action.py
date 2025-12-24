from ultralytics import YOLO
import pandas as pd
import os

# def train_and_report():
#     # 1. Khởi tạo mô hình YOLOv8 Nano
#     # model = YOLO("yolov8n.pt") 
#     model = YOLO("runs/detect/train/weights/last.pt") 

#     # 2. Bắt đầu Training
#     # print("[INFO] Đang bắt đầu huấn luyện YOLOv8...")
#     # # Sửa epochs=1 để test, sửa lại 30 hoặc 50 khi chạy thật
#     # results = model.train(
#     #     data="action_data/data.yaml", 
#     #     epochs=30, 
#     #     imgsz=640, 
#     #     device='cpu' 
#     # )
#     print("[INFO] Đang chạy tiếp tục từ Epoch 31 đến 50...")
#     results = model.train(
#         resume=True,   # Kích hoạt chế độ chạy tiếp
#         epochs=50      # Đặt mục tiêu mới là 50 (YOLO sẽ tự hiểu là cần chạy thêm 20 vòng nữa)
#     )

#     # 3. Lấy đường dẫn kết quả
#     train_dir = results.save_dir
#     csv_path = os.path.join(train_dir, "results.csv")
#     output_excel = "yolo_action_training_report.xlsx"

#     if not os.path.exists(csv_path):
#         print("[LỖI] Không tìm thấy kết quả training.")
#         return

#     # 4. Chạy giai đoạn TESTING (Kiểm thử cuối cùng)
#     print("[INFO] Đang chạy giai đoạn TESTING...")
#     test_results = model.val(split='test') 
    
#     test_metrics_df = pd.DataFrame({
#         'Metric': ['Precision', 'Recall', 'mAP50', 'mAP50-95'],
#         'Test_Value': [
#             test_results.results_dict['metrics/precision(B)'],
#             test_results.results_dict['metrics/recall(B)'],
#             test_results.results_dict['metrics/mAP50(B)'],
#             test_results.results_dict['metrics/mAP50-95(B)']
#         ]
#     })

#     # 5. Đọc dữ liệu CSV để vẽ biểu đồ
#     df = pd.read_csv(csv_path)
#     df.columns = [c.strip() for c in df.columns]
#     num_epochs = len(df)

#     # 6. Xuất Excel và TỰ ĐỘNG VẼ BIỂU ĐỒ
#     print("[INFO] Đang khởi tạo file Excel và vẽ biểu đồ...")
#     writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    
#     # Ghi dữ liệu training
#     df.to_excel(writer, sheet_name='Training_Validation_Log', index=False)
#     # Ghi dữ liệu testing
#     test_metrics_df.to_excel(writer, sheet_name='Final_Testing_Result', index=False)
    
#     workbook  = writer.book
#     log_sheet = writer.sheets['Training_Validation_Log']

#     # --- BIỂU ĐỒ 1: TRAINING & VALIDATION LOSS ---
#     # Thể hiện quá trình "Training" và "Validation" theo yêu cầu của thầy
#     chart_loss = workbook.add_chart({'type': 'line'})
#     chart_loss.add_series({
#         'name':       'Train Box Loss',
#         'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
#         'values':     ['Training_Validation_Log', 1, 1, num_epochs, 1], # cột train/box_loss
#     })
#     chart_loss.add_series({
#         'name':       'Val Box Loss',
#         'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
#         'values':     ['Training_Validation_Log', 1, 4, num_epochs, 4], # cột val/box_loss
#     })
#     chart_loss.set_title({'name': 'Quá trình Training & Validation (Loss)'})
#     chart_loss.set_x_axis({'name': 'Epoch'})
#     chart_loss.set_y_axis({'name': 'Loss'})

#     # --- BIỂU ĐỒ 2: ĐỘ CHÍNH XÁC (mAP50) ---
#     # Thể hiện hiệu năng mô hình tăng dần
#     chart_map = workbook.add_chart({'type': 'line'})
#     chart_map.add_series({
#         'name':       'mAP50 (Độ chính xác)',
#         'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
#         'values':     ['Training_Validation_Log', 1, 6, num_epochs, 6], # cột metrics/mAP50(B)
#         'line':       {'color': 'green'},
#     })
#     chart_map.set_title({'name': 'Biểu đồ Độ chính xác qua các Epoch'})
#     chart_map.set_x_axis({'name': 'Epoch'})
#     chart_map.set_y_axis({'name': 'mAP50'})

#     # Chèn biểu đồ vào Excel
#     log_sheet.insert_chart('P2', chart_loss)
#     log_sheet.insert_chart('P18', chart_map)

#     writer.close()

#     print(f"\n[THÀNH CÔNG RỰC RỠ]")
#     print(f"1. File Excel đã có biểu đồ: {output_excel}")
#     print(f"2. Check Sheet 'Final_Testing_Result' để xem kết quả Testing.")
#     print(f"3. Thư mục '{train_dir}' chứa các 'biểu đồ liên quan' khác (Confusion Matrix, PR Curve, v.v.)")

def export_to_excel_with_charts(df, output_excel):
    print("[INFO] Đang vẽ lại biểu đồ cho 50 Epoch...")
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Training_Validation_Log', index=False)
    
    workbook  = writer.book
    log_sheet = writer.sheets['Training_Validation_Log']
    num_epochs = len(df)

    # # Biểu đồ Loss
    # chart_loss = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    # chart_loss.add_series({
    #     'name': 'Train Box Loss',
    #     'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
    #     'values':     ['Training_Validation_Log', 1, 1, num_epochs, 1],
    # })
    # chart_loss.add_series({
    #     'name': 'Val Box Loss',
    #     'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
    #     'values':     ['Training_Validation_Log', 1, 4, num_epochs, 4],
    # })
    # chart_loss.set_title({'name': 'Quá trình 50 Epoch (Loss)'})
    # log_sheet.insert_chart('P2', chart_loss)
    # 1. Biểu đồ Loss (Sửa lại index cột từ 1 thành 2, và từ 4 thành 5)
    chart_loss = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_loss.add_series({
        'name':       'Train Box Loss',
        'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0], # Trục X: Cột A (Epoch)
        'values':     ['Training_Validation_Log', 1, 2, num_epochs, 2], # Trục Y: Cột C (train/box_loss)
    })
    chart_loss.add_series({
        'name':       'Val Box Loss',
        'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0], # Trục X: Cột A
        'values':     ['Training_Validation_Log', 1, 5, num_epochs, 5], # Trục Y: Cột F (val/box_loss)
    })
    chart_loss.set_title({'name': 'Quá trình Training & Validation (Loss)'})
    chart_loss.set_x_axis({'name': 'Epoch', 'min': 0, 'max': 50})
    chart_loss.set_y_axis({'name': 'Loss'}) # Để ý trục Y, Loss phải nhỏ (thường < 2.0)
    log_sheet.insert_chart('P2', chart_loss)

    # Biểu đồ mAP50
    chart_map = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart_map.add_series({
        'name': 'mAP50',
        'categories': ['Training_Validation_Log', 1, 0, num_epochs, 0],
        'values':     ['Training_Validation_Log', 1, 6, num_epochs, 6],
    })
    log_sheet.insert_chart('P18', chart_map)
    
    writer.close()
    print(f"[XONG] File Excel 50 Epoch đã sẵn sàng: {output_excel}")

def train_and_report():
    # 1. Nạp file last.pt (đã học 30 epoch) làm nền tảng
    # KHÔNG dùng resume=True vì nó gây lỗi khi đã hoàn thành mục tiêu cũ
    model = YOLO("runs/detect/train/weights/last.pt") 

    # 2. Bắt đầu Training THÊM 20 Epoch nữa (để tổng là 50)
    print("[INFO] Đang huấn luyện thêm 20 Epoch dựa trên kết quả cũ...")
    results = model.train(
        data="action_data/data.yaml", 
        epochs=20,     # Học thêm 20 vòng
        imgsz=640, 
        device='cpu' 
    )

    # 3. ĐƯỜNG DẪN: Lấy thư mục mới (thường là train3 hoặc train4)
    new_train_dir = results.save_dir
    old_csv_path = "runs/detect/train/results.csv" # File 30 epoch đầu
    new_csv_path = os.path.join(new_train_dir, "results.csv") # File 20 epoch sau
    output_excel = "yolo_action_training_report_50_epochs.xlsx"

    # 4. LOGIC NỐI DỮ LIỆU (Để biểu đồ Excel có đủ 50 dòng)
    if os.path.exists(old_csv_path) and os.path.exists(new_csv_path):
        df_old = pd.read_csv(old_csv_path)
        df_new = pd.read_csv(new_csv_path)
        
        # Xóa khoảng trắng tên cột
        df_old.columns = [c.strip() for c in df_old.columns]
        df_new.columns = [c.strip() for c in df_new.columns]
        
        # Chỉnh lại số Epoch cho phần mới (từ 31 đến 50)
        df_new['epoch'] = df_new['epoch'] + 30
        
        # Nối 2 bảng lại thành 1
        df_final = pd.concat([df_old, df_new], ignore_index=True)
        
        # 5. XUẤT EXCEL CÓ ĐỦ 50 DÒNG VÀ BIỂU ĐỒ
        export_to_excel_with_charts(df_final, output_excel)
    else:
        print("[LỖI] Không tìm thấy đủ file CSV để nối dữ liệu.")

if __name__ == "__main__":
    train_and_report()