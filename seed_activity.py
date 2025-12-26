import random
from datetime import datetime, timedelta
from app import app, db, Student, Subject, Session, SpeechLog

def seed_activity_data():
    with app.app_context():
        print("--- Đang xóa dữ liệu buổi học và phát biểu cũ để làm sạch... ---")
        # Xóa dữ liệu cũ để tránh trùng lặp hoặc nhảy số hiệu buổi học
        db.session.query(SpeechLog).delete()
        db.session.query(Session).delete()
        db.session.commit()

        # 1. Lấy danh sách tất cả các môn học và học sinh đang có
        all_subjects = Subject.query.all()
        all_students = Student.query.all()

        if not all_subjects or not all_students:
            print("Lỗi: Không tìm thấy Môn học hoặc Học sinh. Vui lòng kiểm tra lại DB.")
            return

        print(f"Tìm thấy {len(all_subjects)} môn học và {len(all_students)} học sinh.")

        # 2. Tạo dữ liệu giả lập trong vòng 30 ngày qua
        now = datetime.now()
        
        # Duyệt qua từng môn học để tạo buổi học cho môn đó
        for sub in all_subjects:
            # Giả lập mỗi môn học đã diễn ra từ 5 đến 10 buổi
            num_sessions = random.randint(5, 10)
            
            # Lấy danh sách học sinh thuộc cùng lớp với môn học này
            class_students = [s for s in all_students if s.class_id == sub.class_id]
            
            if not class_students:
                continue # Bỏ qua nếu môn học ở lớp chưa có học sinh nào

            for i in range(num_sessions):
                # Giả lập thời gian kết thúc buổi học ngẫu nhiên trong 30 ngày qua
                random_days = random.randint(1, 30)
                random_hours = random.randint(1, 23)
                session_time = now - timedelta(days=random_days, hours=random_hours)

                # Tạo Buổi học (CỰC KỲ QUAN TRỌNG: Phải gán class_id)
                new_session = Session(
                    subject_id=sub.id,
                    class_id=sub.class_id, # Đảm bảo session thuộc về lớp của môn học
                    status='ended',
                    end_time=session_time
                )
                db.session.add(new_session)
                db.session.flush() # Để lấy được ID của session vừa tạo

                # 3. Tạo lượt phát biểu ngẫu nhiên cho buổi học này
                # Mỗi buổi học có từ 10 đến 30 lượt phát biểu chia cho học sinh trong lớp
                num_speeches = random.randint(10, 30)
                for _ in range(num_speeches):
                    student = random.choice(class_students)
                    log = SpeechLog(
                        student_id=student.id,
                        session_id=new_session.id,
                        timestamp=session_time # Thời điểm phát biểu lấy theo giờ buổi học
                    )
                    db.session.add(log)

        db.session.commit()
        print("--- THÀNH CÔNG: Đã tạo xong dữ liệu Buổi học và Phát biểu ảo! ---")

if __name__ == "__main__":
    seed_activity_data()