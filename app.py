# app.py
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from flask_migrate import Migrate
from database import db, Class, Student, Subject, Session, SpeechLog, Grade
import os
import threading
from dotenv import load_dotenv
from sqlalchemy import func, desc
from datetime import datetime
import pandas as pd
import joblib
from sqlalchemy import text
import json

# =============================================================
# === 1. KHỞI TẠO ỨNG DỤNG VÀ CẤU HÌNH ===
# =============================================================
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_default_secret_key')

# Cấu hình Database
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# =============================================================
# === 2. KHỞI TẠO CÁC EXTENSIONS (THƯ VIỆN) ===
# =============================================================
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")
migrate = Migrate(app, db)


# =============================================================
# === 3. KHAI BÁO BIẾN TOÀN CỤC (NẾU CẦN) ===
# =============================================================
output_frame = None
lock = threading.Lock()


# =============================================================
# === 4. CÁC ROUTE ĐỂ RENDER TRANG GIAO DIỆN (VIEWS) ===
# =============================================================
@app.route('/')
def dashboard(): return render_template('dashboard.html')

@app.route('/subjects')
def subjects(): return render_template('subjects.html')

@app.route('/students')
def students(): return render_template('students.html')

@app.route('/live-class')
def liveclass(): return render_template('liveclass.html') 

@app.route('/stats')
def stats(): return render_template('stats.html') 

# =============================================================
# === 5. CÁC API ENDPOINTS ===
# =============================================================

# --- VIDEO STREAM ---
@app.route('/api/video_stream/push', methods=['POST'])
def video_stream_push():
    global output_frame
    with lock:
        output_frame = request.data
    return jsonify({'status': 'ok'})

def generate_video_stream():
    while True:
        with lock:
            if output_frame is None:
                continue
            frame_bytes = output_frame[:]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- CRUD MÔN HỌC ---
@app.route('/api/subjects', methods=['GET', 'POST'])
def subjects_api():
    class_id = 1
    if request.method == 'POST':
        data = request.get_json()
        new_subject = Subject(name=data['name'], icon=data.get('icon', '📚'), category=data.get('category'), class_id=class_id)
        db.session.add(new_subject)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã thêm môn học '{data['name']}'."})
    
    subjects_query = Subject.query.filter_by(class_id=class_id).order_by(Subject.name).all()
    result = []
    for s in subjects_query:
        session_count = Session.query.filter_by(subject_id=s.id).count()
        total_speeches = db.session.query(func.count(SpeechLog.id)).join(Session).filter(Session.subject_id == s.id).scalar()
        result.append({'id': s.id, 'name': s.name, 'icon': s.icon, 'category': s.category, 'session_count': session_count, 'total_speeches': total_speeches})
    return jsonify(result)

@app.route('/api/subjects/<int:subject_id>', methods=['GET', 'PUT', 'DELETE'])
def single_subject_api(subject_id):
    subject = Subject.query.get_or_404(subject_id)
    if request.method == 'PUT':
        data = request.get_json()
        subject.name = data['name']
        subject.icon = data.get('icon', subject.icon)
        subject.category = data.get('category', subject.category)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã cập nhật môn học '{subject.name}'."})
    elif request.method == 'DELETE':
        Grade.query.filter_by(subject_id=subject.id).delete()
        Session.query.filter_by(subject_id=subject.id).delete() # Cần xóa session trước
        db.session.delete(subject)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã xóa môn học '{subject.name}'."})
    return jsonify({'id': subject.id, 'name': subject.name, 'icon': subject.icon, 'category': subject.category})

# --- CRUD HỌC SINH ---
@app.route('/api/students', methods=['GET', 'POST'])
def students_api():
    class_id = 1
    if request.method == 'POST':
        data = request.get_json()
        if Student.query.filter_by(student_code=data['code']).first():
            return jsonify({'status': 'error', 'message': f"Face ID '{data['code']}' đã tồn tại."}), 409
        
        dob = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date() if data.get('date_of_birth') else None
        new_student = Student(full_name=data['name'], student_code=data['code'], class_id=class_id, date_of_birth=dob, gender=data.get('gender'))
        db.session.add(new_student)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã thêm học sinh '{data['name']}'."})
    
    students_query = Student.query.filter_by(class_id=class_id).order_by(Student.full_name).all()
    result = [{'id': s.id, 'full_name': s.full_name, 'student_code': s.student_code, 'total_speeches': SpeechLog.query.filter_by(student_id=s.id).count(), 'date_of_birth': s.date_of_birth.isoformat() if s.date_of_birth else None, 'gender': s.gender} for s in students_query]
    return jsonify(result)

@app.route('/api/students/<int:student_id>', methods=['GET', 'PUT', 'DELETE'])
def single_student_api(student_id):
    student = Student.query.get_or_404(student_id)
    if request.method == 'PUT':
        data = request.get_json()
        student.full_name = data['name']
        if data.get('date_of_birth'):
            student.date_of_birth = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date()
        if data.get('gender'):
            student.gender = data['gender']
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã cập nhật học sinh '{student.full_name}'."})
    elif request.method == 'DELETE':
        SpeechLog.query.filter_by(student_id=student.id).delete()
        Grade.query.filter_by(student_id=student.id).delete()
        db.session.delete(student)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã xóa học sinh '{student.full_name}'."})
    return jsonify({'id': student.id, 'full_name': student.full_name, 'student_code': student.student_code, 'date_of_birth': student.date_of_birth.isoformat() if student.date_of_birth else None, 'gender': student.gender})

# --- CRUD ĐIỂM SỐ (GRADES) ---
@app.route('/api/students/<int:student_id>/grades', methods=['GET'])
def get_grades_for_student(student_id):
    try:
        if not Student.query.get(student_id):
            return jsonify({'status': 'error', 'message': f'Không tìm thấy học sinh ID {student_id}.'}), 404
        grades_query = Grade.query.filter_by(student_id=student_id).join(Subject).order_by(Grade.exam_date.desc()).all()
        result = [{'id': g.id, 'score': g.score, 'grade_type': g.grade_type, 'term': g.term, 'exam_date': g.exam_date.isoformat() if g.exam_date else None, 'subject_id': g.subject_id, 'subject_name': g.subject.name} for g in grades_query]
        return jsonify(result)
    except Exception as e:
        print(f"!!! LỖI trong get_grades_for_student: {e}")
        return jsonify({'status': 'error', 'message': 'Lỗi server khi tải điểm.'}), 500

@app.route('/api/grades', methods=['POST'])
def add_grade():
    data = request.get_json()
    if not all(k in data for k in ['student_id', 'subject_id', 'score']):
        return jsonify({'status': 'error', 'message': 'Thiếu thông tin.'}), 400
    try:
        exam_date = datetime.strptime(data['exam_date'], '%Y-%m-%d').date() if data.get('exam_date') else datetime.utcnow().date()
        new_grade = Grade(student_id=data['student_id'], subject_id=data['subject_id'], score=float(data['score']), grade_type=data.get('grade_type'), term=data.get('term'), exam_date=exam_date)
        db.session.add(new_grade)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Đã thêm điểm.'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Lỗi server: {e}'}), 500

@app.route('/api/grades/<int:grade_id>', methods=['PUT', 'DELETE'])
def update_or_delete_grade(grade_id):
    grade = Grade.query.get_or_404(grade_id)
    if request.method == 'PUT':
        data = request.get_json()
        try:
            grade.subject_id = data.get('subject_id', grade.subject_id)
            grade.score = float(data.get('score', grade.score))
            grade.grade_type = data.get('grade_type', grade.grade_type)
            # ... (các trường khác)
            db.session.commit()
            return jsonify({'status': 'success', 'message': 'Cập nhật điểm thành công.'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': f'Lỗi server: {e}'}), 500
    elif request.method == 'DELETE':
        db.session.delete(grade)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Đã xóa điểm.'})

# --- CÁC API KHÁC ---
# @app.route('/api/untrained_faces')
# def untrained_faces_api():
#     try:
#         trained_folders = [d for d in os.listdir("datasets/faces") if os.path.isdir(os.path.join("datasets/faces", d))]
#         used_codes = [s.student_code for s in Student.query.all()]
#         return jsonify([code for code in trained_folders if code not in used_codes])
#     except FileNotFoundError:
#         return jsonify([])


# --- API CHO LIVE CLASS (ĐÃ ĐƯỢC BAO GỒM LẠI ĐẦY ĐỦ) ---
@app.route('/api/sessions/start', methods=['POST'])
def start_session_api():
    data = request.get_json()
    subject_id = data.get('subject_id')
    if not subject_id: return jsonify({"status": "error", "message": "Thiếu subject_id"}), 400
    
    ongoing_session = Session.query.filter_by(status='ongoing').first()
    if ongoing_session: return jsonify({"status": "error", "message": "Một buổi học khác đang diễn ra."}), 409

    new_session = Session(subject_id=subject_id, status='ongoing')
    db.session.add(new_session)
    db.session.commit()
    subject = Subject.query.get(subject_id)
    return jsonify({"status": "success", "session_id": new_session.id, "subject_name": subject.name, "start_time": new_session.start_time.isoformat()})

@app.route('/api/sessions/end', methods=['POST'])
def end_session_api():
    data = request.get_json()
    session_id = data.get('session_id')
    if not session_id: return jsonify({"status": "error", "message": "Thiếu session_id"}), 400
    
    session = Session.query.get(session_id)
    if not session or session.status != 'ongoing': return jsonify({"status": "error", "message": "Không tìm thấy buổi học đang diễn ra"}), 404
    
    session.status = 'ended'
    session.end_time = datetime.utcnow()
    db.session.commit()
    return jsonify({"status": "success", "message": "Buổi học đã kết thúc."})

@app.route('/api/sessions/current', methods=['GET'])
def get_current_session_api():
    session = Session.query.filter_by(status='ongoing').first()
    if session:
        speech_counts_query = db.session.query(Student.student_code, func.count(SpeechLog.id)).join(SpeechLog).filter(SpeechLog.session_id == session.id).group_by(Student.student_code).all()
        return jsonify({"status": "found", "session_id": session.id, "subject_name": session.subject.name, "start_time": session.start_time.isoformat(), "speech_counts": dict(speech_counts_query)})
    else:
        return jsonify({"status": "not_found"})

# --- API NHẬN DIỆN TỪ AI PROCESSOR ---
# Sửa hàm recognize_api
@app.route('/api/recognize', methods=['POST'])
def recognize_api():
    data = request.get_json()
    student_code = data.get('student_code')
    current_session = Session.query.filter_by(status='ongoing').first()
    if not current_session:
        return jsonify({"status": "error", "message": "Không có buổi học nào đang diễn ra."}), 400
    
    student = Student.query.filter_by(student_code=student_code).first()
    if not student:
        return jsonify({"status": "error", "message": f"Không tìm thấy học sinh {student_code}"}), 404

    # THAY ĐỔI LỚN: KHÔNG GHI VÀO DB NỮA
    # Thay vào đó, phát sự kiện "chờ xác nhận"
    print(f"[PENDING] Gửi yêu cầu xác nhận cho: {student.full_name}")
    socketio.emit('pending_recognition', {
        'student_code': student.student_code,
        'full_name': student.full_name,
        'student_id': student.id # Gửi cả student_id để xử lý dễ hơn
    }, namespace='/live')
    
    return jsonify({"status": "pending"}) # Phản hồi lại cho AI processor rằng đã nhận

# =============================================================
# === 6. SOCKET.IO EVENTS ===
# =============================================================
@socketio.on('connect', namespace='/live')
def handle_live_connect(): print('Client đã kết nối tới Live Class.')

@socketio.on('disconnect', namespace='/live')
def handle_live_disconnect(): print('Client đã ngắt kết nối khỏi Live Class.')

@socketio.on('confirm_recognition', namespace='/live')
def handle_confirm_recognition(data):
    if data.get('action') == 'accept':
        current_session = Session.query.filter_by(status='ongoing').first()
        student = Student.query.get(data.get('student_id'))
        if current_session and student:
            new_log = SpeechLog(student_id=student.id, session_id=current_session.id)
            db.session.add(new_log)
            db.session.commit()
            print(f"[CONFIRMED] Ghi nhận: {student.full_name}")
            socketio.emit('speech_update', {'student_code': student.student_code, 'full_name': student.full_name, 'timestamp': datetime.utcnow().strftime('%H:%M:%S')}, namespace='/live')

# --- CÁC API KHÁC ---
@app.route('/api/dashboard_stats')
def dashboard_stats_api():
    try:
        class_id = 1
        
        subject_count = Subject.query.filter_by(class_id=class_id).count()
        student_count = Student.query.filter_by(class_id=class_id).count()
        session_count = Session.query.join(Subject).filter(Subject.class_id == class_id).count()
        speech_count = SpeechLog.query.join(Session).join(Subject).filter(Subject.class_id == class_id).count()

        recent_sessions = Session.query.filter(Session.status == 'ended').order_by(Session.end_time.desc()).limit(2).all()
        recent_activity = []
        for s in recent_sessions:
            speech_in_session = SpeechLog.query.filter_by(session_id=s.id).count()
            recent_activity.append({
                'subject_name': s.subject.name,
                'session_number': s.id,
                'end_time': s.end_time.strftime('%d/%m/%Y - %H:%M') if s.end_time else 'N/A',
                'speech_count': speech_in_session
            })

        return jsonify({
            'stats': {
                'subjects': subject_count,
                'students': student_count,
                'sessions': session_count,
                'speeches': speech_count
            },
            'recent_activity': recent_activity
        })
    except Exception as e:
        print(f"Lỗi trong /api/dashboard_stats: {e}")
        return jsonify({"message": "Lỗi server khi lấy dữ liệu dashboard"}), 500


@app.route('/api/statistics')
def statistics_api():
    # ... code cho API này đã được sửa lỗi ...
    try:
        class_id = 1
        top_students_query = db.session.query(Student.full_name, func.count(SpeechLog.id).label('total_speeches')).join(SpeechLog, Student.id == SpeechLog.student_id).filter(Student.class_id == class_id).group_by(Student.id).order_by(desc('total_speeches')).limit(5).all()
        top_students = [{'name': name, 'speeches': speeches} for name, speeches in top_students_query]
        subjects = Subject.query.filter_by(class_id=class_id).all()
        subject_analysis = []
        for s in subjects:
            session_count = Session.query.filter_by(subject_id=s.id).count()
            top_student_in_subject = db.session.query(Student.full_name, func.count(SpeechLog.id).label('speeches')).join(SpeechLog).join(Session).filter(Session.subject_id == s.id).group_by(Student.id).order_by(desc('speeches')).first()
            subject_analysis.append({'name': s.name, 'icon': s.icon, 'session_count': session_count, 'top_student_name': top_student_in_subject[0] if top_student_in_subject else "Chưa có", 'top_student_speeches': top_student_in_subject[1] if top_student_in_subject else 0})
        student_trends_data = {}
        all_students = Student.query.filter_by(class_id=class_id).all()
        for student in all_students:
            student_trends_data[student.full_name] = {}
            for subject in subjects:
                count = db.session.query(func.count(SpeechLog.id)).join(Session).filter(SpeechLog.student_id == student.id, Session.subject_id == subject.id).scalar()
                student_trends_data[student.full_name][subject.name] = count
        return jsonify({'top_students': top_students, 'subject_analysis': subject_analysis, 'student_trends_raw_data': student_trends_data})
    except Exception as e:
        print(f"Lỗi nghiêm trọng trong /api/statistics: {e}")
        return jsonify({"message": f"Lỗi server nội bộ: {e}"}), 500

@app.route('/api/subjects/<int:subject_id>/sessions')
def session_history_api(subject_id):
    # ... code cho API này ...
    return jsonify([])

@app.route('/api/untrained_faces')
def untrained_faces_api():
    DATASET_PATH = "datasets/faces"
    try:
        trained_folders = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        used_codes = [s.student_code for s in Student.query.all()]
        available_codes = [code for code in trained_folders if code not in used_codes]
        return jsonify(available_codes)
    except FileNotFoundError:
        return jsonify([])

# --- API PHÂN TÍCH HỌC SINH BẰNG AI ---
@app.route('/api/students/<int:student_id>/analysis')
def analyze_student_api(student_id):
    try:
        # 1. Tải model, scaler và bản đồ cụm
        model = joblib.load('student_cluster_model.pkl')
        scaler = joblib.load('student_data_scaler.pkl')
        with open('cluster_map.json', 'r', encoding='utf-8') as f:
            cluster_map = json.load(f)
            
        # 2. Lấy dữ liệu của học sinh
        query = text(f"""
            SELECT
                s.id AS student_id, s.full_name,
                COUNT(DISTINCT CASE WHEN sub.category = 'Khoa học Tự nhiên' THEN sl.id END) AS speeches_natural_science,
                COUNT(DISTINCT CASE WHEN sub.category = 'Khoa học Xã hội' THEN sl.id END) AS speeches_social_science,
                COUNT(DISTINCT CASE WHEN sub.category = 'Ngoại ngữ' THEN sl.id END) AS speeches_language,
                COUNT(DISTINCT CASE WHEN sub.category = 'Năng khiếu' THEN sl.id END) AS speeches_aptitude,
                AVG(CASE WHEN sub.category = 'Khoa học Tự nhiên' THEN g.score END) AS avg_grade_natural_science,
                AVG(CASE WHEN sub.category = 'Khoa học Xã hội' THEN g.score END) AS avg_grade_social_science,
                AVG(CASE WHEN sub.category = 'Ngoại ngữ' THEN g.score END) AS avg_grade_language,
                AVG(CASE WHEN sub.category = 'Năng khiếu' THEN g.score END) AS avg_grade_aptitude
            FROM students s
            LEFT JOIN speech_logs sl ON s.id = sl.student_id
            LEFT JOIN sessions sess ON sl.session_id = sess.id
            LEFT JOIN grades g ON s.id = g.student_id
            LEFT JOIN subjects sub ON sub.id = sess.subject_id OR sub.id = g.subject_id
            WHERE s.id = {student_id}
            GROUP BY s.id, s.full_name;
        """)
        
        df = pd.read_sql(query, db.engine)
        if df.empty:
            return jsonify({'tendency': 'Chưa đủ dữ liệu', 'reason': 'Không tìm thấy thông tin học sinh.'})
        
        # 3. Chuẩn bị dữ liệu
        df_cleaned = df.fillna(0)
        features = df_cleaned.drop(columns=['student_id', 'full_name'])
        features_scaled = scaler.transform(features)
        
        # 4. Đưa ra dự đoán
        prediction = model.predict(features_scaled)
        cluster_id = prediction[0]
        
        # 5. Tra cứu kết luận từ bản đồ
        tendency = cluster_map.get(str(cluster_id), "Chưa xác định")
        
        # 6. TẠO CHUỖI LÝ DO ĐẦY ĐỦ CẢ 4 NHÓM MÔN
        reason_data = df_cleaned.to_dict('records')[0]
        reason = (f"Điểm TB KHTN: {reason_data['avg_grade_natural_science']:.2f}, Phát biểu KHTN: {int(reason_data['speeches_natural_science'])}. "
                  f"Điểm TB KHXH: {reason_data['avg_grade_social_science']:.2f}, Phát biểu KHXH: {int(reason_data['speeches_social_science'])}. "
                  f"Điểm TB NN: {reason_data['avg_grade_language']:.2f}, Phát biểu NN: {int(reason_data['speeches_language'])}. "
                  f"Điểm TB NK: {reason_data['avg_grade_aptitude']:.2f}, Phát biểu NK: {int(reason_data['speeches_aptitude'])}.")

        return jsonify({'tendency': tendency, 'reason': reason})

    except FileNotFoundError:
        return jsonify({'tendency': 'Lỗi', 'reason': 'Mô hình AI hoặc bản đồ cụm chưa được tạo. Vui lòng chạy file ai_trainer.py.'}), 500
    except Exception as e:
        print(f"Lỗi trong API phân tích: {e}")
        return jsonify({'tendency': 'Lỗi', 'reason': str(e)}), 500

# =============================================================
# === SOCKET.IO EVENTS CHO LIVE CLASS ===
# =============================================================
@socketio.on('connect', namespace='/live')
def handle_live_connect():
    print('Client đã kết nối tới Live Class.')

@socketio.on('disconnect', namespace='/live')
def handle_live_disconnect():
    print('Client đã ngắt kết nối khỏi Live Class.')

# =============================================================
# === 7. KHỐI CHẠY ỨNG DỤNG ===
# =============================================================
if __name__ == '__main__':
    with app.app_context():
        # Lệnh này dùng để tạo lớp học mặc định nếu chưa có
        if not db.session.get(Class, 1):
            print("[INFO] Tạo lớp học mặc định (ID=1)...")
            db.session.add(Class(id=1, name="9/1", academic_year="2025-2026"))
            db.session.commit()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)