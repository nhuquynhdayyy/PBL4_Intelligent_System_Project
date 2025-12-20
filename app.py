# File: app.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_migrate import Migrate
from database import db, Class, Student, Subject, Session, SpeechLog, Grade
import os
from dotenv import load_dotenv
from sqlalchemy import func, desc, text
from datetime import datetime
import pandas as pd
import joblib
import json

# =============================================================
# === 1. CẤU HÌNH HỆ THỐNG ===
# =============================================================
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret')

# Cấu hình Database
db_user = os.getenv('DB_USER', 'root')
db_password = os.getenv('DB_PASSWORD', '') # Điền mật khẩu MySQL của bạn vào đây nếu có
db_host = os.getenv('DB_HOST', 'localhost')
db_name = os.getenv('DB_NAME', 'pbl4')

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")
migrate = Migrate(app, db)

# =============================================================
# === 2. ROUTES GIAO DIỆN (VIEW) ===
# =============================================================
@app.route('/')
def dashboard(): 
    return render_template('dashboard.html')

@app.route('/subjects')
def subjects(): 
    return render_template('subjects.html')

@app.route('/students')
def students(): 
    return render_template('students.html')

@app.route('/stats')
def stats(): 
    return render_template('stats.html') 

@app.route('/live-class')
def liveclass(): 
    return render_template('liveclass.html')

# =============================================================
# === 3. API ENDPOINTS (LOGIC) ===
# =============================================================

# --- API NHẬN DIỆN TỪ AI SERVER ---
@app.route('/api/recognize', methods=['POST'])
def recognize_api():
    data = request.get_json()
    student_code = data.get('student_code')
    
    current_session = Session.query.filter_by(status='ongoing').first()
    if not current_session:
        return jsonify({"status": "error", "message": "Chưa bắt đầu buổi học"}), 400
    
    student = Student.query.filter_by(student_code=student_code).first()
    if not student:
        return jsonify({"status": "error", "message": "Không tìm thấy HS"}), 404

    socketio.emit('pending_recognition', {
        'student_code': student.student_code,
        'full_name': student.full_name,
        'student_id': student.id
    }, namespace='/live')
    
    return jsonify({"status": "pending"})

# --- QUẢN LÝ MÔN HỌC ---
@app.route('/api/subjects', methods=['GET', 'POST'])
def subjects_api():
    if request.method == 'POST':
        data = request.get_json()
        new_subject = Subject(name=data['name'], icon=data.get('icon', '📚'), category=data.get('category'), class_id=1)
        db.session.add(new_subject)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã thêm môn học '{data['name']}'."})
    
    subjects = Subject.query.filter_by(class_id=1).all()
    result = []
    for s in subjects:
        session_count = Session.query.filter_by(subject_id=s.id).count()
        total_speeches = SpeechLog.query.join(Session).filter(Session.subject_id == s.id).count()
        result.append({'id': s.id, 'name': s.name, 'icon': s.icon, 'category': s.category, 'session_count': session_count, 'total_speeches': total_speeches})
    return jsonify(result)
@app.route('/api/subjects/<int:subject_id>', methods=['GET', 'PUT', 'DELETE'])
def single_subject_api(subject_id):
    subject = db.session.get(Subject, subject_id)
    if not subject: 
        return jsonify({'status': 'error', 'message': 'Không tìm thấy môn học'}), 404

    if request.method == 'PUT':
        data = request.get_json()
        subject.name = data['name']
        subject.icon = data.get('icon', subject.icon)
        subject.category = data.get('category', subject.category)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Cập nhật thành công'})
    
    elif request.method == 'DELETE':
        try:
            # 1. Xóa tất cả điểm số liên quan đến môn học này
            Grade.query.filter_by(subject_id=subject.id).delete()

            # 2. Tìm tất cả các buổi học (Session) của môn học này
            sessions = Session.query.filter_by(subject_id=subject.id).all()
            session_ids = [s.id for s in sessions]

            if session_ids:
                # 3. Xóa tất cả phát biểu (SpeechLog) thuộc về các buổi học này
                SpeechLog.query.filter(SpeechLog.session_id.in_(session_ids)).delete(synchronize_session=False)

            # 4. Xóa tất cả các buổi học (Session) của môn học này
            Session.query.filter_by(subject_id=subject.id).delete()

            # 5. Cuối cùng mới xóa chính môn học đó
            db.session.delete(subject)
            
            db.session.commit()
            return jsonify({'status': 'success', 'message': f"Đã xóa môn học '{subject.name}' và toàn bộ dữ liệu liên quan."})
        
        except Exception as e:
            db.session.rollback()
            print(f"Lỗi khi xóa môn học: {str(e)}")
            return jsonify({'status': 'error', 'message': f'Không thể xóa môn học do có lỗi ràng buộc dữ liệu.'}), 500
    
    return jsonify({'id': subject.id, 'name': subject.name, 'icon': subject.icon, 'category': subject.category})
# --- QUẢN LÝ HỌC SINH ---
@app.route('/api/students', methods=['GET', 'POST'])
def students_api():
    if request.method == 'POST':
        data = request.get_json()
        dob = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date() if data.get('date_of_birth') else None
        new_student = Student(full_name=data['name'], student_code=data['code'], class_id=1, date_of_birth=dob, gender=data.get('gender'))
        db.session.add(new_student)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Thêm học sinh thành công'})
    
    students = Student.query.filter_by(class_id=1).all()
    return jsonify([{'id': s.id, 'full_name': s.full_name, 'student_code': s.student_code} for s in students])

@app.route('/api/students/<int:student_id>', methods=['GET', 'PUT', 'DELETE'])
def single_student_api(student_id):
    student = db.session.get(Student, student_id)
    if not student: return jsonify({'status': 'error', 'message': 'Không tìm thấy học sinh'}), 404

    if request.method == 'PUT':
        data = request.get_json()
        student.full_name = data.get('name', student.full_name)
        if data.get('date_of_birth'):
            student.date_of_birth = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date()
        student.gender = data.get('gender', student.gender)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Cập nhật thành công'})
    
    elif request.method == 'DELETE':
        SpeechLog.query.filter_by(student_id=student.id).delete()
        Grade.query.filter_by(student_id=student.id).delete()
        db.session.delete(student)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Đã xóa học sinh'})
    
    return jsonify({
        'id': student.id, 
        'full_name': student.full_name, 
        'student_code': student.student_code, 
        'date_of_birth': student.date_of_birth.isoformat() if student.date_of_birth else None, 
        'gender': student.gender
    })

# --- QUẢN LÝ ĐIỂM SỐ ---
@app.route('/api/students/<int:student_id>/grades', methods=['GET'])
def get_grades(student_id):
    grades = Grade.query.filter_by(student_id=student_id).all()
    return jsonify([{
        'id': g.id, 
        'score': g.score, 
        'grade_type': g.grade_type, 
        'term': g.term, 
        'exam_date': g.exam_date.isoformat(), 
        'subject_id': g.subject_id, 
        'subject_name': g.subject.name
    } for g in grades])

@app.route('/api/grades', methods=['POST'])
def add_grade():
    data = request.get_json()
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
def update_delete_grade(grade_id):
    grade = db.session.get(Grade, grade_id)
    if not grade: return jsonify({'status': 'error', 'message': 'Không tìm thấy điểm'}), 404
    
    if request.method == 'DELETE':
        db.session.delete(grade)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Đã xóa điểm'})
    
    if request.method == 'PUT':
        data = request.get_json()
        grade.score = float(data['score'])
        grade.grade_type = data['grade_type']
        grade.term = data['term']
        grade.subject_id = data['subject_id']
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Đã cập nhật điểm'})

# --- SESSION (BUỔI HỌC) ---
@app.route('/api/sessions/start', methods=['POST'])
def start_session_api():
    data = request.get_json()
    subject_id = data.get('subject_id')
    ongoing = Session.query.filter_by(status='ongoing').first()
    if ongoing: return jsonify({"status": "error", "message": "Một buổi học khác đang diễn ra."}), 409

    new_session = Session(subject_id=subject_id, status='ongoing')
    db.session.add(new_session)
    db.session.commit()
    return jsonify({"status": "success", "session_id": new_session.id})

@app.route('/api/sessions/end', methods=['POST'])
def end_session_api():
    data = request.get_json()
    session = db.session.get(Session, data.get('session_id'))
    if not session: return jsonify({"status": "error", "message": "Không tìm thấy buổi học"}), 404
    
    session.status = 'ended'
    session.end_time = datetime.utcnow()
    session.speech_count = SpeechLog.query.filter_by(session_id=session.id).count()
    db.session.commit()
    return jsonify({"status": "success", "message": "Buổi học đã kết thúc."})

@app.route('/api/sessions/current', methods=['GET'])
def get_current_session_api():
    session = Session.query.filter_by(status='ongoing').first()
    if session:
        counts = db.session.query(Student.student_code, func.count(SpeechLog.id)).join(SpeechLog).filter(SpeechLog.session_id == session.id).group_by(Student.student_code).all()
        return jsonify({"status": "found", "session_id": session.id, "subject_name": session.subject.name, "speech_counts": dict(counts)})
    return jsonify({"status": "not_found"})

# --- LỊCH SỬ VÀ CHI TIẾT ---
@app.route('/api/subjects/<int:subject_id>/history')
def session_history(subject_id):
    sessions = Session.query.filter_by(subject_id=subject_id).order_by(Session.id.desc()).all()
    res = []
    for i, s in enumerate(sessions):
        count = SpeechLog.query.filter_by(session_id=s.id).count()
        res.append({
            "id": s.id, 
            "session_number": len(sessions) - i, 
            "end_time": s.end_time.strftime('%d/%m/%Y %H:%M') if s.end_time else "Đang dạy", 
            "speech_count": count
        })
    return jsonify(res)

@app.route('/api/sessions/<int:session_id>/details')
def session_details_api(session_id):
    details = db.session.query(Student.full_name, func.count(SpeechLog.id)).join(SpeechLog).filter(SpeechLog.session_id == session_id).group_by(Student.id).all()
    return jsonify([{"name": d[0], "count": d[1]} for d in details])

# --- THỐNG KÊ DASHBOARD ---
@app.route('/api/dashboard_stats')
def dashboard_stats_api():
    try:
        stats = {
            'subjects': Subject.query.count(),
            'students': Student.query.count(),
            'sessions': Session.query.count(),
            'speeches': SpeechLog.query.count()
        }
        recent = []
        sessions = Session.query.filter_by(status='ended').order_by(Session.end_time.desc()).limit(3).all()
        for s in sessions:
            recent.append({
                'subject_name': s.subject.name,
                'session_number': s.id,
                'end_time': s.end_time.strftime('%d/%m/%Y %H:%M'),
                'speech_count': SpeechLog.query.filter_by(session_id=s.id).count()
            })
        return jsonify({'stats': stats, 'recent_activity': recent})
    except: return jsonify({"message": "Lỗi server"}), 500

# --- PHÂN TÍCH AI ---
@app.route('/api/students/<int:student_id>/analysis')
def analyze_student_api(student_id):
    try:
        model = joblib.load('student_cluster_model.pkl')
        scaler = joblib.load('student_data_scaler.pkl')
        with open('cluster_map.json', 'r', encoding='utf-8') as f:
            cluster_map = json.load(f)
            
        query = text(f"""
            SELECT s.id, s.full_name,
                COUNT(DISTINCT CASE WHEN sub.category = 'Khoa học Tự nhiên' THEN sl.id END) AS speeches_natural_science,
                COUNT(DISTINCT CASE WHEN sub.category = 'Khoa học Xã hội' THEN sl.id END) AS speeches_social_science,
                COUNT(DISTINCT CASE WHEN sub.category = 'Ngoại ngữ' THEN sl.id END) AS speeches_language,
                COUNT(DISTINCT CASE WHEN sub.category = 'Năng khiếu' THEN sl.id END) AS speeches_aptitude,
                IFNULL(AVG(CASE WHEN sub.category = 'Khoa học Tự nhiên' THEN g.score END), 0) AS avg_grade_natural_science,
                IFNULL(AVG(CASE WHEN sub.category = 'Khoa học Xã hội' THEN g.score END), 0) AS avg_grade_social_science,
                IFNULL(AVG(CASE WHEN sub.category = 'Ngoại ngữ' THEN g.score END), 0) AS avg_grade_language,
                IFNULL(AVG(CASE WHEN sub.category = 'Năng khiếu' THEN g.score END), 0) AS avg_grade_aptitude
            FROM students s
            LEFT JOIN speech_logs sl ON s.id = sl.student_id
            LEFT JOIN sessions sess ON sl.session_id = sess.id
            LEFT JOIN grades g ON s.id = g.student_id
            LEFT JOIN subjects sub ON sub.id = sess.subject_id OR sub.id = g.subject_id
            WHERE s.id = {student_id}
            GROUP BY s.id;
        """)
        
        df = pd.read_sql(query, db.engine)
        if df.empty: return jsonify({'tendency': 'Chưa đủ dữ liệu', 'reason': 'Học sinh chưa có hoạt động.'})
        
        features = df.drop(columns=['id', 'full_name'])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'tendency': cluster_map.get(str(prediction), "Đang phân tích"),
            'reason': f"Dựa trên dữ liệu học tập của {df.iloc[0]['full_name']}."
        })
    except Exception as e:
        return jsonify({'tendency': 'Lỗi AI', 'reason': str(e)}), 500

@app.route('/api/untrained_faces')
def untrained_faces_api():
    DATASET_PATH = "datasets/faces"
    try:
        trained = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        used = [s.student_code for s in Student.query.all()]
        return jsonify([c for c in trained if c not in used])
    except: return jsonify([])

# =============================================================
# === 4. SOCKET.IO ===
# =============================================================
@socketio.on('confirm_recognition', namespace='/live')
def handle_confirm(data):
    if data.get('action') == 'accept':
        sess = Session.query.filter_by(status='ongoing').first()
        stu = db.session.get(Student, data.get('student_id'))
        if sess and stu:
            db.session.add(SpeechLog(student_id=stu.id, session_id=sess.id))
            db.session.commit()
            socketio.emit('speech_update', {'student_code': stu.student_code, 'full_name': stu.full_name}, namespace='/live')
            # --- API THỐNG KÊ TỔNG HỢP (Dùng cho trang stats.html) ---
@app.route('/api/statistics')
def statistics_api():
    try:
        class_id = 1
        # 1. Lấy Top 5 học sinh phát biểu nhiều nhất
        top_students_query = db.session.query(
            Student.full_name, 
            func.count(SpeechLog.id).label('total_speeches')
        ).join(SpeechLog).filter(Student.class_id == class_id)\
         .group_by(Student.id).order_by(desc('total_speeches')).limit(5).all()
        
        top_students = [{'name': s[0], 'speeches': s[1]} for s in top_students_query]

        # 2. Phân tích theo từng môn học
        subjects = Subject.query.filter_by(class_id=class_id).all()
        subject_analysis = []
        for s in subjects:
            # Đếm số buổi học của môn
            session_count = Session.query.filter_by(subject_id=s.id).count()
            
            # Tìm học sinh hăng hái nhất của môn đó
            top_in_sub = db.session.query(
                Student.full_name, 
                func.count(SpeechLog.id).label('speeches')
            ).join(SpeechLog).join(Session).filter(Session.subject_id == s.id)\
             .group_by(Student.id).order_by(desc('speeches')).first()
            
            subject_analysis.append({
                'name': s.name, 
                'icon': s.icon, 
                'session_count': session_count,
                'top_student_name': top_in_sub[0] if top_in_sub else "Chưa có",
                'top_student_speeches': top_in_sub[1] if top_in_sub else 0
            })

        return jsonify({
            'top_students': top_students, 
            'subject_analysis': subject_analysis
        })
    except Exception as e:
        print(f"Lỗi Statistics API: {str(e)}")
        return jsonify({"message": "Lỗi server khi lấy thống kê"}), 500

if __name__ == '__main__':
    with app.app_context():
        # Tự động tạo bảng nếu chưa có
        db.create_all()
        # Tạo lớp mặc định
        if not db.session.get(Class, 1):
            db.session.add(Class(id=1, name="9/1", academic_year="2025-2026"))
            db.session.commit()
            
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)