# =============================================================
# === 1. KHAI BÁO THƯ VIỆN (IMPORTS) ===
# =============================================================
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_migrate import Migrate
from database import db, Class, Student, Subject, Session, SpeechLog, Grade
import os
from database import Class 
from dotenv import load_dotenv
from sqlalchemy import func, desc, text
from datetime import datetime
import pandas as pd
import joblib
import json
import io
import csv
from flask import request, redirect, url_for # Nhớ import thêm
from database import User # Đảm bảo đã import User
# === THÊM HOẶC SỬA LẠI DÒNG NÀY Ở ĐẦU FILE app.py ===
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
# =============================================================
# === 2. CẤU HÌNH HỆ THỐNG ===
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
# === CẤU HÌNH FLASK-LOGIN ===
# =============================================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Tên HÀM của route đăng nhập
login_manager.login_message = "Vui lòng đăng nhập để truy cập trang này."
login_manager.login_message_category = "info"
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Bạn không có quyền truy cập trang này.', 'danger')
            return redirect(url_for('dashboard')) # Hoặc login
        return f(*args, **kwargs)
    return decorated_function
@login_manager.user_loader
def load_user(user_id):
    # Flask-Login sẽ dùng hàm này để lấy thông tin user từ ID lưu trong session
    return db.session.get(User, int(user_id))

# =============================================================
# === 3. ROUTES GIAO DIỆN (VIEW) ===
# =============================================================
### CÁC ROUTE MỚI CHO XÁC THỰC ###
# Sửa lại hàm register()
@app.route('/register')
def register():
    # Không cho phép đăng ký công khai nữa
    flash('Chức năng đăng ký công khai đã bị vô hiệu hóa. Vui lòng liên hệ Admin để được cấp tài khoản.', 'info')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash('Tên đăng nhập hoặc mật khẩu không đúng.', 'danger')
            return redirect(url_for('login'))
        
        login_user(user)

        # Chuyển hướng đến trang tiếp theo nếu có, nếu không thì về dashboard
        next_page = request.args.get('next')
        if user.role == 'admin':
            return redirect(next_page or url_for('admin_classes'))
        else:
            return redirect(next_page or url_for('dashboard'))

    return render_template('login.html')


@app.route('/logout')
@login_required 
def logout():
    logout_user()
    flash('Bạn đã đăng xuất.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required # <-- THÊM DÒNG NÀY
def dashboard(): 
    return render_template('dashboard.html')


@app.route('/subjects')
@login_required # <-- THÊM DÒNG NÀY
def subjects(): 
    return render_template('subjects.html')

@app.route('/students')
@login_required # <-- THÊM DÒNG NÀY
def students(): 
    return render_template('students.html')

@app.route('/stats')
@login_required # <-- THÊM DÒNG NÀY
def stats(): 
    return render_template('stats.html') 

@app.route('/live-class')
@login_required # <-- THÊM DÒNG NÀY
def liveclass(): 
    return render_template('liveclass.html')

# Thêm vào khu vực ROUTES GIAO DIỆN
@app.route('/admin/dashboard')
@admin_required # <-- THÊM DÒNG NÀY
def admin_dashboard():
    return render_template('admin/dashboard.html')
# Thêm methods=['GET', 'POST'] vào decorator
@app.route('/admin/classes', methods=['GET', 'POST'])
@admin_required # <-- THÊM DÒNG NÀY
def admin_classes():
    if request.method == 'POST':
        name = request.form.get('class_name')
        year = request.form.get('academic_year')
        
        if name and year:
            # Kiểm tra xem tên lớp đã tồn tại chưa
            existing_class = Class.query.filter_by(name=name).first()
            if existing_class:
                flash(f"Lỗi: Tên lớp '{name}' đã tồn tại.", 'danger')
            else:
                new_class = Class(name=name, academic_year=year)
                db.session.add(new_class)
                db.session.commit()
                flash(f"Đã thêm thành công lớp học '{name}'.", 'success')
            return redirect(url_for('admin_classes'))

    all_classes = Class.query.order_by(Class.name).all()
    return render_template('admin/classes.html', classes=all_classes)
@app.route('/admin/students')
@admin_required # <-- THÊM DÒNG NÀY
def admin_students():
    return render_template('admin/students.html')

@app.route('/admin/subjects')
@admin_required # <-- THÊM DÒNG NÀY
def admin_subjects():
    return render_template('admin/subjects.html')
# === THÊM ROUTE MỚI NÀY VÀO ===
# === THAY THẾ HÀM CŨ BẰNG HÀM NÀY ===
@app.route('/admin/users', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_users():
    # Xử lý khi Admin gửi form tạo tài khoản mới (POST request)
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Validation (Kiểm tra dữ liệu)
        if not username or not email or not password:
            flash('Vui lòng điền đầy đủ thông tin.', 'danger')
            return redirect(url_for('admin_users'))

        if User.query.filter_by(username=username).first():
            flash(f"Tên đăng nhập '{username}' đã tồn tại.", 'danger')
            return redirect(url_for('admin_users'))

        if User.query.filter_by(email=email).first():
            flash(f"Email '{email}' đã được sử dụng.", 'danger')
            return redirect(url_for('admin_users'))

        # Tạo tài khoản mới với vai trò 'teacher'
        try:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_teacher = User(
                username=username,
                email=email,
                password_hash=hashed_password,
                role='teacher' # Luôn tạo tài khoản với vai trò teacher
            )
            db.session.add(new_teacher)
            db.session.commit()
            flash(f"Đã tạo thành công tài khoản cho giáo viên '{username}'.", 'success')
        except Exception as e:
            db.session.rollback()
            flash(f"Lỗi khi tạo tài khoản: {e}", 'danger')
        
        return redirect(url_for('admin_users'))

    # Phần xử lý GET request (hiển thị danh sách)
    try:
        # Chỉ lấy những người dùng có vai trò là 'teacher'
        users = User.query.filter_by(role='teacher').order_by(User.username).all()
        # Lấy tất cả các lớp học để hiển thị trong dropdown
        classes = Class.query.order_by(Class.name).all()
        return render_template('admin/users.html', users=users, classes=classes)
    except Exception as e:
        flash(f"Có lỗi xảy ra khi tải trang: {e}", "danger")
# === 4. API ENDPOINTS (LOGIC) ===
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
@login_required
def subjects_api():
    user_class_id = current_user.class_id
    if not user_class_id:
        return jsonify({"message": "Bạn chưa được phân công lớp"}), 403

    if request.method == 'POST':
        data = request.get_json()
        new_subject = Subject(
            name=data['name'], 
            icon=data.get('icon', '📚'), 
            category=data.get('category'), 
            class_id=user_class_id # Thay 1 bằng user_class_id
        )
        db.session.add(new_subject)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã thêm môn học '{data['name']}'."})
    
    # Chỉ lấy môn học của lớp mình
    subjects = Subject.query.filter_by(class_id=user_class_id).all()
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
            Grade.query.filter_by(subject_id=subject.id).delete()
            sessions = Session.query.filter_by(subject_id=subject.id).all()
            session_ids = [s.id for s in sessions]
            if session_ids:
                SpeechLog.query.filter(SpeechLog.session_id.in_(session_ids)).delete(synchronize_session=False)
            Session.query.filter_by(subject_id=subject.id).delete()
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
@login_required
def students_api():
    user_class_id = current_user.class_id
    if not user_class_id:
        return jsonify([]), 403

    if request.method == 'POST':
        data = request.get_json()
        dob = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date() if data.get('date_of_birth') else None
        new_student = Student(
            full_name=data['name'], 
            student_code=data['code'], 
            class_id=user_class_id, # Thay 1 bằng user_class_id
            date_of_birth=dob, 
            gender=data.get('gender')
        )
        db.session.add(new_student)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Thêm học sinh thành công'})
    
    # Chỉ lấy học sinh của lớp mình
    students = Student.query.filter_by(class_id=user_class_id).order_by(Student.full_name).all()
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
# --- BUỔI HỌC (SESSION) ---
@app.route('/api/sessions/start', methods=['POST'])
@login_required
def start_session_api():
    user_class_id = current_user.class_id
    data = request.get_json()
    subject_id = data.get('subject_id')
    
    # Kiểm tra xem môn học đó có thuộc lớp của giáo viên này không
    subject = Subject.query.filter_by(id=subject_id, class_id=user_class_id).first()
    if not subject:
        return jsonify({"status": "error", "message": "Môn học không hợp lệ cho lớp này"}), 403

    ongoing = Session.query.filter_by(status='ongoing', class_id=user_class_id).first()
    if ongoing: 
        return jsonify({"status": "error", "message": "Một buổi học khác của lớp đang diễn ra."}), 409

    new_session = Session(subject_id=subject_id, status='ongoing', class_id=user_class_id) # Thay 1
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
@login_required
def dashboard_stats_api():
    user_class_id = current_user.class_id
    if not user_class_id:
        return jsonify({"message": "Chưa có dữ liệu lớp"}), 403
        
    try:
        stats = {
            'subjects': Subject.query.filter_by(class_id=user_class_id).count(),
            'students': Student.query.filter_by(class_id=user_class_id).count(),
            'sessions': Session.query.filter_by(class_id=user_class_id).count(),
            'speeches': SpeechLog.query.join(Session).filter(Session.class_id == user_class_id).count()
        }
        recent = []
        # Lấy lịch sử của lớp này
        sessions = Session.query.filter_by(status='ended', class_id=user_class_id).order_by(Session.end_time.desc()).limit(3).all()
        for s in sessions:
            recent.append({
                'subject_name': s.subject.name,
                'session_number': s.id,
                'end_time': s.end_time.strftime('%d/%m/%Y %H:%M'),
                'speech_count': SpeechLog.query.filter_by(session_id=s.id).count()
            })
        return jsonify({'stats': stats, 'recent_activity': recent})
    except Exception as e: 
        return jsonify({"message": f"Lỗi: {str(e)}"}), 500
    
@app.route('/api/untrained_faces')
def untrained_faces_api():
    DATASET_PATH = "datasets/faces"
    try:
        trained = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        used = [s.student_code for s in Student.query.all()]
        return jsonify([c for c in trained if c not in used])
    except: return jsonify([])
# --- THỐNG KÊ CHI TIẾT (STATS TAB) ---
@app.route('/api/statistics')
@login_required
def statistics_api():
    try:
        user_class_id = current_user.class_id

        # 1. Bảng xếp hạng học sinh trong lớp
        all_students_ranking_query = db.session.query(
            Student.id, Student.full_name, func.count(SpeechLog.id).label('total_speeches')
        ).outerjoin(SpeechLog).filter(Student.class_id == user_class_id)\
         .group_by(Student.id).order_by(desc('total_speeches')).all()
        
        all_students_ranking = [{'id': s[0], 'name': s[1], 'speeches': s[2]} for s in all_students_ranking_query]

        # 2. KPI tổng quan của lớp
        total_sessions = Session.query.filter_by(class_id=user_class_id).count()
        total_speeches = SpeechLog.query.join(Session).filter(Session.class_id == user_class_id).count()
        total_students = Student.query.filter_by(class_id=user_class_id).count()
        most_active_student = all_students_ranking[0]['name'] if all_students_ranking else "N/A"

        # 3. Phân tích môn học của lớp
        subjects = Subject.query.filter_by(class_id=user_class_id).all()
        subject_analysis = []
        for s in subjects:
            total_speeches_in_sub = SpeechLog.query.join(Session).filter(Session.subject_id == s.id).count()
            subject_analysis.append({
                'id': s.id, 'name': s.name, 'icon': s.icon, 
                'session_count': Session.query.filter_by(subject_id=s.id).count(),
                'total_speeches': total_speeches_in_sub,
                'top_student_name': "N/A" # Có thể tính thêm nếu muốn
            })
        
        return jsonify({
            'kpis': {
                "total_sessions": total_sessions, "total_speeches": total_speeches,
                "total_students": total_students, "most_active_student": most_active_student
            },
            'all_students_ranking': all_students_ranking,
            'subject_analysis': subject_analysis
        })
    except Exception as e:
        return jsonify({"message": str(e)}), 500
@app.route('/api/students/<int:student_id>/analysis')
@login_required
def analyze_student_api(student_id):
    try:
        # 0) Lấy class mà user đang quản lý
        user_class_id = getattr(current_user, "class_id", None)
        if not user_class_id:
            return jsonify({"message": "Tài khoản chưa được phân công lớp"}), 403

        # 1) Học sinh phải thuộc lớp của user
        student = Student.query.filter_by(id=student_id, class_id=user_class_id).first()
        if not student:
            return jsonify({"message": "Học sinh không thuộc lớp bạn quản lý"}), 403

        # 2) XẾP HẠNG TRONG LỚP (lọc theo user_class_id)
        full_ranking = (
            db.session.query(
                Student.id,
                func.count(SpeechLog.id).label("total_speeches")
            )
            .outerjoin(SpeechLog, SpeechLog.student_id == Student.id)
            .filter(Student.class_id == user_class_id)
            .group_by(Student.id)
            .order_by(desc("total_speeches"))
            .all()
        )

        student_rank = 0
        for i, s in enumerate(full_ranking):
            if s.id == student_id:
                student_rank = i + 1
                break

        # 3) KPI: tổng phát biểu (lọc student_id)
        total_speeches = SpeechLog.query.filter_by(student_id=student_id).count()

        # 4) Môn học thế mạnh (lọc theo student_id + class để tránh lẫn lớp)
        best_subject_query = (
            db.session.query(Subject.name)
            .join(Session, Session.subject_id == Subject.id)
            .join(SpeechLog, SpeechLog.session_id == Session.id)
            .filter(
                SpeechLog.student_id == student_id,
                Subject.class_id == user_class_id
            )
            .group_by(Subject.id)
            .order_by(func.count(SpeechLog.id).desc())
            .first()
        )
        best_subject = best_subject_query[0] if best_subject_query else "N/A"

        student_kpis = {
            "rank": student_rank,
            "total_speeches": total_speeches,
            "best_subject": best_subject
        }

        # 5) TREND: số phát biểu theo từng buổi (lọc theo student + class)
        trend_data_query = (
            db.session.query(
                Session.id,
                func.count(SpeechLog.id).label("cnt")
            )
            .join(SpeechLog, SpeechLog.session_id == Session.id)
            .filter(
                SpeechLog.student_id == student_id,
                Session.class_id == user_class_id
            )
            .group_by(Session.id)
            .order_by(Session.id.asc())
            .all()
        )

        trend_data = {
            "labels": [f"Buổi {s[0]}" for s in trend_data_query],
            "data": [s[1] for s in trend_data_query]
        }

        # 6) RADAR: phát biểu theo môn trong lớp (lọc theo user_class_id)
        subjects_in_class = Subject.query.filter_by(class_id=user_class_id).all()
        radar_labels = [s.name for s in subjects_in_class]

        radar_data = []
        for sub in subjects_in_class:
            count = (
                db.session.query(func.count(SpeechLog.id))
                .join(Session, Session.id == SpeechLog.session_id)
                .filter(
                    SpeechLog.student_id == student_id,
                    Session.subject_id == sub.id,
                    Session.class_id == user_class_id
                )
                .scalar()
            ) or 0
            radar_data.append(count)

        radar_chart_data = {"labels": radar_labels, "data": radar_data}

        # 7) AI INSIGHT (giữ logic đơn giản)
        tendency = "Ổn định"
        if len(trend_data["data"]) > 1:
            if trend_data["data"][-1] > trend_data["data"][0]:
                tendency = "Có xu hướng tiến bộ"
            elif trend_data["data"][-1] < trend_data["data"][0]:
                tendency = "Cần cải thiện sự tập trung"

        reason = (
            f"Học sinh hiện có {total_speeches} lượt phát biểu, xếp hạng {student_rank} trong lớp. "
            f"Môn học nổi bật nhất là {best_subject}."
        )
        ai_insight = {"tendency": tendency, "reason": reason}

        # 8) Trả JSON chuẩn
        return jsonify({
            "kpis": student_kpis,
            "trend": trend_data,
            "radar": radar_chart_data,
            "insight": ai_insight
        })

    except Exception as e:
        import traceback
        print(f"--- LỖI TRONG STUDENT ANALYSIS API (ID: {student_id}) ---")
        traceback.print_exc()
        print("--------------------------------------------------------")
        return jsonify({"message": f"Lỗi server khi phân tích: {str(e)}"}), 500

# --- XUẤT DỮ LIỆU ---
@app.route('/api/sessions/export', methods=['POST'])
def export_session_data():
    try:
        export_data_str = request.form.get('export_data')
        if not export_data_str: return "Không có dữ liệu", 400
        
        data = json.loads(export_data_str)
        stats = data.get('stats', {})
        students = data.get('students', [])
        subject_name = data.get('subject_name', 'Unknown Subject')
        session_id = data.get('session_id', 'Unknown_Session')

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Mã Sinh Viên', 'Họ và Tên', 'Số lần phát biểu'])

        student_map = {s['student_code']: s['full_name'] for s in students}
        for student_code, count in stats.items():
            full_name = student_map.get(student_code, 'Không rõ tên')
            writer.writerow([student_code, full_name, count])

        output.seek(0)
        filename = f"ThongKe_BuoiHoc_{session_id}_{subject_name.replace(' ', '_')}.csv"
        return Response(output, mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename={filename}"})
    except Exception as e:
        print(f"Lỗi khi xuất dữ liệu: {e}")
        return "Có lỗi xảy ra trong quá trình xuất file.", 500
# API để lấy thông tin chi tiết của một lớp (dùng cho form Sửa)
@app.route('/api/classes/<int:class_id>', methods=['GET'])
def get_class_details(class_id):
    class_obj = db.session.get(Class, class_id)
    if class_obj:
        return jsonify({
            'id': class_obj.id,
            'name': class_obj.name,
            'academic_year': class_obj.academic_year
        })
    return jsonify({'error': 'Không tìm thấy lớp học'}), 404

# API để cập nhật thông tin một lớp (dùng cho form Sửa)
@app.route('/api/classes/<int:class_id>', methods=['PUT'])
def update_class(class_id):
    class_obj = db.session.get(Class, class_id)
    if not class_obj:
        return jsonify({'status': 'error', 'message': 'Không tìm thấy lớp học'}), 404
    
    data = request.get_json()
    name = data.get('name')
    year = data.get('academic_year')

    if not name or not year:
        return jsonify({'status': 'error', 'message': 'Tên lớp và năm học không được để trống'}), 400

    # Kiểm tra tên mới có trùng với lớp khác không
    existing_class = Class.query.filter(Class.name == name, Class.id != class_id).first()
    if existing_class:
        return jsonify({'status': 'error', 'message': f"Tên lớp '{name}' đã tồn tại"}), 409

    class_obj.name = name
    class_obj.academic_year = year
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Cập nhật lớp học thành công'})

# API để xóa một lớp học
@app.route('/api/classes/<int:class_id>', methods=['DELETE'])
def delete_class(class_id):
    class_obj = db.session.get(Class, class_id)
    if not class_obj:
        return jsonify({'status': 'error', 'message': 'Không tìm thấy lớp học'}), 404
    
    # Kiểm tra xem lớp có học sinh hoặc môn học nào không
    if class_obj.students or class_obj.subjects:
        return jsonify({'status': 'error', 'message': 'Không thể xóa lớp học này vì vẫn còn học sinh hoặc môn học.'}), 409

    db.session.delete(class_obj)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Đã xóa lớp học thành công'})

# Trong app.py, khu vực API ENDPOINTS
@app.route('/api/users/<int:user_id>/assign_class', methods=['POST'])
@login_required
@admin_required
def assign_class_to_user(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'status': 'error', 'message': 'Không tìm thấy người dùng'}), 404
    
    data = request.get_json()
    class_id = data.get('class_id')

    # Nếu class_id là 0 hoặc không có, tức là "Chưa phân công"
    if not class_id or int(class_id) == 0:
        user.class_id = None
    else:
        class_obj = db.session.get(Class, int(class_id))
        if not class_obj:
            return jsonify({'status': 'error', 'message': 'Không tìm thấy lớp học'}), 404
        user.class_id = class_obj.id
    
    db.session.commit()
    return jsonify({'status': 'success', 'message': f'Đã cập nhật phân công cho {user.username}.'})

# =============================================================
# === 5. SOCKET.IO ===
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

# =============================================================
# === 6. KHỐI KHỞI ĐỘNG SERVER (PHẢI LUÔN NẰM CUỐI CÙNG) ===
# =============================================================

@app.context_processor
def inject_user_info():
    if current_user.is_authenticated:
        # Lấy thông tin lớp học từ relationship assigned_class (đã định nghĩa trong database.py)
        user_class = current_user.assigned_class
        class_name = user_class.name if user_class else "Chưa phân công"
        
        # Đếm sĩ số học sinh của riêng lớp đó
        total_students = 0
        if user_class:
            total_students = Student.query.filter_by(class_id=user_class.id).count()
            
        return {
            'header_class_name': class_name,
            'header_total_students': total_students,
            'header_username': current_user.username
        }
    return {
        'header_class_name': "N/A",
        'header_total_students': 0,
        'header_username': "Guest"
    }
if __name__ == '__main__':
    with app.app_context():
        # Tự động tạo bảng nếu chưa có
        db.create_all()
        # Tạo lớp mặc định
        if not db.session.get(Class, 1):
            db.session.add(Class(id=1, name="9/1", academic_year="2025-2026"))
            db.session.commit()
            
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
