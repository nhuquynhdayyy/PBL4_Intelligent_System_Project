# app.py
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from database import db, Class, Student, Subject, Session, SpeechLog
import os, threading
from dotenv import load_dotenv
from sqlalchemy import func, desc
from datetime import datetime

# --- KHỞI TẠO ỨNG DỤNG VÀ SOCKETIO ---
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_change_it_later!'
db_user, db_password, db_host, db_name = os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_HOST'), os.getenv('DB_NAME')
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- KHỞI TẠO ỨNG DỤNG VÀ SOCKETIO ---
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_change_it_later!'
db_user, db_password, db_host, db_name = os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_HOST'), os.getenv('DB_NAME')
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")


# --- THÊM MỚI: BIẾN TOÀN CỤC ĐỂ LƯU TRỮ VIDEO STREAM ---
# output_frame sẽ lưu trữ frame ảnh JPEG mới nhất nhận được từ AI server
# lock là một khóa để đảm bảo việc đọc/ghi frame được an toàn, tránh xung đột
output_frame = None
lock = threading.Lock()


# =============================================================
# === CÁC ROUTE ĐỂ RENDER TRANG ===
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
# === API ENDPOINTS ===
# =============================================================

# --- THÊM MỚI: ENDPOINT ĐỂ NHẬN VIDEO TỪ AI SERVER ---
@app.route('/api/video_stream/push', methods=['POST'])
def video_stream_push():
    """
    Đây là "Cổng nhận tin tức". AI Server sẽ liên tục gửi (POST)
    dữ liệu ảnh JPEG thô (raw bytes) đến endpoint này.
    """
    global output_frame
    # Dùng lock để đảm bảo việc gán giá trị mới là an toàn
    with lock:
        output_frame = request.data
    return jsonify({'status': 'ok'})


# --- THÊM MỚI: ENDPOINT ĐỂ TRÌNH DUYỆT XEM VIDEO STREAM ---
def generate_video_stream():
    """
    Đây là một generator function. Nó sẽ liên tục đọc biến `output_frame`
    và "yield" (đẩy ra) từng frame theo định dạng multipart.
    """
    while True:
        # Dùng lock để đọc frame một cách an toàn
        with lock:
            # Nếu chưa có frame nào thì bỏ qua vòng lặp này
            if output_frame is None:
                continue
            # Sao chép frame để có thể giải phóng lock nhanh chóng
            frame_bytes = output_frame[:]
        
        # Gửi frame đi theo định dạng của MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """
    Đây là "Kênh phát sóng". Trình duyệt sẽ kết nối tới đây.
    Nó trả về một đối tượng Response, sử dụng generator ở trên để
    liên tục stream dữ liệu.
    """
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API CRUD MÔN HỌC ---
@app.route('/api/subjects', methods=['GET', 'POST'])
def subjects_api():
    class_id = 1
    if request.method == 'POST':
        data = request.get_json()
        new_subject = Subject(name=data['name'], icon=data.get('icon', '📚'), class_id=class_id)
        db.session.add(new_subject)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã thêm môn học '{data['name']}'."})
    # GET
    subjects_query = Subject.query.filter_by(class_id=class_id).order_by(Subject.name).all()
    result = []
    for s in subjects_query:
        session_count = Session.query.filter_by(subject_id=s.id).count()
        total_speeches = db.session.query(func.count(SpeechLog.id)).join(Session).filter(Session.subject_id == s.id).scalar()
        result.append({'id': s.id, 'name': s.name, 'icon': s.icon, 'session_count': session_count, 'total_speeches': total_speeches})
    return jsonify(result)

@app.route('/api/subjects/<int:subject_id>', methods=['GET', 'PUT', 'DELETE'])
def single_subject_api(subject_id):
    subject = Subject.query.get_or_404(subject_id)
    if request.method == 'PUT':
        data = request.get_json()
        subject.name = data['name']
        subject.icon = data.get('icon', '📚')
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã cập nhật môn học '{subject.name}'."})
    elif request.method == 'DELETE':
        db.session.delete(subject)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã xóa môn học '{subject.name}'."})
    # GET
    return jsonify({'id': subject.id, 'name': subject.name, 'icon': subject.icon})


# --- API CRUD HỌC SINH ---
@app.route('/api/students', methods=['GET', 'POST'])
def students_api():
    class_id = 1
    if request.method == 'POST':
        data = request.get_json()
        existing_student = Student.query.filter_by(student_code=data['code']).first()
        if existing_student:
            return jsonify({'status': 'error', 'message': f"Face ID '{data['code']}' đã được gán cho học sinh khác."}), 409
        new_student = Student(full_name=data['name'], student_code=data['code'], class_id=class_id)
        db.session.add(new_student)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã thêm học sinh '{data['name']}'."})
    # GET
    students_query = Student.query.filter_by(class_id=class_id).order_by(Student.full_name).all()
    result = []
    for s in students_query:
        total_speeches = SpeechLog.query.filter_by(student_id=s.id).count()
        result.append({'id': s.id, 'full_name': s.full_name, 'student_code': s.student_code, 'total_speeches': total_speeches})
    return jsonify(result)

@app.route('/api/students/<int:student_id>', methods=['GET', 'PUT', 'DELETE'])
def single_student_api(student_id):
    student = Student.query.get_or_404(student_id)
    if request.method == 'PUT':
        data = request.get_json()
        student.full_name = data['name']
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã cập nhật học sinh '{student.full_name}'."})
    elif request.method == 'DELETE':
        db.session.delete(student)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f"Đã xóa học sinh '{student.full_name}'."})
    # GET
    return jsonify({'id': student.id, 'full_name': student.full_name, 'student_code': student.student_code})


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

# Thêm một hàm xử lý SocketIO mới
@socketio.on('confirm_recognition', namespace='/live')
def handle_confirm_recognition(data):
    """
    Hàm này được gọi khi giáo viên nhấn nút Chấp nhận/Từ chối.
    Data nhận được sẽ có dạng: {'student_id': 123, 'action': 'accept'}
    """
    action = data.get('action')
    student_id = data.get('student_id')
    
    # Chỉ xử lý khi được chấp nhận
    if action == 'accept':
        current_session = Session.query.filter_by(status='ongoing').first()
        student = Student.query.get(student_id)

        if current_session and student:
            # BÂY GIỜ MỚI GHI VÀO DATABASE
            new_log = SpeechLog(student_id=student.id, session_id=current_session.id)
            db.session.add(new_log)
            db.session.commit()
            
            print(f"[CONFIRMED] Ghi nhận: {student.full_name}")
            
            # Phát sự kiện cập nhật cho toàn bộ lớp học như cũ
            socketio.emit('speech_update', {
                'student_code': student.student_code,
                'full_name': student.full_name,
                'timestamp': datetime.utcnow().strftime('%H:%M:%S')
            }, namespace='/live')

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

# =============================================================
# === SOCKET.IO EVENTS CHO LIVE CLASS ===
# =============================================================
@socketio.on('connect', namespace='/live')
def handle_live_connect():
    print('Client đã kết nối tới Live Class.')

@socketio.on('disconnect', namespace='/live')
def handle_live_disconnect():
    print('Client đã ngắt kết nối khỏi Live Class.')

# --- CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    with app.app_context():
        # db.create_all() # Chạy 1 lần để tạo DB, sau đó comment lại
        # Tự động tạo lớp học mặc định nếu chưa có
        default_class = db.session.get(Class, 1) # Thử lấy lớp có id=1
        if not default_class:
            print("[INFO] Lớp học mặc định (ID=1) không tồn tại. Đang tạo...")
            new_class = Class(id=1, name="10/1", academic_year="2025-2026")
            db.session.add(new_class)
            db.session.commit()
            print("[INFO] Đã tạo lớp học mặc định.")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)