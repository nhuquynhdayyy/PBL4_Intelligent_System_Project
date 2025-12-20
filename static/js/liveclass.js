// static/js/liveclass_custom.js
document.addEventListener('DOMContentLoaded', function() {
    // --- KHAI BÁO BIẾN ---
    const socket = io('/live');
    let currentSession = null;
    let studentList = []; // Lưu danh sách học sinh để lấy tên đầy đủ
    let studentStats = {}; // Lưu trữ số lần phát biểu: { 'student_code': count }

    // DOM Elements
    const startSessionBtn = document.getElementById('start-session-btn');
    const endSessionBtn = document.getElementById('end-session-btn');
    const confirmStartBtn = document.getElementById('confirm-start-session');
    const subjectSelectList = document.getElementById('subject-select-list');
    const pendingArea = document.getElementById('pending-confirmation-area');
    const studentListUl = document.getElementById('student-speech-list');
    const totalSpeechesCount = document.getElementById('total-speeches-count');
    const classInfoEl = document.querySelector('.header .class-info'); // Lấy từ layout
    const sessionStatusEl = document.getElementById('session-status-badge');
    const connectionStatusEl = document.getElementById('connection-status');

    const openModal = window.openModal;
    const closeModal = window.closeModal;

    // --- KẾT NỐI SOCKET.IO ---
    socket.on('connect', () => {
        updateConnectionStatus(true);
        fetch('/api/sessions/current')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'found') {
                    currentSession = { id: data.session_id, subjectName: data.subject_name };
                    studentStats = data.speech_counts || {};
                    updateUIForActiveSession();
                    fetchAllStudents();
                }
            });
    });

    socket.on('disconnect', () => {
        updateConnectionStatus(false);
    });

    // --- CÁC HÀM XỬ LÝ SỰ KIỆN ---

    startSessionBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/subjects');
            const subjects = await response.json();
            subjectSelectList.innerHTML = subjects.map(s => `<option value="${s.id}">${s.icon} ${s.name}</option>`).join('');
            if (subjects.length > 0 && typeof openModal === 'function') {
                openModal('selectSubjectModal');
            } else {
                alert('Chưa có môn học nào hoặc hàm openModal không tồn tại.');
            }
        } catch (error) {
            alert('Không thể tải danh sách môn học.');
        }
    });

    confirmStartBtn.addEventListener('click', () => {
        const subjectId = subjectSelectList.value;
        const subjectName = subjectSelectList.options[subjectSelectList.selectedIndex].text;
        
        fetch('/api/sessions/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject_id: subjectId })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                currentSession = { id: data.session_id, subjectName: subjectName };
                studentStats = {};
                updateUIForActiveSession();
                fetchAllStudents();
                if (typeof closeModal === 'function') closeModal('selectSubjectModal');
            } else {
                alert('Lỗi: ' + data.message);
            }
        });
    });

    endSessionBtn.addEventListener('click', () => {
        if (!currentSession || !confirm('Bạn có chắc muốn kết thúc buổi học này không?')) return;
        
        fetch('/api/sessions/end', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSession.id })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                alert('Buổi học đã kết thúc.');
                updateUIForInactiveSession();
            } else {
                alert('Lỗi: ' + data.message);
            }
        });
    });

    // --- LẮNG NGHE SỰ KIỆN TỪ SERVER ---
    socket.on('pending_recognition', function(data) {
        const pendingId = `pending-${data.student_id}`;
        if (document.getElementById(pendingId) || !currentSession) return;

        if (pendingArea.querySelector('.placeholder-text')) {
            pendingArea.innerHTML = '';
        }

        const pendingItem = document.createElement('div');
        pendingItem.className = 'pending-item';
        pendingItem.id = pendingId;
        pendingItem.innerHTML = `
            <span><strong>${data.full_name}</strong>?</span>
            <div class="actions">
                <button class="btn-accept">✓</button>
                <button class="btn-decline">✗</button>
            </div>
        `;

        pendingItem.querySelector('.btn-accept').addEventListener('click', () => {
            socket.emit('confirm_recognition', {
                student_id: data.student_id,
                action: 'accept'
            });
            pendingItem.remove();
            checkEmptyPendingArea();
        });

        pendingItem.querySelector('.btn-decline').addEventListener('click', () => {
            pendingItem.remove();
            checkEmptyPendingArea();
        });

        pendingArea.appendChild(pendingItem);
    });

    socket.on('speech_update', function(data) {
        if (!studentStats.hasOwnProperty(data.student_code)) {
            studentStats[data.student_code] = 0;
        }
        studentStats[data.student_code]++;
        renderStudentList();
    });

    // --- CÁC HÀM TIỆN ÍCH ---
    function updateConnectionStatus(isConnected) {
        const icon = connectionStatusEl.querySelector('.icon');
        const text = connectionStatusEl.querySelector('.text');
        if (isConnected) {
            icon.className = 'icon text-success';
            text.textContent = 'Đã kết nối';
        } else {
            icon.className = 'icon text-danger';
            text.textContent = 'Mất kết nối';
        }
    }
    
    function updateUIForActiveSession() {
        startSessionBtn.style.display = 'none';
        endSessionBtn.style.display = 'inline-block';
        sessionStatusEl.textContent = 'Đang diễn ra';
        sessionStatusEl.className = 'session-status bg-success';
    }
    
    function updateUIForInactiveSession() {
        startSessionBtn.style.display = 'inline-block';
        endSessionBtn.style.display = 'none';
        sessionStatusEl.textContent = 'Chưa bắt đầu';
        sessionStatusEl.className = 'session-status bg-secondary';
        studentListUl.innerHTML = '<li class="placeholder-text">Bắt đầu buổi học để xem danh sách</li>';
        totalSpeechesCount.textContent = '0';
        currentSession = null;
        studentStats = {};
    }

    function checkEmptyPendingArea() {
        if (pendingArea.children.length === 0) {
            pendingArea.innerHTML = '<p class="placeholder-text">Chưa có yêu cầu nào...</p>';
        }
    }

    async function fetchAllStudents() {
        try {
            const response = await fetch('/api/students');
            studentList = await response.json(); // Lưu lại để tra cứu tên
            studentList.forEach(s => {
                if (!studentStats.hasOwnProperty(s.student_code)) {
                    studentStats[s.student_code] = 0;
                }
            });
            renderStudentList();
        } catch (error) {
            console.error('Lỗi khi tải danh sách học sinh:', error);
        }
    }
    
    function renderStudentList() {
        const sortedStudentCodes = Object.keys(studentStats).sort((a, b) => studentStats[b] - studentStats[a]);
        
        studentListUl.innerHTML = sortedStudentCodes.map(code => {
            const studentInfo = studentList.find(s => s.student_code === code);
            const fullName = studentInfo ? studentInfo.full_name : code;
            return `
                <li>
                    <span>${fullName}</span>
                    <span class="count">${studentStats[code]}</span>
                </li>
            `;
        }).join('');
        
        const total = Object.values(studentStats).reduce((sum, count) => sum + count, 0);
        totalSpeechesCount.textContent = total;
    }
});