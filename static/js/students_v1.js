// static/js/students.js (FULL CODE)

document.addEventListener('DOMContentLoaded', function() {
    const studentList = document.getElementById('studentList');
    const addStudentBtn = document.getElementById('addStudentBtn');

    // === CÁC BIẾN CHO MODAL HỌC SINH ===
    const studentModal = document.getElementById('studentModal');
    const studentModalTitle = document.getElementById('studentModalTitle');
    const studentForm = document.getElementById('studentForm');
    const studentIdInput = document.getElementById('studentId');
    const studentNameInput = document.getElementById('studentName');
    const studentDobInput = document.getElementById('studentDob');
    const studentGenderInput = document.getElementById('studentGender');
    const studentCodeSelect = document.getElementById('studentCode');
    
    // === CÁC BIẾN CHO MODAL ĐIỂM SỐ ===
    const gradeModal = document.getElementById('gradeModal');
    const gradeModalTitle = document.getElementById('gradeModalTitle');
    const gradeForm = document.getElementById('gradeForm');
    const gradeFormTitle = document.getElementById('gradeFormTitle');
    const gradeStudentIdInput = document.getElementById('gradeStudentId');
    const gradeIdInput = document.getElementById('gradeId');
    const gradeSubjectSelect = document.getElementById('gradeSubject');
    const gradeScoreInput = document.getElementById('gradeScore');
    const gradeTypeSelect = document.getElementById('gradeType');
    const gradeTermSelect = document.getElementById('gradeTerm');
    const gradeExamDateInput = document.getElementById('gradeExamDate');
    const gradeListContainer = document.getElementById('gradeList');
    const cancelGradeEditBtn = document.getElementById('cancelGradeEditBtn');


    // =============================================================
    // === CHỨC NĂNG CHUNG VÀ TẢI DỮ LIỆU BAN ĐẦU ===
    // =============================================================

    // Hàm tải danh sách học sinh
    async function loadStudents() {
        try {
            const response = await fetch('/api/students');
            if (!response.ok) throw new Error('Network response was not ok');
            const students = await response.json();
            renderStudents(students);
        } catch (error) {
            studentList.innerHTML = `<p class="error">Lỗi tải danh sách học sinh: ${error.message}</p>`;
        }
    }

    // Hàm hiển thị danh sách học sinh
    function renderStudents(students) {
        if (students.length === 0) {
            studentList.innerHTML = '<p>Chưa có học sinh nào trong lớp.</p>';
            return;
        }
        studentList.innerHTML = students.map(s => `
            <div class="list-item">
                <div class="item-main">
                    <i class="fas fa-user-graduate item-icon"></i>
                    <div class="item-details">
                        <span class="item-name">${s.full_name}</span>
                        <small class="item-meta">Face ID: <strong>${s.student_code}</strong></small>
                    </div>
                </div>
                <div class="item-actions">
                    <!-- THÊM MỚI: Nút Phân tích AI -->
                    <button class="btn btn-success btn-sm" data-action="analyze-student" data-student-id="${s.id}" data-student-name="${s.full_name}">
                        <i class="fas fa-brain"></i> Phân tích AI
                    </button>
                    <button class="btn btn-info btn-sm" data-action="manage-grades" data-student-id="${s.id}" data-student-name="${s.full_name}">
                        <i class="fas fa-chart-bar"></i> Quản lý điểm
                    </button>
                    <button class="btn btn-primary btn-sm" data-action="edit-student" data-id="${s.id}">
                        <i class="fas fa-edit"></i> Sửa
                    </button>
                    <button class="btn btn-danger btn-sm" data-action="delete-student" data-id="${s.id}">
                        <i class="fas fa-trash"></i> Xóa
                    </button>
                </div>
            </div>
        `).join('');
    }

    // Tải danh sách các Face ID chưa được sử dụng
    async function loadUntrainedFaces() {
        try {
            const response = await fetch('/api/untrained_faces');
            const availableCodes = await response.json();
            studentCodeSelect.innerHTML = '<option value="">-- Chọn Face ID --</option>';
            availableCodes.forEach(code => {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = code;
                studentCodeSelect.appendChild(option);
            });
        } catch (error) {
            console.error("Lỗi tải Face IDs:", error);
            studentCodeSelect.innerHTML = '<option value="">Lỗi tải danh sách</option>';
        }
    }
    
    // Tải danh sách môn học cho dropdown
    async function loadSubjectsIntoSelect(selectElement) {
        try {
            const response = await fetch('/api/subjects');
            const subjects = await response.json();
            selectElement.innerHTML = '<option value="">-- Chọn môn học --</option>';
            subjects.forEach(subject => {
                const option = document.createElement('option');
                option.value = subject.id;
                option.textContent = subject.name;
                selectElement.appendChild(option);
            });
        } catch (error) {
            console.error("Lỗi tải môn học:", error);
            selectElement.innerHTML = '<option value="">Lỗi tải danh sách</option>';
        }
    }


    // =============================================================
    // === XỬ LÝ SỰ KIỆN CRUD HỌC SINH ===
    // =============================================================

    // Mở modal thêm học sinh
    addStudentBtn.addEventListener('click', () => {
        studentModalTitle.textContent = '➕ Thêm học sinh mới';
        studentForm.reset();
        studentIdInput.value = '';
        loadUntrainedFaces(); // Tải lại danh sách face ID mỗi khi mở
        studentCodeSelect.disabled = false;
        openModal('studentModal');
    });
    
    // Xử lý submit form học sinh (Thêm/Sửa)
    studentForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const studentId = studentIdInput.value;
        const url = studentId ? `/api/students/${studentId}` : '/api/students';
        const method = studentId ? 'PUT' : 'POST';

        // CẬP NHẬT: Gửi thêm date_of_birth và gender
        const body = {
            name: studentNameInput.value,
            code: studentCodeSelect.value,
            date_of_birth: studentDobInput.value || null,
            gender: studentGenderInput.value || null
        };

        try {
            const response = await fetch(url, {
                method: method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.message || 'Có lỗi xảy ra');
            }
            alert(result.message);
            closeModal('studentModal');
            loadStudents();
        } catch (error) {
            alert(`Lỗi: ${error.message}`);
        }
    });

    // Bắt sự kiện click trên toàn bộ danh sách (Sửa/Xóa học sinh & Mở modal điểm)
    studentList.addEventListener('click', async function(e) {
        const target = e.target.closest('button[data-action]');
        if (!target) return;

        const action = target.dataset.action;
        const id = target.dataset.id || target.dataset.studentId;
        
        // --- SỬA HỌC SINH ---
        if (action === 'edit-student') {
            try {
                const response = await fetch(`/api/students/${id}`);
                const student = await response.json();
                studentModalTitle.textContent = '✏️ Chỉnh sửa thông tin học sinh';
                studentIdInput.value = student.id;
                studentNameInput.value = student.full_name;
                // CẬP NHẬT: Điền dữ liệu ngày sinh, giới tính vào form
                studentDobInput.value = student.date_of_birth || '';
                studentGenderInput.value = student.gender || '';
                
                // Hiển thị Face ID hiện tại và không cho sửa
                studentCodeSelect.innerHTML = `<option value="${student.student_code}">${student.student_code}</option>`;
                studentCodeSelect.value = student.student_code;
                studentCodeSelect.disabled = true;
                
                openModal('studentModal');
            } catch (error) {
                alert('Không thể tải thông tin học sinh.');
            }
        }
        
        // --- XÓA HỌC SINH ---
        if (action === 'delete-student') {
            if (confirm('Bạn có chắc chắn muốn xóa học sinh này? Mọi dữ liệu liên quan (phát biểu, điểm số) cũng sẽ bị xóa vĩnh viễn.')) {
                try {
                    const response = await fetch(`/api/students/${id}`, { method: 'DELETE' });
                    const result = await response.json();
                    alert(result.message);
                    loadStudents();
                } catch (error) {
                    alert('Xóa thất bại.');
                }
            }
        }
        
        // --- MỞ MODAL QUẢN LÝ ĐIỂM ---
        if (action === 'manage-grades') {
            const studentName = target.dataset.studentName;
            openGradeModalForStudent(id, studentName);
        }

        // --- THÊM MỚI: XỬ LÝ PHÂN TÍCH AI ---
        if (action === 'analyze-student') {
            aiModalTitle.innerHTML = `<i class="fas fa-brain text-primary"></i> Phân tích AI - ${name}`;
            
            // 1. Hiển thị trạng thái đang tải
            aiAnalysisContent.innerHTML = `
                <div class="text-center p-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">AI đang phân tích dữ liệu...</p>
                </div>`;
            openModal('aiAnalysisModal');

            // 2. Gọi API
            const result = await apiCall(`/api/students/${id}/analysis`);

            // 3. Hiển thị kết quả
            if (result) {
                aiAnalysisContent.innerHTML = `
                    <div class="alert alert-info">
                        <h4 class="alert-heading">Kết luận: ${result.tendency}</h4>
                        <hr>
                        <p class="mb-0"><strong>Dữ liệu phân tích:</strong> ${result.reason}</p>
                    </div>
                `;
            } else {
                aiAnalysisContent.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Lỗi:</strong> Không thể thực hiện phân tích. Vui lòng thử lại.
                    </div>
                `;
            }
        }
    });


    // =============================================================
    // === XỬ LÝ SỰ KIỆN CRUD ĐIỂM SỐ (KHÔNG THAY ĐỔI) ===
    // =============================================================
    
    async function openGradeModalForStudent(studentId, studentName) {
        gradeModalTitle.textContent = `📊 Quản lý điểm - ${studentName}`;
        gradeStudentIdInput.value = studentId;
        resetGradeForm();
        await Promise.all([
            loadSubjectsIntoSelect(gradeSubjectSelect),
            loadGradesForStudent(studentId)
        ]);
        openModal('gradeModal');
    }

    async function loadGradesForStudent(studentId) {
        gradeListContainer.innerHTML = '<p>Đang tải điểm...</p>';
        try {
            const response = await fetch(`/api/students/${studentId}/grades`);
            const grades = await response.json();
            if (grades.length === 0) {
                gradeListContainer.innerHTML = '<p class="placeholder-text">Học sinh này chưa có điểm nào.</p>';
                return;
            }
            const tableRows = grades.map(g => `
                <tr>
                    <td>${g.subject_name}</td>
                    <td><strong>${g.score}</strong></td>
                    <td>${g.grade_type}</td>
                    <td>${g.term}</td>
                    <td>${new Date(g.exam_date).toLocaleDateString('vi-VN')}</td>
                    <td>
                        <button class="btn btn-secondary btn-sm" data-action="edit-grade" data-grade-data='${JSON.stringify(g)}'><i class="fas fa-edit"></i></button>
                        <button class="btn btn-danger btn-sm" data-action="delete-grade" data-grade-id="${g.id}"><i class="fas fa-trash"></i></button>
                    </td>
                </tr>`).join('');
            gradeListContainer.innerHTML = `<table><thead><tr><th>Môn học</th><th>Điểm</th><th>Loại</th><th>Học kỳ</th><th>Ngày</th><th></th></tr></thead><tbody>${tableRows}</tbody></table>`;
        } catch (error) {
            gradeListContainer.innerHTML = `<p class="error">Lỗi tải điểm: ${error.message}</p>`;
        }
    }
    
    function resetGradeForm() {
        gradeForm.reset();
        gradeIdInput.value = '';
        gradeFormTitle.textContent = '📝 Thêm điểm mới';
        cancelGradeEditBtn.style.display = 'none';
        gradeExamDateInput.valueAsDate = new Date();
    }
    
    gradeForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const gradeId = gradeIdInput.value;
        const studentId = gradeStudentIdInput.value;
        const url = gradeId ? `/api/grades/${gradeId}` : '/api/grades';
        const method = gradeId ? 'PUT' : 'POST';
        const body = {
            student_id: studentId,
            subject_id: gradeSubjectSelect.value,
            score: gradeScoreInput.value,
            grade_type: gradeTypeSelect.value,
            term: gradeTermSelect.value,
            exam_date: gradeExamDateInput.value
        };
        try {
            const response = await fetch(url, { method: method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const result = await response.json();
            if (!response.ok) throw new Error(result.message);
            alert(result.message);
            resetGradeForm();
            loadGradesForStudent(studentId);
        } catch (error) {
            alert(`Lỗi: ${error.message}`);
        }
    });
    
    cancelGradeEditBtn.addEventListener('click', resetGradeForm);

    gradeListContainer.addEventListener('click', async function(e) {
        const target = e.target.closest('button[data-action]');
        if (!target) return;
        const action = target.dataset.action;
        const gradeId = target.dataset.gradeId;

        if (action === 'edit-grade') {
            const gradeData = JSON.parse(target.dataset.gradeData);
            gradeFormTitle.textContent = '✏️ Chỉnh sửa điểm';
            gradeIdInput.value = gradeData.id;
            gradeSubjectSelect.value = gradeData.subject_id;
            gradeScoreInput.value = gradeData.score;
            gradeTypeSelect.value = gradeData.grade_type;
            gradeTermSelect.value = gradeData.term;
            gradeExamDateInput.value = gradeData.exam_date;
            cancelGradeEditBtn.style.display = 'inline-block';
            gradeScoreInput.focus();
        }

        if (action === 'delete-grade') {
            if (confirm('Bạn có chắc chắn muốn xóa điểm này?')) {
                try {
                    const response = await fetch(`/api/grades/${gradeId}`, { method: 'DELETE' });
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.message);
                    alert(result.message);
                    loadGradesForStudent(gradeStudentIdInput.value);
                } catch (error) {
                    alert(`Lỗi: ${error.message}`);
                }
            }
        }
    });

    // Khởi chạy
    loadStudents();
});
