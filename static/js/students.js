// static/js/students.js
document.addEventListener('DOMContentLoaded', function () {
    const studentList = document.getElementById('studentList');
    const addStudentBtn = document.getElementById('addStudentBtn');
    const exportGradeBtn = document.getElementById('exportGradeBtn');

    // Elements cho Học sinh
    const studentForm = document.getElementById('studentForm');
    const studentModalTitle = document.getElementById('studentModalTitle');
    const studentIdInput = document.getElementById('studentId');
    const studentNameInput = document.getElementById('studentName');
    const studentDobInput = document.getElementById('studentDob');
    const studentGenderInput = document.getElementById('studentGender');
    const studentCodeSelect = document.getElementById('studentCode');

    // Elements cho Điểm
    const gradeForm = document.getElementById('gradeForm');
    const gradeStudentIdInput = document.getElementById('gradeStudentId');
    const gradeIdInput = document.getElementById('gradeId');
    const gradeSubjectSelect = document.getElementById('gradeSubject');
    const gradeScoreInput = document.getElementById('gradeScore');
    const gradeTypeSelect = document.getElementById('gradeType');
    const gradeTermSelect = document.getElementById('gradeTerm');
    const gradeExamDateInput = document.getElementById('gradeExamDate');
    const gradeListContainer = document.getElementById('gradeList');

    // ==========================================
    // 1. TẢI DANH SÁCH HỌC SINH
    // ==========================================
    async function loadStudents() {
        const students = await apiCall('/api/students');
        if (!students) return;

        studentList.innerHTML = students.map(s => `
            <div class="card mb-3 shadow-sm border-0" style="border-radius: 12px;">
                <div class="card-body d-flex justify-content-between align-items-center p-3">
                    <div class="d-flex align-items-center">
                        <div class="student-avatar me-3" style="width: 45px; height: 45px; background: #eef2ff; color: #4f46e5; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem;">
                            ${s.full_name?.charAt(0) || "?"}
                        </div>
                        <div>
                            <h6 class="mb-0 fw-bold" style="color: #1e293b;">${s.full_name}</h6>
                            <small class="text-muted">Face ID: <span class="badge bg-light text-dark border">${s.student_code}</span></small>
                        </div>
                    </div>
                    <div class="btn-group-sm">
                        <button class="btn btn-success btn-sm px-3" data-action="analyze-student" data-id="${s.id}" data-name="${s.full_name}">
                            <i class="fas fa-brain me-1"></i> Phân tích AI
                        </button>
                        <button class="btn btn-info btn-sm px-3 text-white" data-action="manage-grades" data-id="${s.id}" data-name="${s.full_name}">
                            <i class="fas fa-chart-bar me-1"></i> Điểm
                        </button>
                        <button class="btn btn-primary btn-sm px-3" data-action="edit-student" data-id="${s.id}">
                            <i class="fas fa-edit me-1"></i> Sửa
                        </button>
                        <button class="btn btn-danger btn-sm px-3" data-action="delete-student" data-id="${s.id}" data-name="${s.full_name}">
                            <i class="fas fa-trash me-1"></i> Xóa
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    }

    // ==========================================
    // 2. CLICK TRÊN DANH SÁCH HỌC SINH
    // ==========================================
    studentList.addEventListener('click', async (e) => {
        const btn = e.target.closest('button');
        if (!btn) return;

        const id = btn.dataset.id;
        const name = btn.dataset.name;
        const action = btn.dataset.action;

        if (action === 'delete-student') {
            if (confirm(`Bạn có chắc muốn xóa học sinh ${name}?`)) {
                const res = await apiCall(`/api/students/${id}`, 'DELETE');
                if (res) loadStudents();
            }
            return;
        }

        if (action === 'edit-student') {
            const s = await apiCall(`/api/students/${id}`);
            if (s) {
                studentIdInput.value = s.id;
                studentNameInput.value = s.full_name;
                studentDobInput.value = s.date_of_birth || '';
                studentGenderInput.value = s.gender || '';

                studentModalTitle.textContent = '✏️ Sửa thông tin học sinh';
                studentCodeSelect.innerHTML = `<option value="${s.student_code}">${s.student_code}</option>`;
                studentCodeSelect.disabled = true;
                openModal('studentModal');
            }
            return;
        }

        if (action === 'manage-grades') {
            openGradeModal(id, name);
            return;
        }

        if (action === 'analyze-student') {
            const aiModalTitle = document.getElementById('aiModalTitle');
            const aiAnalysisContent = document.getElementById('aiAnalysisContent');

            aiModalTitle.innerHTML = `<i class="fas fa-brain text-primary"></i> AI Phân tích - ${name}`;
            aiAnalysisContent.innerHTML = `<div class="text-center p-4"><div class="spinner-border text-primary"></div><p class="mt-2">Đang phân tích...</p></div>`;

            openModal('aiAnalysisModal');

            const result = await apiCall(`/api/students/${id}/analysis`);
            if (!result) return;

            // ✅ API mới trả về: { kpis, trend, radar, insight: {tendency, reason} }
            const tendency = result.insight?.tendency || "N/A";
            const reason = result.insight?.reason || "Không có nhận định.";

            aiAnalysisContent.innerHTML = `
                <div class="alert alert-info border-0 shadow-sm">
                    <h5 class="fw-bold">Kết luận: ${tendency}</h5>
                    <hr>
                    <p class="mb-0">${reason}</p>
                </div>
            `;
        }
    });

    // ==========================================
    // 3. THÊM MỚI HỌC SINH
    // ==========================================
    addStudentBtn?.addEventListener('click', async () => {
        studentForm.reset();
        studentIdInput.value = '';
        studentModalTitle.textContent = '➕ Thêm học sinh mới';
        studentCodeSelect.disabled = false;

        openModal('studentModal');

        studentCodeSelect.innerHTML = '<option value="">-- Đang tải Face ID... --</option>';
        const codes = await apiCall('/api/untrained_faces');

        if (codes && codes.length > 0) {
            studentCodeSelect.innerHTML =
                '<option value="">-- Chọn Face ID --</option>' +
                codes.map(c => `<option value="${c}">${c}</option>`).join('');
        } else {
            studentCodeSelect.innerHTML = '<option value="" disabled>-- Không có Face ID mới --</option>';
        }
    });

    studentForm?.addEventListener('submit', async (e) => {
        e.preventDefault();

        const id = studentIdInput.value;
        const data = {
            name: studentNameInput.value,
            code: studentCodeSelect.value,
            date_of_birth: studentDobInput.value,
            gender: studentGenderInput.value
        };

        const url = id ? `/api/students/${id}` : '/api/students';
        const method = id ? 'PUT' : 'POST';

        const res = await apiCall(url, method, data);
        if (res) {
            closeModal('studentModal');
            loadStudents();
        }
    });

    // ==========================================
    // 4. QUẢN LÝ ĐIỂM SỐ
    // ==========================================
    async function openGradeModal(studentId, studentName) {
        document.getElementById('gradeModalTitle').textContent = `Điểm - ${studentName}`;

        gradeStudentIdInput.value = studentId;
        gradeIdInput.value = '';
        gradeForm.reset();

        // Load môn học
        const subjects = await apiCall('/api/subjects');
        if (subjects) {
            gradeSubjectSelect.innerHTML = subjects.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
        }

        await loadGrades(studentId);
        openModal('gradeModal');
    }

    async function loadGrades(studentId) {
        const grades = await apiCall(`/api/students/${studentId}/grades`);
        if (!grades) return;

        gradeListContainer.innerHTML = grades.map(g => `
            <div class="border-bottom py-2 d-flex justify-content-between align-items-center">
                <span><b>${g.subject_name}</b>: ${g.score} (${g.grade_type})</span>
                <div>
                    <button class="btn btn-sm btn-outline-primary"
                        data-action="edit-grade"
                        data-grade-id="${g.id}"
                        data-subject="${g.subject_id}"
                        data-score="${g.score}"
                        data-type="${g.grade_type}"
                        data-term="${g.term}"
                        data-date="${g.exam_date}">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger"
                        data-action="delete-grade"
                        data-grade-id="${g.id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `).join('') || '<p class="text-muted mt-2">Chưa có điểm nào.</p>';
    }

    gradeListContainer?.addEventListener('click', async (e) => {
        const btn = e.target.closest('button');
        if (!btn) return;

        const gId = btn.dataset.gradeId;

        if (btn.dataset.action === 'delete-grade') {
            if (confirm("Xóa điểm này?")) {
                const res = await apiCall(`/api/grades/${gId}`, 'DELETE');
                if (res) loadGrades(gradeStudentIdInput.value);
            }
            return;
        }

        if (btn.dataset.action === 'edit-grade') {
            gradeIdInput.value = gId;
            gradeSubjectSelect.value = btn.dataset.subject;
            gradeScoreInput.value = btn.dataset.score;
            gradeTypeSelect.value = btn.dataset.type;
            gradeTermSelect.value = btn.dataset.term;
            gradeExamDateInput.value = btn.dataset.date;
        }
    });

    gradeForm?.addEventListener('submit', async (e) => {
        e.preventDefault();

        const id = gradeIdInput.value;
        const data = {
            student_id: gradeStudentIdInput.value,
            subject_id: gradeSubjectSelect.value,
            score: gradeScoreInput.value,
            grade_type: gradeTypeSelect.value,
            term: gradeTermSelect.value,
            exam_date: gradeExamDateInput.value
        };

        const url = id ? `/api/grades/${id}` : '/api/grades';
        const method = id ? 'PUT' : 'POST';

        const res = await apiCall(url, method, data);
        if (res) {
            gradeForm.reset();
            gradeIdInput.value = '';
            loadGrades(gradeStudentIdInput.value);
        }
    });

    // ==========================================
    // 5. XUẤT EXCEL AI (GỘP VÀO TRONG DOMContentLoaded)
    // ==========================================
    exportGradeBtn?.addEventListener('click', () => {
        const studentId = gradeStudentIdInput?.value; // ✅ lấy đúng input đang dùng trong modal
        if (!studentId) {
            alert("Vui lòng mở mục Điểm của 1 học sinh trước, rồi bấm Xuất Excel!");
            return;
        }

        const oldHtml = exportGradeBtn.innerHTML;
        exportGradeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> AI đang phân tích...';

        // Tải file bằng điều hướng
        window.location.href = `/api/students/${studentId}/export_smart_report`;

        setTimeout(() => {
            exportGradeBtn.innerHTML = oldHtml || '<i class="fas fa-file-excel me-1"></i> Xuất Excel AI';
        }, 3000);
    });

    // Khởi chạy
    loadStudents();
});
