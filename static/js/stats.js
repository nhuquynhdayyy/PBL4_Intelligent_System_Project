document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const topStudentsContainer = document.getElementById('topStudentsRanking');
    const subjectAnalysisContainer = document.getElementById('subjectAnalysis');
    const aiAnalysisResultContainer = document.getElementById('aiAnalysisResult');
    const studentSelectDropdown = document.getElementById('student-select-for-analysis');

    // ... (Hàm renderTopStudents và renderSubjectAnalysis giữ nguyên)

    function renderTopStudents(students) {
        if (!students || students.length === 0) {
            topStudentsContainer.innerHTML = '<p class="placeholder-text">Chưa có dữ liệu.</p>'; return;
        }
        const rankColors = ['rank-1', 'rank-2', 'rank-3'];
        topStudentsContainer.innerHTML = `<ol>${students.map((student, index) => `
            <li>
                <span class="rank-badge ${rankColors[index] || 'rank-other'}">${index + 1}</span>
                <span class="student-name">${student.name}</span>
                <span class="student-score">${student.speeches}</span>
            </li>`).join('')}</ol>`;
    }

    function renderSubjectAnalysis(subjects) {
        if (!subjects || subjects.length === 0) {
            subjectAnalysisContainer.innerHTML = '<p class="placeholder-text">Chưa có dữ liệu.</p>'; return;
        }
        subjectAnalysisContainer.innerHTML = subjects.map(subject => {
            const topStudentText = subject.top_student_name === 'Chưa có' ? 'Chưa có' : `${subject.top_student_name} (${subject.top_student_speeches} lần)`;
            return `
            <div class="subject-item">
                <h4>${subject.icon || '📚'} ${subject.name}</h4>
                <p><strong>Buổi học:</strong> ${subject.session_count}</p>
                <p><strong>Tích cực nhất:</strong> ${topStudentText}</p>
            </div>`;
        }).join('');
    }


    // --- THÊM MỚI: CÁC HÀM CHO PHÂN TÍCH AI ---
    
    // Hàm tải danh sách học sinh vào dropdown
    async function populateStudentDropdown() {
        const students = await apiCall('/api/students');
        if (students && students.length > 0) {
            studentSelectDropdown.innerHTML = '<option value="">-- Chọn học sinh --</option>';
            students.forEach(student => {
                const option = document.createElement('option');
                option.value = student.id;
                option.textContent = student.full_name;
                studentSelectDropdown.appendChild(option);
            });
        } else {
            studentSelectDropdown.innerHTML = '<option value="">Không có học sinh nào</option>';
        }
    }

    // Gán sự kiện 'change' cho dropdown
    studentSelectDropdown.addEventListener('change', async function() {
        const studentId = this.value;
        if (!studentId) {
            aiAnalysisResultContainer.innerHTML = '<p class="placeholder-text">Vui lòng chọn một học sinh để bắt đầu phân tích...</p>';
            return;
        }

        // Hiển thị trạng thái đang tải
        aiAnalysisResultContainer.innerHTML = `
            <div class="text-center p-3">
                <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
                <span class="ms-2">AI đang phân tích...</span>
            </div>`;
        
        // Gọi API phân tích
        const result = await apiCall(`/api/students/${studentId}/analysis`);

        // Hiển thị kết quả
        if (result) {
            aiAnalysisResultContainer.innerHTML = `
                <div class="alert alert-info mt-2">
                    <h5 class="alert-heading">Kết luận: ${result.tendency}</h5>
                    <hr>
                    <p class="mb-0 small"><strong>Dữ liệu:</strong> ${result.reason}</p>
                </div>
            `;
        } else {
            aiAnalysisResultContainer.innerHTML = `
                <div class="alert alert-danger mt-2">
                    <strong>Lỗi:</strong> Không thể thực hiện phân tích.
                </div>
            `;
        }
    });

    // Hàm chính để tải tất cả dữ liệu thống kê
    async function loadStats() {
        // Tải dữ liệu cho dropdown trước
        await populateStudentDropdown();

        // Tải dữ liệu thống kê chung
        const data = await apiCall('/api/statistics');
        if (data) {
            renderTopStudents(data.top_students);
            renderSubjectAnalysis(data.subject_analysis);
            // Không render AI analysis chung nữa, để người dùng tự chọn
        } else {
            // Xử lý lỗi nếu không tải được
            topStudentsContainer.innerHTML = '<p class="placeholder-text text-danger">Lỗi tải dữ liệu.</p>';
            subjectAnalysisContainer.innerHTML = '<p class="placeholder-text text-danger">Lỗi tải dữ liệu.</p>';
        }
    }

    // Chạy hàm khi trang được tải
    loadStats();
});