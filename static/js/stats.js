// static/js/stats.js
document.addEventListener('DOMContentLoaded', function() {
    const topStudentsContainer = document.getElementById('topStudentsRanking');
    const subjectAnalysisContainer = document.getElementById('subjectAnalysis');
    const aiAnalysisContainer = document.getElementById('aiAnalysisResult');

    // Hàm render bảng xếp hạng học sinh
    function renderTopStudents(students) {
        if (!students || students.length === 0) {
            topStudentsContainer.innerHTML = '<p class="placeholder-text">Chưa có dữ liệu để xếp hạng.</p>';
            return;
        }

        const rankColors = ['rank-1', 'rank-2', 'rank-3'];

        const studentsHtml = students.map((student, index) => {
            const rankClass = rankColors[index] || 'rank-other';
            return `
                <li>
                    <span class="rank-badge ${rankClass}">${index + 1}</span>
                    <span class="student-name">${student.name}</span>
                    <span class="student-score">${student.speeches}</span>
                </li>
            `;
        }).join('');

        topStudentsContainer.innerHTML = `<ol>${studentsHtml}</ol>`;
    }

    // Hàm render phân tích môn học
    function renderSubjectAnalysis(subjects) {
        if (!subjects || subjects.length === 0) {
            subjectAnalysisContainer.innerHTML = '<p class="placeholder-text">Chưa có dữ liệu môn học.</p>';
            return;
        }

        const subjectsHtml = subjects.map(subject => {
            const topStudentText = subject.top_student_name === 'Chưa có'
                ? 'Chưa có (0 phát biểu)'
                : `${subject.top_student_name} (${subject.top_student_speeches} phát biểu)`;

            return `
                <div class="subject-item">
                    <h4>${subject.icon || '📚'} ${subject.name} - ${subject.session_count} buổi học</h4>
                    <p>Học sinh tích cực nhất: <strong>${topStudentText}</strong></p>
                    <button class="btn btn-secondary">Xem chi tiết</button>
                </div>
            `;
        }).join('');

        subjectAnalysisContainer.innerHTML = subjectsHtml;
    }

    // Hàm render phân tích của AI (ví dụ đơn giản)
    function renderAiAnalysis(rawData) {
        // Đây là một ví dụ phân tích rất đơn giản.
        // Bạn có thể thay thế bằng một mô hình phức tạp hơn.
        if (!rawData || Object.keys(rawData).length === 0) {
            aiAnalysisContainer.textContent = 'Không đủ dữ liệu để phân tích.';
            return;
        }

        let maxSpeeches = 0;
        let topStudent = '';
        let engagedSubjects = new Set();
        
        for (const student in rawData) {
            let total = 0;
            for (const subject in rawData[student]) {
                total += rawData[student][subject];
                if (rawData[student][subject] > 0) {
                    engagedSubjects.add(subject);
                }
            }
            if (total > maxSpeeches) {
                maxSpeeches = total;
                topStudent = student;
            }
        }

        if (topStudent) {
            aiAnalysisContainer.innerHTML = `
                Dựa trên dữ liệu, <strong>${topStudent}</strong> là học sinh năng nổ nhất toàn diện. 
                Các môn học đang có sự tương tác tốt bao gồm: 
                <strong>${Array.from(engagedSubjects).join(', ')}</strong>. 
                Cần khuyến khích thêm ở các môn còn lại.
            `;
        } else {
             aiAnalysisContainer.textContent = 'Chưa có hoạt động nào được ghi nhận để phân tích.';
        }
    }

    // Hàm chính để fetch và hiển thị dữ liệu
    async function loadStats() {
        try {
            const response = await fetch('/api/statistics');
            if (!response.ok) {
                throw new Error('Không thể tải dữ liệu thống kê.');
            }
            const data = await response.json();

            renderTopStudents(data.top_students);
            renderSubjectAnalysis(data.subject_analysis);
            renderAiAnalysis(data.student_trends_raw_data);

        } catch (error) {
            console.error('Lỗi:', error);
            topStudentsContainer.innerHTML = '<p class="placeholder-text text-danger">Lỗi khi tải dữ liệu.</p>';
            subjectAnalysisContainer.innerHTML = '<p class="placeholder-text text-danger">Lỗi khi tải dữ liệu.</p>';
            aiAnalysisContainer.textContent = 'Lỗi khi tải dữ liệu.';
        }
    }

    // Chạy hàm khi trang được tải
    loadStats();
});