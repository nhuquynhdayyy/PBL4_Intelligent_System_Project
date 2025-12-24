document.addEventListener('DOMContentLoaded', function() {

    // --- BIẾN TOÀN CỤC & THỂ HIỆN BIỂU ĐỒ ---
    let charts = {
        topStudents: null,
        subjectDistribution: null,
        studentTrend: null,
        studentSubjectRadar: null
    };

    // --- LẤY CÁC PHẦN TỬ DOM ---
    const kpiContainer = document.getElementById('kpi-container');
    const fullRankingTableBody = document.getElementById('fullRankingTableBody');
    const subjectDetailsList = document.getElementById('subjectDetailsList');
    const studentSelectDropdown = document.getElementById('student-select-for-analysis');
    const studentDashboardContainer = document.getElementById('studentDashboardContainer');
    const studentKpiContainer = document.getElementById('student-kpi-container');
    const aiInsightBox = document.getElementById('aiInsightBox');

    // --- HÀM HELPER & RENDER ---

    /**
     * Hàm helper để gọi API an toàn.
     */
    async function apiCall(endpoint) {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error(`Lỗi khi gọi API đến ${endpoint}:`, error);
            return null;
        }
    }

    /**
     * Hủy một biểu đồ nếu nó đã tồn tại.
     * @param {string} chartName - Tên của biểu đồ trong đối tượng 'charts'.
     */
    function destroyChart(chartName) {
        if (charts[chartName]) {
            charts[chartName].destroy();
            charts[chartName] = null;
        }
    }
    
    // -- RENDER CÁC THÀNH PHẦN CHUNG --

    function renderKpiCards(kpis) {
        kpiContainer.innerHTML = `
            <div class="kpi-card"><div class="value">${kpis.total_sessions || 0}</div><div class="label">Buổi học đã diễn ra</div></div>
            <div class="kpi-card"><div class="value">${kpis.total_speeches || 0}</div><div class="label">Lượt phát biểu</div></div>
            <div class="kpi-card"><div class="value">${kpis.total_students || 0}</div><div class="label">Học sinh tham gia</div></div>
            <div class="kpi-card"><div class="value highlight">${kpis.most_active_student || 'N/A'}</div><div class="label">Tích cực nhất</div></div>
        `;
    }

    function renderTopStudentsChart(students) {
        const ctx = document.getElementById('topStudentsChart').getContext('2d');
        destroyChart('topStudents');
        charts.topStudents = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: students.slice(0, 5).map(s => s.name),
                datasets: [{
                    label: 'Số lần phát biểu',
                    data: students.slice(0, 5).map(s => s.speeches),
                    backgroundColor: '#007bff',
                    borderRadius: 5
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    }

    function renderFullRankingTable(students) {
        fullRankingTableBody.innerHTML = students.map((s, index) => `
            <tr>
                <td><span class="rank-badge rank-${index + 1}">${index + 1}</span></td>
                <td>${s.name}</td>
                <td>${s.speeches}</td>
            </tr>
        `).join('');
    }
    
    function renderSubjectDistributionChart(subjects) {
        const ctx = document.getElementById('subjectDistributionChart').getContext('2d');
        destroyChart('subjectDistribution');
        charts.subjectDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: subjects.map(s => s.name),
                datasets: [{
                    data: subjects.map(s => s.total_speeches),
                    backgroundColor: ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1'],
                    hoverOffset: 4
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right' } } }
        });
    }

    function renderSubjectDetails(subjects) {
        subjectDetailsList.innerHTML = subjects.map(s => `
            <div class="list-item">
                <div class="item-main">
                    <span class="item-icon">${s.icon || '📚'}</span>
                    <span class="item-name">${s.name}</span>
                </div>
                <div class="item-details">
                    <span>Tích cực nhất: <strong>${s.top_student_name || 'N/A'}</strong></span>
                    <span>(${s.top_student_speeches || 0} lần)</span>
                </div>
            </div>
        `).join('');
    }

    // -- RENDER CÁC THÀNH PHẦN DASHBOARD CÁ NHÂN --

    function renderStudentKpis(data) {
        studentKpiContainer.innerHTML = `
            <div class="kpi-card nested"><div class="value">${data.rank}</div><div class="label">Thứ hạng</div></div>
            <div class="kpi-card nested"><div class="value">${data.total_speeches}</div><div class="label">Tổng phát biểu</div></div>
            <div class="kpi-card nested"><div class="value highlight">${data.best_subject || 'N/A'}</div><div class="label">Môn học thế mạnh</div></div>
        `;
    }

    function renderStudentTrendChart(trend) {
        const ctx = document.getElementById('studentTrendChart').getContext('2d');
        destroyChart('studentTrend');
        charts.studentTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: trend.labels,
                datasets: [{
                    label: 'Số lần phát biểu',
                    data: trend.data,
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    }

    function renderStudentSubjectRadarChart(radar) {
        const ctx = document.getElementById('studentSubjectRadarChart').getContext('2d');
        destroyChart('studentSubjectRadar');
        charts.studentSubjectRadar = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: radar.labels,
                datasets: [{
                    label: 'Mức độ tích cực',
                    data: radar.data,
                    backgroundColor: 'rgba(255, 193, 7, 0.2)',
                    borderColor: 'rgba(255, 193, 7, 1)',
                    pointBackgroundColor: 'rgba(255, 193, 7, 1)',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { r: { beginAtZero: true, ticks: { stepSize: 1 } } },
                plugins: { legend: { display: false } }
            }
        });
    }

    function renderAiInsight(insight) {
        aiInsightBox.innerHTML = `
            <h4><i class="fas fa-lightbulb text-warning"></i> AI Insight</h4>
            <p><strong>Xu hướng chung:</strong> <span class="text-primary">${insight.tendency}</span></p>
            <p class="small text-muted"><strong>Phân tích:</strong> ${insight.reason}</p>
        `;
    }

    // --- CÁC HÀM ĐIỀU KHIỂN CHÍNH ---

    async function populateStudentDropdown() {
        studentSelectDropdown.disabled = true;
        const students = await apiCall('/api/students');
        if (students && students.length > 0) {
            studentSelectDropdown.innerHTML = '<option value="">-- Chọn học sinh --</option>' + 
                students.map(s => `<option value="${s.id}">${s.full_name}</option>`).join('');
        } else {
            studentSelectDropdown.innerHTML = '<option value="">Không có dữ liệu</option>';
        }
        studentSelectDropdown.disabled = false;
    }

    async function handleStudentSelection() {
        const studentId = this.value;
        if (!studentId) {
            studentDashboardContainer.classList.add('hidden');
            return;
        }

        studentDashboardContainer.classList.remove('hidden');
        aiInsightBox.innerHTML = `<div class="loading-spinner"></div><p>AI đang phân tích...</p>`;
        
        const data = await apiCall(`/api/students/${studentId}/analysis`);
        if (data) {
            renderStudentKpis(data.kpis);
            renderStudentTrendChart(data.trend);
            renderStudentSubjectRadarChart(data.radar);
            renderAiInsight(data.insight);
        } else {
            studentDashboardContainer.innerHTML = '<p class="text-danger">Lỗi tải dữ liệu phân tích cho học sinh này.</p>';
        }
    }

    async function initializePage() {
        populateStudentDropdown();
        const data = await apiCall('/api/statistics');
        if (data) {
            renderKpiCards(data.kpis);
            renderTopStudentsChart(data.all_students_ranking);
            renderFullRankingTable(data.all_students_ranking);
            renderSubjectDistributionChart(data.subject_analysis);
            renderSubjectDetails(data.subject_analysis);
        } else {
            document.body.innerHTML = '<p class="text-danger text-center mt-5">Không thể tải dữ liệu dashboard. Vui lòng kiểm tra kết nối và API.</p>';
        }
    }

    // --- GÁN SỰ KIỆN VÀ KHỞI CHẠY ---
    studentSelectDropdown.addEventListener('change', handleStudentSelection);
    initializePage();
});