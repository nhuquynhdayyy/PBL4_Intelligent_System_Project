// static/js/dashboard.js
document.addEventListener('DOMContentLoaded', async function() {
    const statsGrid = document.getElementById('statsGrid');
    const activityFeed = document.getElementById('activityFeed');
    const totalStudentsHeader = document.getElementById('totalStudentsHeader');

    // Gọi API để lấy dữ liệu thống kê cho dashboard
    const data = await apiCall('/api/dashboard_stats');
    if (!data) {
        statsGrid.innerHTML = '<p>Không thể tải dữ liệu thống kê.</p>';
        return;
    }

    // Render các thẻ thống kê chính
    statsGrid.innerHTML = `
        <div class="stat-card"><h3>${data.stats.subjects}</h3><p>Môn học</p></div>
        <div class="stat-card"><h3>${data.stats.students}</h3><p>Học sinh</p></div>
        <div class="stat-card"><h3>${data.stats.sessions}</h3><p>Buổi học</p></div>
        <div class="stat-card"><h3>${data.stats.speeches}</h3><p>Phát biểu</p></div>
    `;
    
    // Cập nhật tổng số học sinh trên header
    if (totalStudentsHeader) {
        totalStudentsHeader.textContent = data.stats.students;
    }

    // Render các hoạt động gần đây
    if (data.recent_activity && data.recent_activity.length > 0) {
        activityFeed.innerHTML = data.recent_activity.map(act => `
            <div class="card">
                <strong>${act.subject_name} - Buổi học #${act.session_number}</strong>
                <p style="color: #666; font-size: 14px;">Ngày ${act.end_time} - ${act.speech_count} phát biểu được ghi nhận</p>
            </div>
        `).join('');
    } else {
        activityFeed.innerHTML = '<p>Chưa có hoạt động nào gần đây.</p>';
    }
});