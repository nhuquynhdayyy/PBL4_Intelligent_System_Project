/**
 * static/js/dashboard.js
 * Quản lý biểu đồ và số liệu cho Admin Dashboard
 */

let raceChart, radarChart;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        // 1. Tải dữ liệu tổng quát cho KPIs và biểu đồ Đường đua
        const res = await fetch('/api/admin/stats');
        const data = await res.json();

        // Cập nhật số liệu KPI (Fix lỗi hiển thị số 0)
        if (data.kpis) {
            if (document.getElementById('kpi-teachers'))
                document.getElementById('kpi-teachers').innerText = data.kpis.teachers;
            if (document.getElementById('kpi-classes'))
                document.getElementById('kpi-classes').innerText = data.kpis.classes;
            if (document.getElementById('kpi-students'))
                document.getElementById('kpi-students').innerText = data.kpis.students;
            if (document.getElementById('kpi-speeches'))
                document.getElementById('kpi-speeches').innerText = data.kpis.speeches;
        }

        // Khởi tạo biểu đồ Đường đua (Bar Chart ngang)
        const ctxRaceEl = document.getElementById('classRaceChart');
        if (ctxRaceEl) {
            const ctxRace = ctxRaceEl.getContext('2d');
            raceChart = new Chart(ctxRace, {
                type: 'bar',
                data: {
                    labels: data.race.labels,
                    datasets: [{
                        label: 'Tổng lượt tương tác',
                        data: data.race.values,
                        backgroundColor: 'rgba(78, 115, 223, 0.8)',
                        borderRadius: 10,
                        indexAxis: 'y', // Chế độ biểu đồ ngang
                    }]
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } }
                }
            });
        }

        // 2. Xử lý sự kiện khi chọn lớp từ Dropdown để xem Radar Chart
        const classSelect = document.getElementById('classSelect');
        if (classSelect) {
            classSelect.addEventListener('change', async function() {
                const classId = this.value;
                if (!classId) return;

                const resPer = await fetch(`/api/admin/class_performance/${classId}`);
                const perData = await resPer.json();

                updateRadarChart(perData);
            });
        }

    } catch (error) {
        console.error("Lỗi khi tải dữ liệu Dashboard:", error);
    }
});

/**
 * Hàm cập nhật hoặc khởi tạo biểu đồ Radar cho thế mạnh lớp
 * @param {Object} data Dữ liệu labels và values từ API
 */
function updateRadarChart(data) {
    const ctxRadarEl = document.getElementById('classRadarChart');
    if (!ctxRadarEl) return;

    const ctxRadar = ctxRadarEl.getContext('2d');
    
    if (radarChart) radarChart.destroy();

    radarChart = new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Lượt phát biểu theo môn',
                data: data.values,
                fill: true,
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                borderColor: '#10b981',
                pointBackgroundColor: '#10b981',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#10b981'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: { 
                    beginAtZero: true,
                    ticks: { stepSize: 1 }
                }
            }
        }
    });
}