// static/js/shared.js

/**
 * Hàm gọi API chung, xử lý GET, POST, PUT, DELETE và các lỗi một cách tập trung.
 * @param {string} endpoint - Đường dẫn API (ví dụ: '/api/students').
 * @param {string} method - Phương thức HTTP (ví dụ: 'GET', 'POST').
 * @param {object|null} body - Dữ liệu cần gửi đi cho phương thức POST/PUT.
 * @returns {Promise<object|null>} Dữ liệu JSON trả về từ API hoặc null nếu có lỗi.
 */
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : null
    };
    try {
        const response = await fetch(endpoint, options);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || 'Có lỗi không xác định xảy ra từ server.');
        }
        return data;
    } catch (error) {
        console.error(`Lỗi khi gọi API ${endpoint}:`, error);
        alert(`Lỗi: ${error.message}`);
        return null;
    }
}

/**
 * Mở một modal dựa trên ID của nó.
 * @param {string} modalId - ID của modal cần mở (ví dụ: 'subjectModal').
 */
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add('show');
}

/**
 * Đóng một modal dựa trên ID của nó.
 * @param {string} modalId - ID của modal cần đóng.
 */
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove('show');
}

// Gán sự kiện đóng cho tất cả các nút đóng modal trên toàn trang một lần duy nhất
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.modal-close, .modal button[type="button"].btn-danger').forEach(el => {
        el.addEventListener('click', () => {
            // Tìm modal cha gần nhất và đóng nó
            const modalToClose = el.closest('.modal');
            if (modalToClose) {
                closeModal(modalToClose.id);
            }
        });
    });
});