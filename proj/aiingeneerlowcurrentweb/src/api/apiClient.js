// src/apiClient.js
import axios from 'axios';

// ---- токен ----
const AUTH_TOKEN_KEY = 'auth_token';

export const getAuthToken = () => {
    return window.localStorage.getItem(AUTH_TOKEN_KEY);
};

export const setAuthToken = (t) => {
    window.localStorage.setItem(AUTH_TOKEN_KEY, t);
};

export const clearAuthToken = () => {
    window.localStorage.removeItem(AUTH_TOKEN_KEY);
};

// ---- axios instance ----
const reminderApi = axios.create({
    baseURL: 'http://localhost:8080', // или '/api' если настроен прокси
    timeout: 15000,
    withCredentials: true, // включай, если у тебя cookie-сессии
});

// ---- интерцепторы ----
reminderApi.interceptors.request.use(
    (config) => {
        const token = getAuthToken();
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

reminderApi.interceptors.response.use(
    (res) => res,
    (err) => {
        if (err?.response?.status === 401) {
            // clearAuthToken();
            // location.href = '/login.html';
        }
        return Promise.reject(err);
    }
);

// ---- утилиты ----
const toForm = (obj) => {
    const fd = new FormData();
    Object.entries(obj || {}).forEach(([k, v]) => {
        if (v !== undefined && v !== null) fd.append(k, v);
    });
    return fd;
};

// ==================== AUTH ====================

// регистрация (публичный эндпоинт)
export const register = async (fullName, email, password) => {
    try {
        const response = await reminderApi.post('/api/public/register', {
            fullName,
            email,
            password,
        });
        return response.data;
    } catch (error) {
        throw error;
    }
};

// вход (ожидаем {token} либо cookie; правь под свой ответ)
export const login = async (email, password) => {
    try {
        const response = await reminderApi.post('/api/public/login', {
            email,
            password,
        });
        const { token } = response.data;
        if (token) setAuthToken(token);
        return response.data;
    } catch (error) {
        throw error;
    }
};

export const logout = () => {
    try {
        clearAuthToken();
        // если используешь logout на бэке:
        // return await reminderApi.post('/api/auth/logout');
    } catch (error) {
        throw error;
    }
};

// ==================== PROJECTS ====================

export const getProjects = async (params) => {
    try {
        const response = await reminderApi.get('/api/projects', { params });
        return response.data;
    } catch (error) {
        throw error;
    }
};

export const getProjectById = async (projectId) => {
    try {
        const response = await reminderApi.get(`/api/projects/${projectId}`);
        return response.data;
    } catch (error) {
        throw error;
    }
};

export const createProject = async (payload) => {
    try {
        const response = await reminderApi.post('/api/projects', payload);
        return response.data;
    } catch (error) {
        throw error;
    }
};

export const updateProject = async (projectId, payload) => {
    try {
        const response = await reminderApi.put(`/api/projects/${projectId}`, payload);
        return response.data;
    } catch (error) {
        throw error;
    }
};

export const deleteProject = async (projectId) => {
    try {
        const response = await reminderApi.delete(`/api/projects/${projectId}`);
        return response.data;
    } catch (error) {
        throw error;
    }
};

// файлы в проект (мультипарт на твой контроллер)
export const uploadProjectFile = async (projectId, file, extra = {}) => {
    try {
        const form = toForm({ file, ...extra });
        const response = await reminderApi.post(
            `/api/projects/${projectId}/upload`,
            form,
            { headers: { 'Content-Type': 'multipart/form-data' } }
        );
        return response.data;
    } catch (error) {
        throw error;
    }
};

// если используешь pre-signed URL (S3/MinIO)
export const uploadViaPresignedUrl = async (url, file, method = 'PUT') => {
    try {
        const res = await fetch(url, { method, body: file });
        if (!res.ok) {
            throw new Error(`Upload failed: ${res.status} ${res.statusText}`);
        }
        return true;
    } catch (error) {
        throw error;
    }
};

// ==================== JOBS (если нужно) ====================

export const getJobs = async (params) => {
    try {
        const response = await reminderApi.get('/api/jobs', { params });
        return response.data;
    } catch (error) {
        throw error;
    }
};

export const createJob = async (payload) => {
    try {
        const response = await reminderApi.post('/api/jobs', payload);
        return response.data;
    } catch (error) {
        throw error;
    }
};

export const getJobById = async (jobId) => {
    try {
        const response = await reminderApi.get(`/api/jobs/${jobId}`);
        return response.data;
    } catch (error) {
        throw error;
    }
};

// ==================== USERS / ROLES (пример) ====================

export const me = async () => {
    try {
        const response = await reminderApi.get('/api/me');
        return response.data;
    } catch (error) {
        throw error;
    }
};

export const getRoles = async () => {
    try {
        const response = await reminderApi.get('/api/roles');
        return response.data;
    } catch (error) {
        throw error;
    }
};

// ==================== экспорт по умолчанию ====================

export default reminderApi;
