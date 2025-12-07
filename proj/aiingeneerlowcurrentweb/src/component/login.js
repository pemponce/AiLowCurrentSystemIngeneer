// src/components/auth/Login.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login as apiLogin } from '../api/apiClient';

function Login({ onLogin }) {
    const [email,    setEmail]    = useState('');
    const [password, setPassword] = useState('');
    const [loading,  setLoading]  = useState(false);
    const [error,    setError]    = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            const res = await apiLogin(email, password); // ждёт /api/public/login на бэке
            if (typeof onLogin === 'function') onLogin(res);
            navigate('/projects');
        } catch (err) {
            setError(err?.response?.data?.message || err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-container">
            <h2>Вход</h2>
            <form onSubmit={handleSubmit} className="auth-form">
                <input
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="auth-input"
                    required
                />
                <input
                    type="password"
                    placeholder="Пароль"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="auth-input"
                    required
                />
                <button type="submit" className="auth-button" disabled={loading}>
                    {loading ? 'Вхожу…' : 'Войти'}
                </button>
            </form>
            {error && <div className="auth-error">Ошибка: {error}</div>}
        </div>
    );
}

export default Login;
