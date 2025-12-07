// src/components/auth/Register.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { register as apiRegister } from '../api/apiClient';

function Register({ onRegistered }) {
    const [fullName, setFullName] = useState('');
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
            await apiRegister(fullName, email, password);
            if (typeof onRegistered === 'function') onRegistered({ fullName, email });
            // после успешной регистрации — на /login (как у тебя редиректит HTML-версия)
            navigate('/login?registered=1');
        } catch (err) {
            setError(err?.response?.data?.message || err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-container">
            <h2>Регистрация</h2>
            <form onSubmit={handleSubmit} className="auth-form">
                <input
                    type="text"
                    placeholder="Полное имя"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="auth-input"
                    required
                />
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
                    {loading ? 'Отправляю…' : 'Зарегистрироваться'}
                </button>
            </form>
            {error && <div className="auth-error">Ошибка: {error}</div>}
        </div>
    );
}

export default Register;
