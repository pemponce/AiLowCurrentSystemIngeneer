// src/App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Login from "./component/login";
import Register from "./component/register";
import ProjectsPage from "./component/projects";

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Navigate to="/projects" replace />} />
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route path="/projects" element={<ProjectsPage />} />
                <Route path="*" element={<div>404</div>} />
            </Routes>
        </BrowserRouter>
    );
}

export default App;
