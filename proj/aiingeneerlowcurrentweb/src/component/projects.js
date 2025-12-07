// src/components/project/ProjectsPage.jsx
import React, { useEffect, useState } from 'react';
import {getProjects} from "../api/apiClient";

function ProjectsPage({
                          query = { page: 0, size: 20, sort: 'createdAt,desc' },
                      }) {
    const [data, setData] = useState(null);   // сюда прилетит page
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        let cancelled = false;

        (async () => {
            try {
                const page = await getProjects(query);
                if (!cancelled) setData(page);
                // раньше просто console.log(page)
                console.log('Projects page:', page);
            } catch (e) {
                if (!cancelled) setError(e?.response?.data?.message || e.message);
                console.error(e);
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();

        return () => { cancelled = true; };
    }, [getProjects, query]);

    if (loading) return <div>Загрузка…</div>;
    if (error)   return <div className="error">Ошибка: {error}</div>;

    // подстрой под свою структуру page (content/totalElements и т.п.)
    const items = data?.content ?? data?.items ?? data ?? [];

    return (
        <div className="projects-page">
            <h2>Проекты</h2>

            {Array.isArray(items) && items.length > 0 ? (
                <ul className="project-list">
                    {items.map((p, i) => (
                        <li key={p.id ?? i} className="project-item">
                            <strong>{p.name ?? p.projectName ?? `Проект ${i + 1}`}</strong>
                            {p.createdAt && <span> · {new Date(p.createdAt).toLocaleString()}</span>}
                        </li>
                    ))}
                </ul>
            ) : (
                <div>Проектов нет</div>
            )}

            {/* отладочный вывод всего ответа */}
            <pre style={{ marginTop: 16, maxWidth: '100%', overflowX: 'auto' }}>
        {JSON.stringify(data, null, 2)}
      </pre>
        </div>
    );
}

export default ProjectsPage;
