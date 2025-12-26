import { useState } from 'react';
import { HomePage } from './components/HomePage';
import { LoginPage } from './components/LoginPage';
import { RegistrationPage } from './components/RegistrationPage';
import { SubscribePage } from './components/SubscribePage';
import { DashboardPage } from './components/DashboardPage';

type Page = 'home' | 'login' | 'registration' | 'subscribe' | 'dashboard';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home');

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="bg-primary text-primary-foreground border-b border-border">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary-foreground rounded flex items-center justify-center">
                <span className="text-primary font-bold">AI</span>
              </div>
              <div>
                <h1 className="font-bold">AILCE</h1>
                <p className="text-xs opacity-80">Инженер слаботочных систем на базе ИИ</p>
              </div>
            </div>
            
            <div className="flex gap-6">
              <button 
                onClick={() => setCurrentPage('home')}
                className={`hover:opacity-80 transition-opacity ${currentPage === 'home' ? 'underline' : ''}`}
              >
                Главная
              </button>
              <button 
                onClick={() => setCurrentPage('subscribe')}
                className={`hover:opacity-80 transition-opacity ${currentPage === 'subscribe' ? 'underline' : ''}`}
              >
                Подписка
              </button>
              <button 
                onClick={() => setCurrentPage('dashboard')}
                className={`hover:opacity-80 transition-opacity ${currentPage === 'dashboard' ? 'underline' : ''}`}
              >
                Панель управления
              </button>
              <button 
                onClick={() => setCurrentPage('login')}
                className={`hover:opacity-80 transition-opacity ${currentPage === 'login' ? 'underline' : ''}`}
              >
                Вход
              </button>
              <button 
                onClick={() => setCurrentPage('registration')}
                className={`px-4 py-2 bg-primary-foreground text-primary rounded hover:opacity-90 transition-opacity`}
              >
                Регистрация
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Page Content */}
      <main>
        {currentPage === 'home' && <HomePage onNavigate={setCurrentPage} />}
        {currentPage === 'login' && <LoginPage onNavigate={setCurrentPage} />}
        {currentPage === 'registration' && <RegistrationPage onNavigate={setCurrentPage} />}
        {currentPage === 'subscribe' && <SubscribePage onNavigate={setCurrentPage} />}
        {currentPage === 'dashboard' && <DashboardPage onNavigate={setCurrentPage} />}
      </main>

      {/* Footer */}
      <footer className="bg-muted text-muted-foreground border-t border-border mt-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="text-center">
            <p>© 2024 AILCE - Инженер слаботочных систем на базе ИИ. Все права защищены.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}