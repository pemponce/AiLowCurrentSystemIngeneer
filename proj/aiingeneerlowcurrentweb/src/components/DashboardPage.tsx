type Page = 'home' | 'login' | 'registration' | 'subscribe' | 'dashboard';

interface DashboardPageProps {
  onNavigate: (page: Page) => void;
}

export function DashboardPage({ onNavigate }: DashboardPageProps) {
  return (
    <div className="max-w-7xl mx-auto px-6 py-16">
      <div className="mb-8">
        <h1 className="mb-2">Панель управления</h1>
        <p className="text-muted-foreground">С возвращением! Вот обзор ваших проектов.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid md:grid-cols-4 gap-6 mb-8">
        <div className="bg-card border border-border rounded-lg p-6">
          <p className="text-muted-foreground mb-2">Активные проекты</p>
          <p className="text-3xl font-bold">12</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-6">
          <p className="text-muted-foreground mb-2">Завершенные</p>
          <p className="text-3xl font-bold">48</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-6">
          <p className="text-muted-foreground mb-2">Члены команды</p>
          <p className="text-3xl font-bold">8</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-6">
          <p className="text-muted-foreground mb-2">ИИ анализов</p>
          <p className="text-3xl font-bold">156</p>
        </div>
      </div>

      {/* Recent Projects */}
      <div className="bg-card border border-border rounded-lg p-6 mb-8">
        <h2 className="mb-6">Последние проекты</h2>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 border border-border rounded hover:bg-accent transition-colors">
            <div>
              <h4>Коммерческое здание - Фаза 2</h4>
              <p className="text-muted-foreground text-sm">Проектирование системы пожарной сигнализации</p>
            </div>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">Активен</span>
          </div>
          <div className="flex items-center justify-between p-4 border border-border rounded hover:bg-accent transition-colors">
            <div>
              <h4>Жилой комплекс - Сеть</h4>
              <p className="text-muted-foreground text-sm">Инфраструктура передачи данных</p>
            </div>
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">На проверке</span>
          </div>
          <div className="flex items-center justify-between p-4 border border-border rounded hover:bg-accent transition-colors">
            <div>
              <h4>Офисная башня - Безопасность</h4>
              <p className="text-muted-foreground text-sm">Система контроля доступа</p>
            </div>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">Активен</span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid md:grid-cols-3 gap-6">
        <button className="bg-card border border-border rounded-lg p-6 hover:bg-accent transition-colors text-left">
          <div className="text-2xl mb-2">➕</div>
          <h3 className="mb-1">Новый проект</h3>
          <p className="text-muted-foreground text-sm">Начать новый инженерный проект</p>
        </button>
        <button className="bg-card border border-border rounded-lg p-6 hover:bg-accent transition-colors text-left">
          <div className="text-2xl mb-2">🤖</div>
          <h3 className="mb-1">ИИ анализ</h3>
          <p className="text-muted-foreground text-sm">Запустить ИИ анализ существующего проекта</p>
        </button>
        <button 
          onClick={() => onNavigate('subscribe')}
          className="bg-card border border-border rounded-lg p-6 hover:bg-accent transition-colors text-left"
        >
          <div className="text-2xl mb-2">⬆️</div>
          <h3 className="mb-1">Улучшить тариф</h3>
          <p className="text-muted-foreground text-sm">Получите больше функций и возможностей</p>
        </button>
      </div>
    </div>
  );
}
