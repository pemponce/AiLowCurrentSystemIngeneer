type Page = 'home' | 'login' | 'registration' | 'subscribe' | 'dashboard' | 'ai' | 'profile';

interface ProfilePageProps {
  onNavigate: (page: Page) => void;
}

export function ProfilePage({ onNavigate }: ProfilePageProps) {
  return (
    <div className="max-w-5xl mx-auto px-6 py-16">
      <div className="mb-8">
        <h1 className="mb-2">Личный кабинет</h1>
        <p className="text-muted-foreground">Управление вашим профилем и настройками</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          <div className="bg-card border border-border rounded-lg p-6">
            <div className="text-center mb-6">
              <div className="w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-4xl">👤</span>
              </div>
              <h3 className="mb-1">Иван Иванов</h3>
              <p className="text-sm text-muted-foreground">ivan@example.com</p>
            </div>
            
            <div className="space-y-2">
              <button className="w-full px-4 py-2 bg-primary text-primary-foreground rounded hover:opacity-90 text-left">
                ℹ️ Личная информация
              </button>
              <button className="w-full px-4 py-2 border border-border rounded hover:bg-accent text-left">
                🔐 Безопасность
              </button>
              <button className="w-full px-4 py-2 border border-border rounded hover:bg-accent text-left">
                💳 Подписка и оплата
              </button>
              <button className="w-full px-4 py-2 border border-border rounded hover:bg-accent text-left">
                🔔 Уведомления
              </button>
              <button className="w-full px-4 py-2 border border-border rounded hover:bg-accent text-left">
                📊 История проектов
              </button>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-2">
          {/* Личная информация */}
          <div className="bg-card border border-border rounded-lg p-6 mb-6">
            <h2 className="mb-6">Личная информация</h2>
            
            <form className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="firstname" className="block mb-2">
                    Имя
                  </label>
                  <input
                    id="firstname"
                    type="text"
                    defaultValue="Иван"
                    className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
                <div>
                  <label htmlFor="lastname" className="block mb-2">
                    Фамилия
                  </label>
                  <input
                    id="lastname"
                    type="text"
                    defaultValue="Иванов"
                    className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="email" className="block mb-2">
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  defaultValue="ivan@example.com"
                  className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              <div>
                <label htmlFor="phone" className="block mb-2">
                  Телефон
                </label>
                <input
                  id="phone"
                  type="tel"
                  defaultValue="+7 (999) 123-45-67"
                  className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              <div>
                <label htmlFor="company" className="block mb-2">
                  Компания
                </label>
                <input
                  id="company"
                  type="text"
                  defaultValue="ООО 'Электромонтаж'"
                  className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              <div>
                <label htmlFor="position" className="block mb-2">
                  Должность
                </label>
                <input
                  id="position"
                  type="text"
                  defaultValue="Ведущий инженер"
                  className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              <button
                type="submit"
                className="px-6 py-2 bg-primary text-primary-foreground rounded hover:opacity-90"
              >
                Сохранить изменения
              </button>
            </form>
          </div>

          {/* Текущая подписка */}
          <div className="bg-card border border-border rounded-lg p-6 mb-6">
            <h2 className="mb-4">Текущая подписка</h2>
            
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="mb-1">Тариф "Стандарт"</h3>
                <p className="text-muted-foreground">$79 / месяц</p>
              </div>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                Активна
              </span>
            </div>

            <div className="bg-muted p-4 rounded mb-4">
              <p className="text-sm mb-2">Следующее списание: 24 января 2025</p>
              <p className="text-sm">Осталось проектов в этом месяце: Неограниченно</p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => onNavigate('subscribe')}
                className="px-4 py-2 border border-border rounded hover:bg-accent"
              >
                Изменить план
              </button>
              <button className="px-4 py-2 border border-destructive text-destructive rounded hover:bg-destructive/10">
                Отменить подписку
              </button>
            </div>
          </div>

          {/* Статистика */}
          <div className="bg-card border border-border rounded-lg p-6">
            <h2 className="mb-4">Статистика использования</h2>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-muted p-4 rounded">
                <p className="text-muted-foreground text-sm mb-1">Всего проектов</p>
                <p className="text-2xl font-bold">60</p>
              </div>
              <div className="bg-muted p-4 rounded">
                <p className="text-muted-foreground text-sm mb-1">ИИ генераций</p>
                <p className="text-2xl font-bold">156</p>
              </div>
              <div className="bg-muted p-4 rounded">
                <p className="text-muted-foreground text-sm mb-1">В этом месяце</p>
                <p className="text-2xl font-bold">12</p>
              </div>
              <div className="bg-muted p-4 rounded">
                <p className="text-muted-foreground text-sm mb-1">Сэкономлено часов</p>
                <p className="text-2xl font-bold">240+</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
