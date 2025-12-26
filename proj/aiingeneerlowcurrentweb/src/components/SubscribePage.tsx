type Page = 'home' | 'login' | 'registration' | 'subscribe' | 'dashboard';

interface SubscribePageProps {
  onNavigate: (page: Page) => void;
}

export function SubscribePage({ onNavigate }: SubscribePageProps) {
  return (
    <div className="max-w-7xl mx-auto px-6 py-16">
      <div className="text-center mb-12">
        <h1 className="mb-4">Выберите свой тариф</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Выберите идеальный план подписки для ваших инженерных потребностей
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        {/* Simple Plan */}
        <div className="bg-card border border-border rounded-lg p-8 flex flex-col">
          <h3 className="mb-2">Простой</h3>
          <div className="mb-6">
            <span className="text-4xl font-bold">$29</span>
            <span className="text-muted-foreground">/месяц</span>
          </div>
          <p className="text-muted-foreground mb-6">
            Идеально для индивидуальных инженеров и небольших проектов
          </p>
          
          <ul className="space-y-3 mb-8 flex-grow">
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>До 5 проектов</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Базовый анализ ИИ</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Поддержка по электронной почте</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Стандартные шаблоны</span>
            </li>
          </ul>
          
          <button
            onClick={() => onNavigate('registration')}
            className="w-full px-6 py-3 border border-border rounded hover:bg-accent"
          >
            Начать
          </button>
        </div>

        {/* Standard Plan */}
        <div className="bg-card border-2 border-primary rounded-lg p-8 flex flex-col relative">
          <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-primary text-primary-foreground px-4 py-1 rounded-full text-sm">
            Популярный
          </div>
          <h3 className="mb-2">Стандарт</h3>
          <div className="mb-6">
            <span className="text-4xl font-bold">$79</span>
            <span className="text-muted-foreground">/месяц</span>
          </div>
          <p className="text-muted-foreground mb-6">
            Идеально для профессиональных инженеров и средних команд
          </p>
          
          <ul className="space-y-3 mb-8 flex-grow">
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Неограниченное количество проектов</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Расширенный анализ ИИ</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Приоритетная поддержка по email и чату</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Премиум шаблоны</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Командная работа (до 10 человек)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Доступ к API</span>
            </li>
          </ul>
          
          <button
            onClick={() => onNavigate('registration')}
            className="w-full px-6 py-3 bg-primary text-primary-foreground rounded hover:opacity-90"
          >
            Начать
          </button>
        </div>

        {/* Business Plan */}
        <div className="bg-card border border-border rounded-lg p-8 flex flex-col">
          <h3 className="mb-2">Бизнес</h3>
          <div className="mb-6">
            <span className="text-4xl font-bold">$199</span>
            <span className="text-muted-foreground">/месяц</span>
          </div>
          <p className="text-muted-foreground mb-6">
            Полное решение для больших команд и предприятий
          </p>
          
          <ul className="space-y-3 mb-8 flex-grow">
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Неограниченные проекты и пользователи</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Индивидуальное обучение моделей ИИ</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Поддержка 24/7 по телефону и в чате</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Индивидуальные шаблоны и брендинг</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Расширенное управление командой</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Полный доступ к API</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Персональный менеджер аккаунта</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <span>Гарантия SLA</span>
            </li>
          </ul>
          
          <button
            onClick={() => onNavigate('registration')}
            className="w-full px-6 py-3 border border-border rounded hover:bg-accent"
          >
            Связаться с отделом продаж
          </button>
        </div>
      </div>

      <div className="mt-12 text-center text-muted-foreground">
        <p>Все тарифы включают 14-дневный бесплатный пробный период. Кредитная карта не требуется.</p>
      </div>
    </div>
  );
}
